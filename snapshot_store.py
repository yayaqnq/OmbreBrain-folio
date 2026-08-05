# ============================================================
# Module: Per-bucket snapshot store (snapshot_store.py)
# 模块：单桶快照存储 —— "改错了能撤回"
#
# 每次实质性修改记忆前，先把改动前的样子存一份到 .snapshots/。
# 文件格式沿用数据目录里已有的历史快照（2026-07 某个版本写下的）：
#   .snapshots/{bucket_id}_{unix_ts}.json
#   {"bucket_id":..., "timestamp": ISO, "metadata": {...}, "content": "..."}
# 沿用而不是另起一套，是为了让那些孤儿快照直接变成可用历史。
#
# 设计红线：本模块的任何失败都不允许影响记忆写入本身。
# 调用方必须用 try/except 包住 —— 快照丢了只是少一次后悔药，
# 记忆丢了是不可逆的。
# ============================================================

import json
import logging
import os
import re
from datetime import datetime, timezone

from utils import atomic_write_text, safe_path

logger = logging.getLogger("ombre_brain.snapshot")

SNAPSHOTS_DIRNAME = ".snapshots"
DEFAULT_KEEP = 20

# {bucket_id}_{unix_ts}.json —— bucket_id 是 12 位 hex，ts 是 10 位左右整数
_SNAPSHOT_RE = re.compile(r"^(?P<bid>[A-Za-z0-9_-]+)_(?P<ts>\d+)\.json$")

# 触发快照的"实质性"字段。update() 每次都会无条件刷 last_active，
# 不做过滤的话，任何一次浮现命中都会存一版，几天就把目录塞满。
SUBSTANTIVE_FIELDS = frozenset({
    "content", "name", "tags", "domain", "summary", "importance",
    "meaning", "meaning_append", "meaning_replace", "why_remembered",
    "type", "event_time", "raw_source", "source_excerpt", "parent_id",
})


def snapshots_dir(base_dir: str) -> str:
    """返回 .snapshots 目录路径（不创建）。"""
    return os.path.join(base_dir, SNAPSHOTS_DIRNAME)


def is_substantive(kwargs: dict) -> bool:
    """这次 update 是否值得存快照。"""
    return bool(SUBSTANTIVE_FIELDS & set(kwargs or {}))


def _ts_to_iso(ts: int) -> str:
    """unix 秒 → naive ISO（跟目录里既有快照的写法保持一致）。"""
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None).isoformat()


def write_snapshot(base_dir: str, bucket_id: str, metadata: dict, content: str,
                   keep: int = DEFAULT_KEEP) -> str | None:
    """存一版快照，返回文件路径；失败返回 None（绝不抛）。"""
    try:
        bucket_id = str(bucket_id or "").strip()
        if not bucket_id:
            return None
        root = snapshots_dir(base_dir)
        os.makedirs(root, exist_ok=True)

        ts = int(datetime.now(tz=timezone.utc).timestamp())
        # 同一秒内重复改，避免互相覆盖
        for bump in range(10):
            candidate = safe_path(root, f"{bucket_id}_{ts + bump}.json")
            if not candidate.exists():
                break
        else:
            return None

        payload = {
            "bucket_id": bucket_id,
            "timestamp": _ts_to_iso(ts),
            "metadata": _jsonable(metadata),
            "content": content or "",
        }
        atomic_write_text(candidate, json.dumps(payload, ensure_ascii=False, indent=2))
        prune_snapshots(base_dir, bucket_id, keep=keep)
        return str(candidate)
    except Exception as e:
        logger.warning(f"写快照失败（已忽略，不影响记忆写入）: {bucket_id}: {e}")
        return None


def _jsonable(meta):
    """frontmatter 里可能有 datetime 等非 JSON 类型，统统降级成字符串。"""
    if isinstance(meta, dict):
        return {str(k): _jsonable(v) for k, v in meta.items()}
    if isinstance(meta, (list, tuple)):
        return [_jsonable(v) for v in meta]
    if isinstance(meta, (str, int, float, bool)) or meta is None:
        return meta
    return str(meta)


def list_snapshots(base_dir: str, bucket_id: str) -> list[dict]:
    """列出某个桶的所有快照，最新在前。每项含 ts / timestamp / path / preview。"""
    root = snapshots_dir(base_dir)
    if not os.path.isdir(root):
        return []
    bucket_id = str(bucket_id or "").strip()
    out = []
    try:
        names = os.listdir(root)
    except OSError:
        return []
    for name in names:
        m = _SNAPSHOT_RE.match(name)
        if not m or m.group("bid") != bucket_id:
            continue
        ts = int(m.group("ts"))
        path = os.path.join(root, name)
        entry = {"ts": ts, "timestamp": _ts_to_iso(ts), "path": path,
                 "preview": "", "name": ""}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entry["timestamp"] = str(data.get("timestamp") or entry["timestamp"])
            entry["preview"] = str(data.get("content") or "").strip().replace("\n", " ")[:60]
            entry["name"] = str((data.get("metadata") or {}).get("name") or "")
        except Exception:
            # 坏文件不该让整个列表挂掉，保留条目但内容留空
            pass
        out.append(entry)
    out.sort(key=lambda e: e["ts"], reverse=True)
    return out


def read_snapshot(base_dir: str, bucket_id: str, ts: int) -> dict | None:
    """读某一版快照，返回 {bucket_id,timestamp,metadata,content}。"""
    try:
        path = safe_path(snapshots_dir(base_dir), f"{str(bucket_id).strip()}_{int(ts)}.json")
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning(f"读快照失败: {bucket_id}@{ts}: {e}")
        return None


def prune_snapshots(base_dir: str, bucket_id: str, keep: int = DEFAULT_KEEP) -> int:
    """只保留最近 keep 版，返回删掉的数量。"""
    if keep <= 0:
        return 0
    snaps = list_snapshots(base_dir, bucket_id)
    removed = 0
    for entry in snaps[keep:]:
        try:
            os.remove(entry["path"])
            removed += 1
        except OSError:
            pass
    return removed
