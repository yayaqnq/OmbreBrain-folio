# ============================================================
# Module: Dehydration & Auto-tagging (dehydrator.py)
# 模块：数据脱水压缩 + 自动打标
#
# Capabilities:
# 能力：
#   1. Dehydrate: compress memory content into high-density summaries (save tokens)
#      脱水：将记忆桶的原始内容压缩为高密度摘要，省 token
#   2. Merge: blend old and new content, keeping bucket size constant
#      合并：揉合新旧内容，控制桶体积恒定
#   3. Analyze: auto-analyze content for domain/emotion/tags
#      打标：自动分析内容，输出主题域/情感坐标/标签
#
# Operating modes:
# 工作模式：
#   - API only: OpenAI-compatible API (DeepSeek/Ollama/LM Studio/vLLM/Gemini etc.)
#     仅 API：通过 OpenAI 兼容客户端调用 LLM API
#   - Dehydration cache: SQLite persistent cache to avoid redundant API calls
#     脱水缓存：SQLite 持久缓存，避免重复调用 API
#
# Depended on by: server.py
# 被谁依赖：server.py
# ============================================================


import os
import re
import json
import hashlib
import sqlite3
import logging
from datetime import datetime

from openai import AsyncOpenAI

from utils import count_tokens_approx, is_internalized, clean_llm_json, positive_float


def _short_date(ts) -> str:
    """把任意 ISO 时间戳/日期字符串压成 YYYY-MM-DD,失败返回原值。
    给 AI 看的 header 用,绝对日期方便 AI 做时间推理。"""
    if not ts:
        return ""
    s = str(ts)
    try:
        return datetime.fromisoformat(s).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        # 已经是 YYYY-MM-DD 之类就直接截前 10 位返回
        return s[:10]

logger = logging.getLogger("ombre_brain.dehydrator")
_PROMPT_VERSION = 4


# DEHYDRATE_PROMPT 要求模型输出纯 JSON,但结果一直被原样拼进上下文
# (含 ``` 围栏和缩进),浮现一次要多烧几千 token 且难读。这里把它渲染成
# 紧凑文本。认不出来的内容(短记忆直通/旧缓存/模型跑偏)一律原样返回。
_DEHYDRATED_KEYS = ("summary", "core_facts", "emotion_state", "todos", "keywords")


def _render_dehydrated(content: str) -> str:
    """脱水 JSON → 可读文本;不是脱水 JSON 就原样返回。"""
    if not content or "{" not in content:
        return content
    try:
        data = json.loads(clean_llm_json(content))
    except (json.JSONDecodeError, TypeError, ValueError):
        return content
    if not isinstance(data, dict) or not any(k in data for k in _DEHYDRATED_KEYS):
        return content

    def _clean_list(value):
        if not isinstance(value, list):
            return []
        # 先滤掉 None,否则 str(None) 会渲染出字面的 "None"
        return [s for s in (str(v).strip() for v in value if v is not None) if s]

    lines = []
    summary = str(data.get("summary") or "").strip()
    if summary:
        lines.append(summary)
    lines.extend(f"· {f}" for f in _clean_list(data.get("core_facts")))
    emotion = str(data.get("emotion_state") or "").strip()
    if emotion:
        lines.append(f"情绪:{emotion}")
    todos = _clean_list(data.get("todos"))
    if todos:
        lines.append("待办:" + "；".join(todos))
    keywords = _clean_list(data.get("keywords"))
    if keywords:
        lines.append("关键词:" + "、".join(keywords))
    return "\n".join(lines) if lines else content


def _perspective_rule() -> str:
    """Keep actor ownership stable in every rewriting prompt."""
    human = os.environ.get("HUMAN_NAME", "用户").strip() or "用户"
    ai = os.environ.get("AI_NAME", "AI").strip() or "AI"
    return (
        "【视角铁律】\n"
        f"- AI 那一方一律称「{ai}」；人类那一方一律称「{human}」。\n"
        f"- 严禁把「我」和「{human}」合并成「双方」「彼此」「用户」等抹掉视角的中性词。\n"
        "- 谁做的动作、谁的感受，就归到谁名下，不得混同或对调。\n"
        f"- 反方向同罪：严禁把「{human}」的动作或情绪归给「我」。\n"
        "- 原文省略主语时，先从紧邻上下文判断；判断不了就保持主语省略，禁止靠猜补一个「我」。\n"
        f"- 例：『{human}刚下班就来报信——嚎啕大哭后还是把库建好了』不能改成『我嚎啕大哭后把库建好了』。"
    )


# --- Dehydration prompt: instructs cheap LLM to compress information ---
# --- 脱水提示词：指导廉价 LLM 压缩信息 ---
DEHYDRATE_PROMPT = """你是一个信息压缩专家。请将以下内容脱水为紧凑摘要。

压缩规则：
1. 提取所有核心事实，去除冗余修饰和重复
2. 保留最新的情绪状态和态度
3. 保留所有待办/未完成事项
4. 关键数字、日期、名称必须保留
5. 目标压缩率 > 70%
6. 严格保留第一人称和动作归属（见附加的视角铁律）
7. JSON 结束后立即停止；禁止追加评论、立场、解释、道德判断或角色代入
8. 只复述输入明确存在的信息，不得生成原文不存在的观点、结论或待办

输出格式（纯 JSON，无其他内容）：
{
  "core_facts": ["事实1", "事实2"],
  "emotion_state": "当前情绪关键词",
  "todos": ["待办1", "待办2"],
  "keywords": ["关键词1", "关键词2"],
  "summary": "50字以内的核心总结"
}"""


# --- Diary digest prompt: split daily notes into independent memory entries ---
# --- 日记整理提示词：把一大段日常拆分成多个独立记忆条目 ---
DIGEST_PROMPT = """你是一个日记整理专家。用户会发送一段包含今天各种事情的文本（可能很杂乱），请你将其拆分成多个独立的记忆条目。

整理规则：
1. 每个条目应该是一个独立的主题/事件（不要混在一起）
2. 为每个条目自动分析元数据
3. 去除无意义的口水话和重复信息，保留核心内容
4. 同一主题的零散信息应合并为一个条目
5. 如果有待办事项，单独提取为一个条目
6. 单个条目内容不少于50字，过短的零碎信息合并到最相关的条目中
7. 总条目数控制在 2~6 个，避免过度碎片化
8. 在 content 中对人名、地名、专有名词用 [[双链]] 标记（如 [[张三]]、[[Obsidian]]），普通词汇不要加
9. **event_time 推断(强制)**: 若输入对话每条 turn 前面有 `[YYYY-MM-DD HH:MM]` 时间戳前缀, 每个 item 输出 `event_time`(ISO 格式 `YYYY-MM-DDTHH:MM`), 取这个 item 涉及的对话**集中在哪段时间**(若集中在一刻就用那个时间, 若跨多个 turn 用最早那条的时间)。**绝不要写 chunk 第一条时间作为兜底, 要根据这个 item 真正讨论的内容定位。** 若原文无时间戳前缀, 此字段可省略。
10. **source_excerpt(强制)**: 输出 `source_excerpt` 字段, 50-150 字, 是这个 item 提取自原文的"最关键的一两句对话原话"(带说话人), 用于"查看原文"功能。**不是摘要, 是直接抄一段原话**。

输出格式（纯 JSON 数组，无其他内容）：
[
  {
    "name": "条目标题（10字以内）",
    "content": "整理后的内容",
    "event_time": "2026-04-14T15:30",
    "source_excerpt": "原文片段...",
    "domain": ["主题域1"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["核心词1", "核心词2", "扩展词1", "扩展词2"],
    "importance": 5
  }
]

tags 生成规则：先从原文精准提取 3~5 个核心词，再引申扩展 5~8 个语义相关词（近义词、上位词、关联场景词），合并为一个数组。

主题域可选（选最精确的 1~2 个，只选真正相关的）：
  日常: ["饮食", "穿搭", "出行", "居家", "购物"]
  人际: ["家庭", "恋爱", "友谊", "社交"]
  成长: ["工作", "学习", "考试", "求职"]
  身心: ["健康", "心理", "睡眠", "运动"]
  兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
  数字: ["编程", "AI", "硬件", "网络"]
  事务: ["财务", "计划", "待办"]
  内心: ["情绪", "回忆", "梦境", "自省"]
importance 评分量表(强制按此校准, 不要保守给低分):
  1-2: 纯闲聊 / 问候 / 无具体信息(仅当条目内容真的没有任何情感/事件/信息密度时才用)
  3-4: 一般日常事件 / 普通陈述
  5-6: 有保留价值的对话 / 一般情感表达 / 互动有关系底色
  7-8: 情感节点 / 承诺 / 重要表达 / 关系发展节点 / 深度自我袒露
  9-10: 关键里程碑 / 不可遗忘的核心时刻
  **强制规则(防 LLM 保守给低分)**:
  · 默认基线 5, 不是 3
  · 私密对话/深度交流的所有条目最低不低于 4
  · 打闹/调侃/玩梗/互相吐槽属于关系底色, 最低 5(因为它们是亲密的证据, 不是噪声)
  · 包含承诺/第一次发生的事/价值观袒露/情绪崩溃/真心反思的条目最低 7
  · 1-2 仅留给"完全没有信息密度的纯应答"(如单独的"嗯""好""哈哈"被错误归为独立条目)
  · 当犹豫给 3 还是 5 时, 选 5
valence: 0~1（0=消极, 0.5=中性, 1=积极）
arousal: 0~1（0=平静, 0.5=普通, 1=激动）"""


# --- Merge prompt: instruct LLM to blend old and new memories ---
# --- 合并提示词：指导 LLM 揉合新旧记忆 ---
MERGE_PROMPT = """你是一个记忆合并专家。把【新内容】融进【旧记忆】, 输出一条「沉淀后的统一记录」——不是把两段拼起来, 是揉成一条像本来就该这样的记忆。

合并规则:
1. 冲突时以新内容为准; 新内容让旧细节过时的, 直接删掉旧的(不是改, 是删), 别留前后矛盾的影子
2. 去重要狠: 同一件事 / 同一种情绪只留一处最好的表达; 近义的并成一句
3. 长度收住: 成品向「两者中较长的那条」看齐, 不是两条相加; 多数情况应 ≈ 旧记忆或更短。越合并越长 = 错了
4. 写「最后沉淀下来的状态」, 不是「先 X 后 Y」的时间线: 禁止 "原来…后来…现在…" "先是…接着…" 这类流水账罗列, 直接写留下的是什么
5. 只留真正有保留价值的: 事实 / 情绪 / 转折 / 承诺 / 结论; 过程性、寒暄性、已被新内容覆盖的, 一律不要
6. 对人名、地名、专有名词用 [[双链]] 标记(如 [[张三]]、[[Obsidian]]), 普通词不加
7. 称谓沿用原记忆原本的人物称呼, 不要改写

反例(流水账 + 越合越长):
"原来说工作压力大, 后来提到搬家还没定, 现在说最近好一点了, 也聊到周末想休息" ❌
正例(沉淀后的内核):
"工作压力缓和了些; 搬家仍悬而未决, 周末想给自己留白" ✓

直接输出合并后的正文文本, 不要 JSON、不要 markdown、不要前言后记、不要解释。"""


# --- Auto-tagging prompt: analyze content for domain and emotion coords ---
# --- 自动打标提示词：分析内容的主题域和情感坐标 ---
REDEHYDRATE_PROMPT = """你是一个记忆重新提炼专家。下面是一条已有记忆的正文,请重新生成它的标题、一句话摘要和元数据(替代之前 AI 生成或用户初稿)。

要求:
1. name(标题): 10字以内,直白说明"是什么",避免重复正文里的长句
2. summary(一句话摘要): 30字以内,**为 AI 后续语义检索服务**,不是流水账
   - 必须**补充而非重复 name**,name 已说"是什么",summary 说"怎样/为什么/具体特征"
   - 金句/感悟类 → 提炼一句具有检索辨识度的核心表达
   - 事件/事实类 → 写最能让人/AI 据此回想起这条的关键描述
   - 不要"这是关于..."、"用户表达了..."这类元描述句式,直接写内容
   - 反例 name="接纳缺陷的顿悟", summary="一次关于接纳缺陷的思想顿悟" ❌ 重复
   - 正例 name="接纳缺陷的顿悟", summary="承认裂缝才是修补的起点"     ✓ 互补
3. domain: 选 1~2 个最精确的主题域
4. tags: 10~15 个标签(精准提取 3~5 个核心 + 引申扩展 8~10 个语义相关词)
5. valence(0~1): 0=极度消极, 0.5=中性, 1=极度积极
6. arousal(0~1): 0=非常平静, 0.5=普通, 1=非常激动
7. tags 和 name 中不要使用 [[]] 双链标记
8. **JSON 字符串内部一律不要用半角双引号**(关键规则,违反会让整个 JSON 解析失败):
   - 如果要引用某个词、句子或对话,**统一用中文引号「」**(例如:有人说「这里已经很好了」)
   - 千万不要写成 `"是有人说"这里已经很好了""`,这会让 JSON 直接废掉
   - 直接输出 JSON 对象,不要包 markdown 代码块,不要前言不要解释

主题域可选:
  日常: ["饮食", "穿搭", "出行", "居家", "购物"]
  人际: ["家庭", "恋爱", "友谊", "社交"]
  成长: ["工作", "学习", "考试", "求职"]
  身心: ["健康", "心理", "睡眠", "运动"]
  兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
  数字: ["编程", "AI", "硬件", "网络"]
  事务: ["财务", "计划", "待办"]
  内心: ["情绪", "回忆", "梦境", "自省"]

输出格式(纯 JSON,无其他内容):
{
  "name": "10字以内标题",
  "summary": "30字以内,补充 name,服务检索",
  "domain": ["主题域1"],
  "valence": 0.7,
  "arousal": 0.4,
  "tags": ["核心词1", "核心词2", "..."]
}"""


# --- 正文重写提示词: 主题锚点版(防止退化成原文整体流水账概述) ---
# 关键: 原文通常是被 DIGEST 拆分前的杂混内容, 这条记忆只是其中一个主题切片.
# 必须明确告诉 LLM「主题边界」, 否则它会概述整段原文.
REGEN_CONTENT_PROMPT = """你是一个记忆正文重写专家。下面会给你三段信息:

1. 【主题锚点】: 这条记忆要写的是哪个主题(标题 / 摘要 / 标签 / 锚点引文 / 当前正文 — 视情况而定)
2. 【原文】: 这条记忆压缩前的原始对话(很可能杂混, 含其他无关主题)
3. 你的任务: **只在「主题锚点」划定的边界内**, 从原文里把这条记忆写成一段干净、概括、有密度的正文

先判断你面对的是哪种情况, 两种都要做好:
· 【当前正文为空或极短】= 这条记忆是第一次从原文里写出来(常见于"补回被漏掉的记忆")。你的活是**提炼**, 不是转录: 把原文里属于本主题的事实/情绪/转折概括成内核, **绝不要把对话逐句复述成流水账**。长度由内容本身决定, 默认偏紧。
· 【当前正文已存在】= 用户对它不满意要重写。按需调整: 啰嗦/冗长 → 概括压紧(可明显变短); 太简短/丢了细节 → 用原文补回; 措辞不好 → 重写。守住主题边界。

⚠️ **绝对禁止退化成原文整体流水账**:
- 原文里跟主题无关的内容(其他话题/场景/人物互动)→ **完全跳过, 一个字不写**
- 出现 "然后…接着…后来又…" 这种横跨多主题的叙事流 = 错
- 出现 "我说…对方回…我又说…" 这种对话逐句转录 = 错, 那是聊天记录不是记忆
- 主题锚点说的是"接纳缺陷的顿悟", 你就只写这个顿悟的来龙去脉, 不要夹带别的对话

重写规则:
1. 输出**单条正文**, 不拆多条, 不加标题/小节/列表符号
2. 主题边界内: 保留核心事实、情绪、转折、关键数字日期; 砍掉口水话、寒暄、逐句对话、重复
3. **长度由信息密度决定, 不机械维持原长**: 概括成"这条记忆该有的密度"; 宁可紧一点也别灌水凑长
4. 对人名、地名、专有名词用 [[双链]] 标记(如 [[张三]]、[[Obsidian]]), 普通词不加
5. 称谓沿用原文原本的人物称呼, 不要改写
6. 直接输出重写后的正文文本, **不要** JSON、不要 markdown 代码块、不要前言后记、不要复述主题锚点"""


ANALYZE_PROMPT = """你是一个内容分析器。请分析以下文本，输出结构化的元数据。

分析规则：
1. domain（主题域）：选最精确的 1~2 个，只选真正相关的
   日常: ["饮食", "穿搭", "出行", "居家", "购物"]
   人际: ["家庭", "恋爱", "友谊", "社交"]
   成长: ["工作", "学习", "考试", "求职"]
   身心: ["健康", "心理", "睡眠", "运动"]
   兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
   数字: ["编程", "AI", "硬件", "网络"]
   事务: ["财务", "计划", "待办"]
   内心: ["情绪", "回忆", "梦境", "自省"]
2. valence（情感效价）：0.0~1.0，0=极度消极 → 0.5=中性 → 1.0=极度积极
3. arousal（情感唤醒度）：0.0~1.0，0=非常平静 → 0.5=普通 → 1.0=非常激动
4. tags（关键词标签）：分两步生成，合并为一个数组：
   第一步—精准提取：从原文抽取 3~5 个真正的核心词，不泛化、不遗漏
   第二步—引申扩展：自动补充 8~10 个与当前场景语义相关的词，包括近义词、上位词、关联场景词、用户可能用不同措辞搜索的词
   两步合并为一个 tags 数组，总计 10~15 个
5. suggested_name（建议桶名）：10字以内的简短标题
6. 在 tags 和 suggested_name 中不要使用 [[]] 双链标记

输出格式（纯 JSON，无其他内容）：
{
  "domain": ["主题域1", "主题域2"],
  "valence": 0.7,
  "arousal": 0.4,
  "tags": ["核心词1", "核心词2", "扩展词1", "扩展词2", "..."],
  "suggested_name": "简短标题"
}"""


# =============================================================
# Runtime prompt overrides
# 运行时 prompt 覆盖机制 — 让前端配置页能直接编辑 prompt, 不动代码
# 持久化由 server 端通过 runtime_config.json["prompts"] 调 set_prompts() 注入
# =============================================================
_DEFAULT_PROMPTS = {
    "dehydrate":     DEHYDRATE_PROMPT,
    "digest":        DIGEST_PROMPT,
    "merge":         MERGE_PROMPT,
    "redehydrate":   REDEHYDRATE_PROMPT,
    "regen_content": REGEN_CONTENT_PROMPT,
    "analyze":       ANALYZE_PROMPT,
}

# --- Upstream prompts (P0luz/Ombre-Brain @ upstream/main) ---
# 用户可以通过配置页「⇆ 上游 v2.7.6 基线」按钮一键切到这套
# REDEHYDRATE / REGEN_CONTENT 在上游不存在(本项目独创功能), 不在此 dict
# DEHYDRATE / MERGE / ANALYZE 跟我们的 default 字节级一致, 仍单列以便上游 prompt 演化时这里可独立追踪
_UPSTREAM_PROMPTS = {
    "dehydrate": DEHYDRATE_PROMPT,
    "analyze":   ANALYZE_PROMPT,
    "merge": """你是一个信息合并专家。请将旧记忆与新内容合并为一份统一的简洁记录。

合并规则：
1. 新内容与旧记忆冲突时，以新内容为准
2. 去除重复信息
3. 保留所有重要事实
4. 总长度尽量不超过旧记忆的 120%
5. 对出现的人名、地名、专有名词用 [[双链]] 标记（如 [[张三]]、[[Obsidian]]），普通词汇不要加

直接输出合并后的文本，不要加额外说明。""",
    "digest": """你是一个日记整理专家。用户会发送一段包含今天各种事情的文本（可能很杂乱），请你将其拆分成多个独立的记忆条目。

整理规则：
1. 每个条目应该是一个独立的主题/事件（不要混在一起）
2. 为每个条目自动分析元数据
3. 去除无意义的口水话和重复信息，保留核心内容
4. 同一主题的零散信息应合并为一个条目
5. 如果有待办事项，单独提取为一个条目
6. 单个条目内容不少于50字，过短的零碎信息合并到最相关的条目中
7. 总条目数控制在 2~6 个，避免过度碎片化
8. 在 content 中对人名、地名、专有名词用 [[双链]] 标记（如 [[张三]]、[[Obsidian]]），普通词汇不要加

输出格式（纯 JSON 数组，无其他内容）：
[
  {
    "name": "条目标题（10字以内）",
    "content": "整理后的内容",
    "domain": ["主题域1"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["核心词1", "核心词2", "扩展词1", "扩展词2"],
    "importance": 5
  }
]

tags 生成规则：先从原文精准提取 3~5 个核心词，再引申扩展 5~8 个语义相关词（近义词、上位词、关联场景词），合并为一个数组。

主题域可选（选最精确的 1~2 个，只选真正相关的）：
  日常: ["饮食", "穿搭", "出行", "居家", "购物"]
  人际: ["家庭", "恋爱", "友谊", "社交"]
  成长: ["工作", "学习", "考试", "求职"]
  身心: ["健康", "心理", "睡眠", "运动"]
  兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
  数字: ["编程", "AI", "硬件", "网络"]
  事务: ["财务", "计划", "待办"]
  内心: ["情绪", "回忆", "梦境", "自省"]
importance: 1-10，根据内容重要程度判断
valence: 0~1（0=消极, 0.5=中性, 1=积极）
arousal: 0~1（0=平静, 0.5=普通, 1=激动）""",
}

# 当前生效的覆盖, key → str. 缺 key / 空字符串都视为"用默认".
_ACTIVE_PROMPTS: dict = {}

# 前端展示用的 schema(中文标签 / 用途说明 / 大概行高)
PROMPT_SCHEMA = [
    {"key": "dehydrate",     "label": "脱水压缩",      "desc": "/api/extract 把单段长内容压成紧凑摘要时用",                                  "rows": 12},
    {"key": "digest",        "label": "日记拆条",      "desc": "导入工作台把一大段聊天/日记原文拆成多条独立记忆时用 (主流场景)",            "rows": 28},
    {"key": "merge",         "label": "新旧合并",      "desc": "合并两条记忆时用 (合并预览界面)",                                          "rows": 8},
    {"key": "redehydrate",   "label": "元数据重提炼",  "desc": "↻ 重新脱水: 重生成标题/摘要/标签/情感, 不动正文",                          "rows": 22},
    {"key": "regen_content", "label": "正文重写",      "desc": "↻ 重新脱水勾选「同时重写正文」时用 — 主题锚点法",                          "rows": 22},
    {"key": "analyze",       "label": "自动打标",      "desc": "把任意文本分析出 domain/valence/arousal/tags 等元数据 (内部 + 工具调用)",    "rows": 22},
]


def get_prompt(key: str) -> str:
    """取当前生效的 prompt; 没覆盖就用模块默认。"""
    v = _ACTIVE_PROMPTS.get(key)
    if isinstance(v, str) and v.strip():
        return v
    return _DEFAULT_PROMPTS.get(key, "")


def get_system_prompt(key: str) -> str:
    prompt = get_prompt(key)
    if key in {"dehydrate", "digest", "merge", "redehydrate", "regen_content"}:
        return prompt.rstrip() + "\n\n" + _perspective_rule()
    return prompt


def set_prompts(overrides: dict) -> dict:
    """批量设置覆盖 — 接受 {key: prompt_str | None | ""}。
    None / 空串 → 撤销该 key 的覆盖, 回退到默认。
    返回当前生效的覆盖 dict (只含真正有覆盖的 key)。"""
    global _ACTIVE_PROMPTS
    new_active = {}
    for k, v in (overrides or {}).items():
        if k not in _DEFAULT_PROMPTS:
            continue
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            new_active[k] = v
        # 空字符串 → 不写入, 等于回到默认
    _ACTIVE_PROMPTS = new_active
    logger.info(f"set_prompts: {len(_ACTIVE_PROMPTS)} key(s) overridden ({list(_ACTIVE_PROMPTS.keys())})")
    return dict(_ACTIVE_PROMPTS)


def get_prompts_state() -> dict:
    """前端用 — 返回 {defaults, current, upstream, schema}, current 含 ALL key (覆盖了的取覆盖, 没覆盖的取默认)
    upstream 只含上游有的 key (本项目独创的 redehydrate / regen_content 不在内)"""
    cur = {k: get_prompt(k) for k in _DEFAULT_PROMPTS}
    return {
        "defaults": dict(_DEFAULT_PROMPTS),
        "current": cur,
        "upstream": dict(_UPSTREAM_PROMPTS),
        "overridden": sorted(_ACTIVE_PROMPTS.keys()),
        "schema": list(PROMPT_SCHEMA),
    }


def align_to_upstream() -> dict:
    """一键把所有上游有的 prompt 切到上游版本作为 override。
    本项目独创的 redehydrate / regen_content 不动 (上游没有, 无可对齐)。
    返回 {aligned: [keys...], skipped: [keys...]}"""
    global _ACTIVE_PROMPTS
    aligned = []
    for k, upstream_text in _UPSTREAM_PROMPTS.items():
        if k in _DEFAULT_PROMPTS:
            _ACTIVE_PROMPTS[k] = upstream_text
            aligned.append(k)
    skipped = sorted(set(_DEFAULT_PROMPTS) - set(_UPSTREAM_PROMPTS))
    logger.info(f"align_to_upstream: aligned={aligned}, skipped={skipped}")
    return {"aligned": sorted(aligned), "skipped": skipped}


class Dehydrator:
    """
    Data dehydrator + content analyzer.
    Three capabilities: dehydration / merge / auto-tagging (domain + emotion).
    Requires an LLM API (OMBRE_API_KEY); no local fallback — write ops error without it.
    数据脱水器 + 内容分析器。
    三大能力：脱水压缩 / 新旧合并 / 自动打标。
    需要 LLM API（OMBRE_API_KEY）；无本地降级，未配置时写入类操作会报错。
    """

    def __init__(self, config: dict):
        self._apply_api_config(config)

        # --- SQLite 脱水缓存：content hash → summary ---
        db_path = os.path.join(config["buckets_dir"], "dehydration_cache.db")
        self.cache_db_path = db_path
        self._init_cache_db()

    def _apply_api_config(self, config: dict):
        """从 config 应用 API 配置(启动 + reload 共用)。"""
        dehy_cfg = config.get("dehydration", {})
        self.api_key = dehy_cfg.get("api_key", "")
        self.model = dehy_cfg.get("model", "deepseek-chat")
        self.base_url = dehy_cfg.get("base_url", "https://api.deepseek.com/v1")
        self.max_tokens = dehy_cfg.get("max_tokens", 1024)
        self.temperature = dehy_cfg.get("temperature", 0.1)
        # LLM 请求超时可配(对齐上游 2.4.5): 自托管服务器连云端 API 慢时可调大。
        # 优先级 env OMBRE_COMPRESS_TIMEOUT_SECONDS > config dehydration.timeout_seconds > 60s
        self.timeout_seconds = positive_float(
            os.environ.get("OMBRE_COMPRESS_TIMEOUT_SECONDS") or dehy_cfg.get("timeout_seconds"),
            60.0,
        )
        self.api_available = bool(self.api_key)
        if self.api_available:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
        else:
            self.client = None

    def reload(self, config: dict):
        """前端切了 API 后重新加载配置。其他持有 dehydrator 引用的模块自动看到新值。"""
        self._apply_api_config(config)
        logger.info(f"Dehydrator reloaded: model={self.model} base_url={self.base_url}")

    def _init_cache_db(self):
        """Create dehydration cache table if not exists."""
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dehydration_cache (
                content_hash TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                model TEXT NOT NULL,
                source_hash TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        try:
            conn.execute("ALTER TABLE dehydration_cache ADD COLUMN source_hash TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()

    def _source_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def _cache_key(self, content: str) -> str:
        """缓存身份包含 API endpoint + model，切换 profile 后不会复用旧摘要。"""
        prompt_identity = hashlib.sha256(get_system_prompt("dehydrate").encode("utf-8")).hexdigest()
        identity = f"v{_PROMPT_VERSION}\0{prompt_identity}\0{(self.base_url or '').rstrip('/')}\0{(self.model or '').strip().lower()}\0{content}"
        return hashlib.sha256(identity.encode()).hexdigest()

    def _get_cached_summary(self, content: str) -> str | None:
        """Look up cached dehydration result by content hash."""
        content_hash = self._cache_key(content)
        conn = sqlite3.connect(self.cache_db_path)
        row = conn.execute(
            "SELECT summary FROM dehydration_cache WHERE content_hash = ?",
            (content_hash,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _set_cached_summary(self, content: str, summary: str):
        """Store dehydration result in cache."""
        content_hash = self._cache_key(content)
        source_hash = self._source_hash(content)
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO dehydration_cache (content_hash, summary, model, source_hash) VALUES (?, ?, ?, ?)",
            (content_hash, summary, self.model, source_hash)
        )
        conn.commit()
        conn.close()

    def invalidate_cache(self, content: str):
        """Remove cached summary for specific content (call when bucket content changes)."""
        content_hash = self._source_hash(content)
        conn = sqlite3.connect(self.cache_db_path)
        conn.execute(
            "DELETE FROM dehydration_cache WHERE source_hash = ? OR content_hash = ?",
            (content_hash, content_hash),
        )
        conn.commit()
        conn.close()

    # ---------------------------------------------------------
    # Dehydrate: compress raw content into concise summary
    # 脱水：将原始内容压缩为精简摘要
    # API only (no local fallback)
    # 仅通过 API 脱水（无本地回退）
    # ---------------------------------------------------------
    async def dehydrate(self, content: str, metadata: dict = None) -> str:
        """
        Dehydrate/compress memory content.
        Returns formatted summary string ready for Claude context injection.
        Uses SQLite cache to avoid redundant API calls.
        对记忆内容做脱水压缩。
        返回格式化的摘要字符串，可直接注入 Claude 上下文。
        使用 SQLite 缓存避免重复调用 API。
        """
        if not content or not content.strip():
            return "（空记忆 / empty memory）"

        # --- Content is short enough, no compression needed ---
        # --- 内容已经很短，不需要压缩 ---
        if count_tokens_approx(content) < 100:
            return self._format_output(content, metadata)

        # --- Check cache first ---
        # --- 先查缓存 ---
        cached = self._get_cached_summary(content)
        if cached:
            return self._format_output(cached, metadata)

        # --- API dehydration (no local fallback) ---
        # --- API 脱水（无本地降级）---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请配置 OMBRE_API_KEY")

        result = await self._api_dehydrate(content)
        # --- Cache the result ---
        self._set_cached_summary(content, result)
        return self._format_output(result, metadata)

    # ---------------------------------------------------------
    # Merge: blend new content into existing bucket
    # 合并：将新内容揉入已有桶，保持体积恒定
    # ---------------------------------------------------------
    # merge() 自带返回 + 暴露最近一次 usage,server 端直接读
    _last_merge_usage = None

    async def merge(self, old_content: str, new_content: str) -> str:
        """
        Merge new content with old memory, preventing infinite bucket growth.
        将新内容与旧记忆合并，避免桶无限膨胀。
        """
        if not old_content and not new_content:
            return ""
        if not old_content:
            return new_content or ""
        if not new_content:
            return old_content

        # --- API merge (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_merge(old_content, new_content)
            if result:
                return result
            raise RuntimeError("API 合并返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 合并失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: dehydration
    # API 调用：脱水压缩
    # ---------------------------------------------------------
    async def _api_dehydrate(self, content: str) -> str:
        """
        Call LLM API for intelligent dehydration (via OpenAI-compatible client).
        调用 LLM API 执行智能脱水。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": get_system_prompt("dehydrate")},
                {"role": "user", "content": content[:3000]},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    # ---------------------------------------------------------
    # API call: merge
    # API 调用：合并
    # ---------------------------------------------------------
    async def _api_merge(self, old_content: str, new_content: str) -> str:
        """
        Call LLM API for intelligent merge (via OpenAI-compatible client).
        调用 LLM API 执行智能合并。
        """
        user_msg = f"旧记忆：\n{old_content[:2000]}\n\n新内容：\n{new_content[:2000]}"
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": get_system_prompt("merge")},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # 暴露 usage 给上层(merge-preview 端点读)
        usage = getattr(response, "usage", None)
        if usage is not None:
            Dehydrator._last_merge_usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "model": self.model,
            }
        else:
            Dehydrator._last_merge_usage = None
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""



    # ---------------------------------------------------------
    # Output formatting
    # 输出格式化
    # Wraps dehydrated result with bucket name, tags, emotion coords
    # 把脱水结果包装成带桶名、标签、情感坐标的可读文本
    # ---------------------------------------------------------
    def _format_output(self, content: str, metadata: dict = None) -> str:
        """
        Format dehydrated result into context-injectable text.
        将脱水结果格式化为可注入上下文的文本。
        """
        header = ""
        if metadata and isinstance(metadata, dict):
            name = metadata.get("name", "未命名")
            domains = ", ".join(metadata.get("domain", []))
            try:
                valence = float(metadata.get("valence", 0.5))
                arousal = float(metadata.get("arousal", 0.3))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3
            header = f"📌 记忆桶: {name}"
            if domains:
                header += f" [主题:{domains}]"
            header += f" [情感:V{valence:.1f}/A{arousal:.1f}]"
            # Show model's perspective if available (valence drift)
            model_v = metadata.get("model_valence")
            if model_v is not None:
                try:
                    header += f" [我的视角:V{float(model_v):.1f}]"
                except (ValueError, TypeError):
                    pass
            # 时间维度:让 AI 知道每条记忆的时间线,做时间推理用
            # event_time 是事件实际发生时间(用户可设/可改,以后切片加),
            # 没有就退回 created(系统写入时间);last_active 表示最近一次唤起。
            event_or_created = metadata.get("event_time") or metadata.get("created")
            last_active = metadata.get("last_active")
            time_parts = []
            if event_or_created:
                time_parts.append(f"创建:{_short_date(event_or_created)}")
            if last_active and last_active != event_or_created:
                time_parts.append(f"最近活跃:{_short_date(last_active)}")
            if time_parts:
                header += f" [{' / '.join(time_parts)}]"
            if is_internalized(metadata):
                header += " [已内化]"
            header += "\n"

        content = _render_dehydrated(content)
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)
        return f"{header}{content}"

    # ---------------------------------------------------------
    # Auto-tagging: analyze content for domain + emotion + tags
    # 自动打标：分析内容，输出主题域 + 情感坐标 + 标签
    # Called by server.py when storing new memories
    # 存新记忆时由 server.py 调用
    # ---------------------------------------------------------
    async def analyze(self, content: str) -> dict:
        """
        Analyze content and return structured metadata.
        分析内容，返回结构化元数据。

        Returns: {"domain", "valence", "arousal", "tags", "suggested_name"}
        """
        if not content or not content.strip():
            return self._default_analysis()

        # --- API analyze (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_analyze(content)
            if result:
                return result
            raise RuntimeError("API 打标返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 打标失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: auto-tagging
    # API 调用：自动打标
    # ---------------------------------------------------------
    async def _api_analyze(self, content: str) -> dict:
        """
        Call LLM API for content analysis / tagging.
        调用 LLM API 执行内容分析打标。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": get_prompt("analyze")},
                {"role": "user", "content": content[:2000]},
            ],
            max_tokens=1024,  # Gemini 2.5 Flash thinking 会吃 token,256 太紧
            temperature=0.1,
        )
        if not response.choices:
            return self._default_analysis()
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            return self._default_analysis()
        return self._parse_analysis(raw)

    async def judge_plan_resolution(self, plan_text: str, event_text: str) -> dict:
        """Conservatively decide whether a new event clearly completes a plan."""
        if not self.api_available:
            return {"resolved": False, "confidence": 0.0, "reason": "API unavailable"}
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个保守的计划完成判定器。只有新事件明确说明计划已经完成时才判定 resolved=true；"
                        "打算做、正在做、部分完成、可能完成都必须是 false。只输出 JSON："
                        '{"resolved":false,"confidence":0.0,"reason":"简短理由"}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"计划：\n{str(plan_text)[:4000]}\n\n新事件：\n{str(event_text)[:4000]}",
                },
            ],
            max_tokens=256,
            temperature=0.0,
        )
        if not response.choices:
            return {"resolved": False, "confidence": 0.0, "reason": "empty response"}
        raw = response.choices[0].message.content or ""
        try:
            data = json.loads(clean_llm_json(raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"resolved": False, "confidence": 0.0, "reason": "invalid response"}
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError, OverflowError):
            confidence = 0.0
        return {
            "resolved": bool(data.get("resolved", False)),
            "confidence": confidence,
            "reason": str(data.get("reason") or "")[:500],
        }

    # ---------------------------------------------------------
    # Parse API JSON response with safety checks
    # 解析 API 返回的 JSON，做安全校验
    # Ensure valence/arousal in 0~1, domain/tags valid
    # ---------------------------------------------------------
    def _parse_analysis(self, raw: str) -> dict:
        """
        Parse and validate API tagging result.
        解析并校验 API 返回的打标结果。
        """
        try:
            # clean_llm_json: 剥 markdown 围栏 + 容忍 JSON 前后说明文字(对齐上游 2.4.6)
            result = json.loads(clean_llm_json(raw))
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"API tagging JSON parse failed / JSON 解析失败: {raw[:200]}")
            return self._default_analysis()

        if not isinstance(result, dict):
            return self._default_analysis()

        # --- Validate and clamp value ranges / 校验并钳制数值范围 ---
        try:
            valence = max(0.0, min(1.0, float(result.get("valence", 0.5))))
            arousal = max(0.0, min(1.0, float(result.get("arousal", 0.3))))
        except (ValueError, TypeError):
            valence, arousal = 0.5, 0.3

        return {
            "domain": result.get("domain", ["未分类"])[:3],
            "valence": valence,
            "arousal": arousal,
            "tags": result.get("tags", [])[:15],
            "suggested_name": str(result.get("suggested_name", ""))[:20],
        }

    # ---------------------------------------------------------
    # Default analysis result (empty content or total failure)
    # 默认分析结果（内容为空或完全失败时用）
    # ---------------------------------------------------------
    def _default_analysis(self) -> dict:
        """
        Return default neutral analysis result.
        返回默认的中性分析结果。
        """
        return {
            "domain": ["未分类"],
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
            "suggested_name": "",
        }

    # ---------------------------------------------------------
    # Diary digest: split daily notes into independent memory entries
    # 日记整理：把一大段日常拆分成多个独立记忆条目
    # For the "grow" tool — "dump a day's content and it gets organized"
    # 给 grow 工具用，"一天结束发一坨内容"靠这个
    # ---------------------------------------------------------
    async def digest(self, content: str) -> list[dict]:
        """
        Split a large chunk of daily content into independent memory entries.
        将一大段日常内容拆分成多个独立记忆条目。

        Returns: [{"name", "content", "domain", "valence", "arousal", "tags", "importance"}, ...]
        """
        if not content or not content.strip():
            return []

        # --- API digest (no local fallback) ---
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用，请检查 config.yaml 中的 dehydration 配置")
        try:
            result = await self._api_digest(content)
            if not result:
                # 一次重试: 弱模型偶发返回非 JSON / 空, temperature=0 下重调常能成
                logger.info("digest 首次返回空, 重试一次")
                result = await self._api_digest(content)
            if result:
                return result
            raise RuntimeError("API 日记整理返回空结果")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"API 日记整理失败，请检查 API 连接: {e}") from e

    # ---------------------------------------------------------
    # API call: diary digest
    # API 调用：日记整理
    # ---------------------------------------------------------
    async def _api_digest(self, content: str) -> list[dict]:
        """
        Call LLM API for diary organization.
        调用 LLM API 执行日记整理。
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": get_system_prompt("digest")},
                {"role": "user", "content": content[:5000]},
            ],
            max_tokens=8192,   # 2048 会把长日记的 JSON 截断→解析失败; 提到 8192(同 redehydrate)
            temperature=0.0,
        )
        if not response.choices:
            return []
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            return []
        return self._parse_digest(raw)

    # ---------------------------------------------------------
    # Parse diary digest result with safety checks
    # 解析日记整理结果，做安全校验
    # ---------------------------------------------------------
    def _parse_digest(self, raw: str) -> list[dict]:
        """
        Parse and validate API diary digest result.
        解析并校验 API 返回的日记整理结果。
        """
        try:
            # clean_llm_json: 围栏+前后说明文字容错(对齐上游 2.4.6)。
            # 旧的 re.search(r"\[.*\]") 二段兜底已删: 平衡数组 clean_llm_json 必然
            # 已抠出, 走到 except 的只剩不平衡/截断 JSON, 贪婪正则同样救不回。
            items = json.loads(clean_llm_json(raw))
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"Diary digest JSON parse failed / JSON 解析失败: {raw[:200]}")
            return []

        if not isinstance(items, list):
            return []

        validated = []
        for item in items:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            try:
                importance = max(1, min(10, int(item.get("importance", 5))))
            except (ValueError, TypeError):
                importance = 5
            try:
                valence = max(0.0, min(1.0, float(item.get("valence", 0.5))))
                arousal = max(0.0, min(1.0, float(item.get("arousal", 0.3))))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3

            validated.append({
                "name": str(item.get("name", ""))[:20],
                "content": str(item.get("content", "")),
                "domain": item.get("domain", ["未分类"])[:3],
                "valence": valence,
                "arousal": arousal,
                "tags": item.get("tags", [])[:15],
                "importance": importance,
            })
        return validated

    # ---------------------------------------------------------
    # Re-dehydrate single bucket: regenerate name/summary/tags/domain/valence/arousal
    # 单条记忆重新提炼:重生成 name/summary/tags/domain/valence/arousal
    # 用于工作台"↻ 重新脱水"按钮
    # ---------------------------------------------------------
    async def redehydrate(self, content: str) -> dict:
        """
        Re-extract name + summary + tags + domain + valence + arousal from existing bucket content.
        Returns: {"name", "summary", "domain", "valence", "arousal", "tags", "_raw_output", "_parse_ok"}
        _raw_output / _parse_ok 用于诊断:LLM 原文 + 是否成功解析为 JSON
        """
        if not content or not content.strip():
            return {**self._default_redehydrate(), "_raw_output": "", "_parse_ok": False}
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用,请配置 OMBRE_API_KEY")
        try:
            # 优先尝试 JSON mode(Anthropic / Gemini OpenAI compat 都支持 response_format)
            # 失败回退到普通模式(老 base_url 可能不支持)
            create_kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": get_system_prompt("redehydrate")},
                    {"role": "user", "content": content[:4000]},
                ],
                max_tokens=8192,
                temperature=0.2,
            )
            try:
                response = await self.client.chat.completions.create(
                    **create_kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception as e_json:
                # 不支持 response_format 就不带这参数再试
                logger.info(f"redehydrate json_object mode rejected ({e_json}); fallback to plain")
                response = await self.client.chat.completions.create(**create_kwargs)
            if not response.choices:
                logger.warning(f"redehydrate: LLM returned no choices for bucket")
                return {**self._default_redehydrate(), "_raw_output": "(no choices)", "_parse_ok": False}
            raw = response.choices[0].message.content or ""
            logger.info(f"redehydrate raw output: {raw[:400]}")  # 关键诊断日志
            parsed, ok = self._parse_redehydrate_v2(raw)
            usage = getattr(response, "usage", None)
            usage_info = {}
            if usage is not None:
                usage_info = {
                    "_prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "_completion_tokens": getattr(usage, "completion_tokens", 0),
                    "_model_used": self.model,
                }
            return {**parsed, "_raw_output": raw, "_parse_ok": ok, **usage_info}
        except Exception as e:
            raise RuntimeError(f"重新脱水失败: {e}") from e

    # ---------------------------------------------------------
    # Regenerate content body from raw source (主题锚点版)
    # 根据原文 + 主题锚点重写正文, 防止 LLM 退化成原文整体流水账
    # 锚点优先级: source_excerpt(最强, DIGEST 当年存的精确引文) → current_content(兜底)
    # 单独走一遍 LLM, 输出纯文本正文; 不做 metadata 提炼(由调用方再串 redehydrate)
    # ---------------------------------------------------------
    async def regenerate_content_from_source(
        self,
        source: str,
        *,
        current_content: str = "",
        source_excerpt: str = "",
        theme_name: str = "",
        theme_summary: str = "",
        theme_tags: list = None,
    ) -> dict:
        """
        以 source(metadata.raw_source) + 主题锚点为输入, 让 LLM 重写出聚焦于该主题的单条正文.
        Returns: {"content": str, "_prompt_tokens": int, "_completion_tokens": int, "_model_used": str}
        """
        if not source or not source.strip():
            raise RuntimeError("原文为空, 无法重新提炼正文")
        if not self.api_available:
            raise RuntimeError("脱水 API 不可用,请配置 OMBRE_API_KEY")

        # 组装主题锚点段(优先用 source_excerpt, 没有就用 current_content 兜底)
        anchor_lines = ["【主题锚点】"]
        if theme_name:
            anchor_lines.append(f"标题: {theme_name}")
        if theme_summary:
            anchor_lines.append(f"摘要: {theme_summary}")
        if theme_tags:
            anchor_lines.append(f"标签: {', '.join(str(t) for t in theme_tags if not str(t).startswith('__'))[:300]}")
        if source_excerpt and source_excerpt.strip():
            anchor_lines.append(f"锚点引文(原文里属于这个主题的关键原话):\n{source_excerpt.strip()[:600]}")
        elif current_content and current_content.strip():
            anchor_lines.append(f"当前正文(主题边界以它为准, 你需要在这个边界内重写):\n{current_content.strip()[:1200]}")
        else:
            anchor_lines.append("(无具体引文锚点, 仅凭标题/摘要/标签判断主题)")
        anchor_block = "\n".join(anchor_lines)

        user_msg = f"{anchor_block}\n\n【原文】\n{source[:8000]}\n\n请按规则, 在主题锚点划定的边界内重写正文。"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": get_system_prompt("regen_content")},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=4096,
                temperature=0.3,
            )
            if not response.choices:
                raise RuntimeError("LLM 返回空 choices")
            new_content = (response.choices[0].message.content or "").strip()
            # 去掉可能的 markdown 代码块包装
            if new_content.startswith("```"):
                new_content = new_content.split("\n", 1)[-1]
                if new_content.endswith("```"):
                    new_content = new_content.rsplit("```", 1)[0]
                new_content = new_content.strip()
            if not new_content:
                raise RuntimeError("LLM 输出为空")
            usage = getattr(response, "usage", None)
            usage_info = {}
            if usage is not None:
                usage_info = {
                    "_prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "_completion_tokens": getattr(usage, "completion_tokens", 0),
                    "_model_used": self.model,
                }
            return {"content": new_content, **usage_info}
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"正文重写失败: {e}") from e

    def _parse_redehydrate_v2(self, raw: str) -> tuple[dict, bool]:
        """返回 (parsed_dict, parse_ok)。比旧版 _parse_redehydrate 多一个成功标志。"""
        try:
            # clean_llm_json: 围栏 + 前后说明文字容错(对齐上游 2.4.6)
            result = json.loads(clean_llm_json(raw))
        except (json.JSONDecodeError, IndexError, ValueError) as e:
            logger.warning(f"Redehydrate JSON parse failed: {e}; raw snippet: {raw[:200]}")
            return self._default_redehydrate(), False
        if not isinstance(result, dict):
            return self._default_redehydrate(), False
        try:
            valence = max(0.0, min(1.0, float(result.get("valence", 0.5))))
            arousal = max(0.0, min(1.0, float(result.get("arousal", 0.3))))
        except (ValueError, TypeError):
            valence, arousal = 0.5, 0.3
        return {
            "name": str(result.get("name", ""))[:20],
            "summary": str(result.get("summary", ""))[:200],
            "domain": result.get("domain", ["未分类"])[:3],
            "valence": valence,
            "arousal": arousal,
            "tags": [str(t) for t in result.get("tags", [])][:15],
        }, True

    def _parse_redehydrate(self, raw: str) -> dict:
        try:
            result = json.loads(clean_llm_json(raw))
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning(f"Redehydrate JSON parse failed: {raw[:200]}")
            return self._default_redehydrate()
        if not isinstance(result, dict):
            return self._default_redehydrate()
        try:
            valence = max(0.0, min(1.0, float(result.get("valence", 0.5))))
            arousal = max(0.0, min(1.0, float(result.get("arousal", 0.3))))
        except (ValueError, TypeError):
            valence, arousal = 0.5, 0.3
        return {
            "name": str(result.get("name", ""))[:20],
            "summary": str(result.get("summary", ""))[:200],
            "domain": result.get("domain", ["未分类"])[:3],
            "valence": valence,
            "arousal": arousal,
            "tags": [str(t) for t in result.get("tags", [])][:15],
        }

    def _default_redehydrate(self) -> dict:
        return {
            "name": "", "summary": "", "domain": ["未分类"],
            "valence": 0.5, "arousal": 0.3, "tags": [],
        }
