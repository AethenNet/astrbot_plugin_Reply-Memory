"""插件级共享常量（main.py 与 memory_ops.py 共用，避免循环 import）。"""

from __future__ import annotations

import re

CONFIDENCE_TIER_DEFAULTS = {
    "major": 0.95,
    "discussion": 0.75,
    "daily": 0.55,
    "chatter": 0.35,
    "spam": 0.10,
}

MEMORY_KINDS = ("fact", "preference", "plan", "episode", "relation", "experience")

MEMORY_KEYWORDS = ("记住", "记一下", "帮我记", "记着", "别忘了")

IMPORTANCE_MAX = 10.0
IMPORTANCE_DEFAULT = 5.0
KEYWORD_MEMORY_IMPORTANCE = 8.0

KEYWORD_QUESTION_RE = re.compile(r"[?？]|(?:吗|呢|么|嘛)\s*$|(?:吗|呢)[，,。.！!]")

MAX_TOOL_MEMORY_CHARS = 1200

MAX_TOOL_METADATA_CHARS = 4000

PERSONAL_FACT_RE = re.compile(
    r"我喜欢|我讨厌|我爱|我不喜欢|我叫|我是|我的|我在|我住|我养|我玩|我生日|我习惯|我通常|我平时"
)


RECALL_TIME_WORDS = (
    "上次",
    "以前",
    "之前",
    "曾经",
    "过去",
    "那天",
    "当时",
    "回忆",
    "记得",
    "说过",
)

RECALL_PRONOUNS = ("你", "我", "他", "她", "它", "我们", "你们", "他们")

RECALL_QUESTION_MARKS = (
    "？",
    "?",
    "吗",
    "呢",
    "什么",
    "怎么",
    "哪",
    "谁",
    "多少",
    "是否",
)

EXTRACT_SKIP_PATTERN = re.compile(
    r"^(?:/[^\s]+|https?://\S+|\[Image\]|[\W_]+|\w{1,2})$"
)

RECALL_KEYWORDS = (
    "记得",
    "还记得",
    "之前",
    "上次",
    "我说过",
    "我喜欢",
    "我讨厌",
    "我偏好",
    "以前",
    "曾经",
    "回忆",
    "想起来",
)

CONFIDENCE_TIER_RANGES: dict[str, tuple[float, float]] = {
    "major": (0.90, 1.00),
    "discussion": (0.70, 0.80),
    "daily": (0.50, 0.60),
    "chatter": (0.30, 0.40),
    "spam": (0.00, 0.20),
}

CONFIDENCE_TIER_ALIASES: dict[str, tuple[str, ...]] = {
    "major": (
        "major",
        "major_decision",
        "important",
        "high",
        "重要",
        "重要决策",
        "重大决策",
    ),
    "discussion": ("discussion", "valuable", "讨论", "有价值", "有结论"),
    "daily": ("daily", "daily_discussion", "normal", "日常", "日常讨论", "一般"),
    "chatter": ("chatter", "chat", "casual", "闲聊", "简单闲聊", "随意"),
    "spam": ("spam", "low", "灌水", "无信息", "刷屏", "寒暄"),
}
