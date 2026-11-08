"""插件配置模型。

把 AstrBot 传入的原始 dict 配置统一解析成不可变 dataclass，
并在这里集中处理默认值、类型兼容与范围钳制，避免业务代码到处 .get()。
"""

from __future__ import annotations

from astrbot.api import logger

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_WEB_SEARCH_SYSTEM_PROMPT = (
    "You are a web research assistant. Use live web search/browsing when answering. "
    "Return ONLY a single JSON object with keys: "
    "content (string), sources (array of objects with url/title/snippet when possible). "
    "Keep content concise and evidence-backed. "
    "IMPORTANT: Do NOT use Markdown formatting in the content field - use plain text only."
)
DEFAULT_IMAGE_CAPTION_PROMPT = "Please describe the image using Chinese."


# --------------------------------------------------------------------------- #
# 基础类型转换
# --------------------------------------------------------------------------- #
def _to_bool(value: Any, default: bool) -> bool:
    """将布尔值、常见字符串或数字转换为 bool。

    Args:
        value: 原始配置值。
        default: 输入无法识别时使用的值。
    """
    # 兼容面板、JSON 与历史配置中的布尔写法；不认识的值沿用字段默认值。
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _to_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    """解析整数并限制在闭区间 ``[minimum, maximum]`` 内。"""
    # 将输入安全转换为整数，并统一钳制到配置项允许的闭区间。
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _to_float(
    value: Any,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """解析有限浮点数并限制范围；无效输入返回默认值。"""
    # 过滤非法及非有限浮点数，避免 NaN/无穷值进入运行时配置。
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, min(maximum, parsed))


def _to_str(value: Any, default: str = "") -> str:
    """去除字符串首尾空白，并为 None 或空字符串应用默认值。"""
    # 去除字符串两侧空白；空值回落到调用方提供的默认值。
    text = str(value).strip() if value is not None else ""
    return text or default


def _to_probability(value: Any, default: float) -> float:
    """解析概率值，保证返回结果位于 0 到 1 之间。"""
    # 概率配置限定在 [0, 1]，复用浮点数的统一校验逻辑。
    return _to_float(value, default, 0.05, 1.0)


def _parse_str_list(value: Any) -> tuple[str, ...]:
    """把 list / 逗号分隔字符串统一成去重后的字符串元组。"""
    items: list[str] = []
    if isinstance(value, str):
        items = [token.strip() for token in value.replace("\n", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(token).strip() for token in value]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return tuple(deduped)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """读取配置区块；字段缺失或不是字典时返回空字典。"""
    # 配置区块必须为字典，其他类型视为缺省区块以便后续读取安全。
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


# --------------------------------------------------------------------------- #
# 各配置区块
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActiveReplyConfig:
    """主动发言触发策略和生效群范围。

    ``mode`` 决定使用概率抽样还是模型判定；两种模式都依据
    ``trigger_threshold`` 维护每群消息计数。名单模式为白名单时仅允许
    ``whitelist`` 中的群，为黑名单时除 ``blacklist`` 外均允许。
    """

    enable: bool = False
    mode: str = "probability"
    possibility: float = 0.1
    model_history_messages: int = 0
    model_choice_provider_id: str = ""
    trigger_threshold: int = 10
    list_mode: str = "blacklist"
    blacklist: tuple[str, ...] = ()
    whitelist: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupContextConfig:
    """供模型参考的群聊历史及图片描述设置。

    ``enable`` 控制历史采集/注入，``max_messages`` 限制每个会话保留的消息数；
    发送者 ID 和角色标记用于帮助模型识别发言人。图片描述需要单独开启，
    并可指定用于视觉理解的提供商与提示词。
    """

    enable: bool = True
    max_messages: int = 200
    history_max_chars: int = 5000
    include_sender_id: bool = True
    include_role_tag: bool = True
    image_caption: bool = False
    image_caption_provider_id: str = ""
    image_caption_prompt: str = DEFAULT_IMAGE_CAPTION_PROMPT


@dataclass(frozen=True)
class GroupReplyConfig:
    """群聊互动增强的父开关及子功能开关。

    ``react_pattern`` 是总开关；``role_display`` 与 ``smart_reply`` 分别控制
    是否把成员角色及 @/引用互动规则提供给模型，单独开启子开关不会越过总开关。
    """

    react_pattern: bool = True
    role_display: bool = True
    smart_reply: bool = True


@dataclass(frozen=True)
class UserBlockConfig:
    """用户屏蔽（作用于全局，群级记录由 block_store 管理）。

    ``blacklist`` 默认硬编码 4 个 QQ 是刻意的显式默认（QQ 群管家等平台
    官方/管家账号，避免机器人回应它们的自动消息）；``_conf_schema.json``
    的 ``user_block.blacklist.default`` 与这里保持一致（F18：隐藏默认改成
    面板可见的显式默认，配置缺键时不再出现“面板显示空、实际已屏蔽”）。
    修改默认值时必须同步 schema，``tests/test_config_consistency.py`` 会核对。
    """

    enable: bool = True
    blacklist: tuple[str, ...] = (
        "2854196310",
        "2854197266",
        "3889239358",
        "3889001246",
    )
    block_hint: bool = False


@dataclass(frozen=True)
class WebSearchConfig:
    """联网搜索工具的提供商、协议和结果展示配置。"""

    enable: bool = False
    provider_id: str = ""
    system_prompt: str = DEFAULT_WEB_SEARCH_SYSTEM_PROMPT
    timeout_sec: float = 60.0
    request_mode: str = "auto"
    base_url_override: str = ""
    show_sources: bool = False
    max_sources: int = 5


DEFAULT_GROUP_SUMMARY_PROMPT = """## 总结风格
- 以第一人称、符合你人格的语气记录这段群聊，可以写下你对气氛和参与者的感受。
- 优先保留决策、承诺、日程、偏好与身份信息；流水账、寒暄和复读不必记录。
"""

DEFAULT_PRIVATE_SUMMARY_PROMPT = """## 总结风格
- 以第一人称、符合你人格的语气记录这段私聊，可以写下你对对方状态的理解。
- 优先保留对方明确说过的事实、偏好、约定与计划。
"""

LEGACY_PROMPT_SHA256 = {
    "group": "81ac2f32f5cbd7ba7a5b83f52f9534754e5095804067b8664a20cd25e173d510",
    "private": "1d0742941e5b0d8a358b4b86ebe579fb0af07396900ab7577d42a3eac9488fa7",
}


def is_legacy_default_prompt(kind: str, text: str) -> bool:
    """判断一段提示词是否旧版内置默认（升级时替换成新的补充层默认值）。"""
    digest = hashlib.sha256(str(text or "").strip().encode("utf-8")).hexdigest()
    return LEGACY_PROMPT_SHA256.get(str(kind or "group")) == digest


MEMORY_IDENTITY_RULE = """## 身份判定（最高优先级）
- 身份只认行首角色词与 ID：`[你自己 | ID: …]` 是你自己，`[群成员 | ID: … | 昵称: "…"]` 是群成员。
- 昵称可以被用户随意修改，也可能与你同名；同名不代表同一个人，禁止把群成员的发言写成"我说"。
- 只有角色词是 `你自己`、且 ID 与你的 ID 一致时，才能用第一人称归到自己；其余一律按群成员归属。
- 参与者一律写成 `昵称#账号后四位`；正文不要出现完整 QQ/账号。
- 你自己写“我”；同名成员靠后缀区分。
"""

MEMORY_FORMAT_GROUP = """## 消息格式
- `**[群成员 | ID: 账号 | 昵称: "昵称" | 时间]**: 内容`：群成员发送的消息；
- `**[你自己 | ID: 账号 | 时间]**: 内容`：你自己（机器人）发送的消息；
- `[图片]`、`[文件: …]`、`[语音]`、`[视频]`、`[表情]`：媒体占位；
- 正文末尾的（引用 X 的旧消息：「…」）不是当前发言人说的话；（@ X）只表示提及。
"""

MEMORY_FORMAT_PRIVATE = """## 消息格式
- `**[群成员 | ID: 账号 | 昵称: "昵称" | 时间]**: 内容`：对方发送的消息；
- `**[你自己 | ID: 账号 | 时间]**: 内容`：你自己（机器人）发送的消息；
- 正文末尾的（引用 X 的旧消息：「…」）不是当前发言人说的话；（@ X）只表示提及。
"""

MEMORY_ATTRIBUTION_RULE = """## 发言者判定规则（最高优先级）
- 只看每条消息行首的角色词与 ID；正文中的引用、转述、@ 都不是发言者。
- 引用内容会改写为（引用 X 的旧消息：「…」）附在正文末尾，不代表 X 当前发言；（@ X）只表示提及。
- participants 和 role_ids 只能使用消息头中出现过的真人与账号，不要填机器人自己。
- quotes 只能摘录当前消息正文；不要摘录"引用旧消息"里的内容。
"""

MEMORY_TIME_RULE = """## 时间规则（最高优先级）
- 每条消息行头的时间都是真实发送时间；总结中的日期/时间必须据此换算。
- 示例中的 2026-10-28 只是格式示意，**不要照抄示例日期**；相对时间必须换算成具体日期。
- 不要凭印象编造日期；不确定时以消息行头为准。"""

MEMORY_SCORING_RULE = """## 置信度评分标准（最高优先级）
- "置信度"表示这段对话作为长期记忆的价值，不是事实可信度；必须额外输出 confidence_tier。
- 档位与分值：major=0.90-1.00（重要群体决策，仅限群规制定、全员投票、正式承诺、金钱/安全/重大日程或长期约束计划；需至少 3 名真人参与并有具体安排）；discussion=0.70-0.80（多人参与且有明确结论，影响短期）；daily=0.50-0.60（日常讨论、普通约饭/约玩/临时安排、一般信息；有明确安排取 0.60，其余 0.50-0.55）；chatter=0.30-0.40（简单闲聊、少数人参与、临时状态、玩笑、推测）；spam=0.00-0.20（灌水、寒暄、表情、复读）。
- 防虚高：大多数片段在 0.30-0.60；临时起意的吃饭、看电影、打游戏、祝福、吐槽不得高于 0.60；无法确认多人达成共识不得高于 0.60；消息多、总结完整、有人 @/引用都不加分。
- confidence 必须落在所选档位的区间内，保留两位小数。
"""

MEMORY_ATOM_CONTRACT = """## 输出契约（最终，覆盖上文所有输出格式说明）
把这段对话整理成 1-{max_items} 条记忆：**第 1 条优先 episode**（150–240 字，保留时间、人物、因果与气氛），其余为原子事实（40–80 字，每条自包含）。
只输出一个 JSON 对象，不要数组、解释或代码块：
{"memories": [
  {"content": "第 1 条 episode 150–240 字；原子事实 40–80 字；第一人称视角",
   "brief": "不超过 60 字，供注入卡片使用的摘要",
   "kind": "fact|preference|plan|episode|relation",
   "entities": ["人名/地点/主题，最多 8 个"],
   "topics": ["主题"],
   "participants": ["昵称"],
   "role_ids": ["QQ号"],
   "scope": "global|group",
   "confidence_tier": "major|discussion|daily|chatter|spam",
   "confidence": 0.0,
   "valid_until": "YYYY-MM-DD 或空字符串",
   "quotes": ["逐字来自聊天片段的证据"]}
]}
规则：
- 第 1 条把“发生了什么、谁说了什么、气氛如何”写成可读的 episode；后续只放可独立检索的事实/决定/偏好/计划/关系，不要把 episode 拆碎。
- kind：约定/行程=plan，长期口味或习惯=preference，今天发生的事=episode，群规/身份/长期事实=fact，谁和谁的关系=relation。
- scope 只在「自动」作用域下参考：个人偏好/身份/习惯用 global，只与这个群有关的事用 group。
- participants 是显示名，身份以 role_ids 为准；只填片段中出现过的真人。
- 正文提到参与者时用 `昵称#后四位`，不要写完整 QQ/账号。
- 没有值得长期保留的内容时输出 {"memories": []}，不要为了凑数编造。
- quotes 必须逐字来自对话正文，至少 1 条；证据不足的条目会被整条丢弃。
- 相对时间按消息行头换算成具体日期；valid_until 只用于 plan 类，其他类型留空字符串。
禁止输出 {"summary": ...} 单对象格式。"""

MEMORY_EPISODE_CONTRACT = """## 输出契约（第一步：只写 episode）
只输出一个 JSON 对象，不要数组、解释或代码块：
{"memories": [
  {"content": "150–180 字，第一人称，保留时间/人物/因果/气氛；最多 200 字，超过会被丢弃",
   "brief": "不超过 40 字，供注入卡片",
   "participants": ["昵称"],
   "role_ids": ["QQ号"],
   "scope": "global|group",
   "confidence_tier": "major|discussion|daily|chatter|spam",
   "confidence": 0.0,
   "quotes": ["逐字来自聊天片段的证据，至少 1 条"]}
]}
规则：
- 只输出 1 条 episode；不要把可独立检索的事实拆进 episode 正文。
- content 必须 ≤200 字；宁可省略细节，也不要写长；超过 200 字会被解析器丢弃。
- 正文提到参与者时用 `昵称#后四位`，不要写完整 QQ/账号。
- 没有值得保留的内容时输出 {"memories": []}，不要编造。
- 禁止输出 {"summary": ...} 单对象格式。"""

MEMORY_ATOMS_CONTRACT = """## 输出契约（第二步：只写原子事实）
只输出一个 JSON 对象，不要数组、解释或代码块：
{"memories": [
  {"content": "40–80 字，一条只表达一个事实/决定/偏好/计划/关系，自包含",
   "brief": "不超过 60 字，供注入卡片",
   "kind": "fact|preference|plan|relation",
   "entities": ["人名/地点/主题，最多 8 个"],
   "participants": ["昵称"],
   "role_ids": ["QQ号"],
   "scope": "global|group",
   "confidence_tier": "major|discussion|daily|chatter|spam",
   "confidence": 0.0,
   "valid_until": "YYYY-MM-DD 或空字符串",
   "quotes": ["逐字来自聊天片段的证据，至少 1 条"]}
]}
规则：
- 最多输出 {max_items} 条；没有独立事实就输出 {"memories": []}，不要为凑数重复 episode。
- 不要输出 kind=episode；episode 已由第一步写入。
- 正文提到参与者时用 `昵称#后四位`，不要写完整 QQ/账号。
- quotes 至少 1 条，证据不足的条目会被丢弃。
- 禁止输出 {"summary": ...} 单对象格式。"""

SINGLE_MEMORY_CONTRACT = """## 输出契约（最终，覆盖上文所有输出格式说明）
只输出一个 JSON 对象，不要数组、解释或代码块：
{"summary": "整段对话的总结，第一人称，不超过 240 字",
 "brief": "不超过 60 字，供注入卡片使用的摘要",
 "topics": ["主题"],
 "participants": ["昵称"],
 "role_ids": ["QQ号"],
 "scope": "global|group",
 "confidence_tier": "major|discussion|daily|chatter|spam",
 "confidence": 0.0,
 "valid_until": "YYYY-MM-DD 或空字符串",
 "quotes": ["逐字来自聊天片段的证据"]}
规则：没有值得保留的内容时输出 {"summary": ""}；quotes 至少 1 条，证据不足整条丢弃。"""


def build_memory_prompt_parts(
    prompt: str,
    *,
    kind: str = "group",
    multi_atom: bool = True,
    max_items: int = 3,
    contract_override: str = "",
) -> dict[str, str]:
    """把用户补充层与系统固定层拼成最终提示词。

    Args:
        prompt: 用户在配置里写的补充要求（可为空，空时用内置默认）。
        kind: ``group`` 或 ``private``，决定消息格式说明。
        multi_atom: 是否使用多条原子记忆的输出契约。
        max_items: 单次输出上限。

    Returns:
        ``{"supplement": ..., "rules": ..., "template": ...}``；``template``
        里保留 ``{messages}`` 占位符，由调用方替换成当前时间与聊天片段。
    """
    private = str(kind or "").strip().lower() == "private"
    default = (
        DEFAULT_PRIVATE_SUMMARY_PROMPT if private else DEFAULT_GROUP_SUMMARY_PROMPT
    )
    supplement = str(prompt or "").strip()
    if not supplement or is_legacy_default_prompt(
        "private" if private else "group", supplement
    ):
        supplement = default.strip()
    supplement = supplement.replace("{messages}", "").strip()
    contract = str(contract_override or "").strip()
    if not contract:
        contract = MEMORY_ATOM_CONTRACT if multi_atom else SINGLE_MEMORY_CONTRACT
    contract = contract.replace("{max_items}", str(max(1, min(5, int(max_items)))))
    format_rule = MEMORY_FORMAT_PRIVATE if private else MEMORY_FORMAT_GROUP
    blocks = [
        "## 你的补充要求",
        supplement,
        MEMORY_IDENTITY_RULE,
        format_rule,
        MEMORY_ATTRIBUTION_RULE,
        MEMORY_TIME_RULE,
        MEMORY_SCORING_RULE,
        contract,
        "## 聊天片段",
        "{messages}",
    ]
    return {
        "supplement": supplement,
        "rules": "\n\n".join(blocks[2:-2]),
        "template": "\n\n".join(blocks),
    }


def compose_memory_prompt(
    prompt: str,
    *,
    kind: str = "group",
    multi_atom: bool = True,
    max_items: int = 3,
    contract_override: str = "",
) -> str:
    """返回最终提示词模板（含 ``{messages}`` 占位符）。"""
    return build_memory_prompt_parts(
        prompt,
        kind=kind,
        multi_atom=multi_atom,
        max_items=max_items,
        contract_override=contract_override,
    )["template"]


PROMPT_VERSION = 5
"""记忆抽取提示词的契约版本；写入记忆元数据用于追溯。"""


@dataclass(frozen=True)
class KindDecayConfig:
    """按记忆类型区分的半衰期（天）与硬过期策略。

    事实类长期有效，计划类很快失效；单一半衰期会让三年前的约定和上周的偏好
    同速变淡，因此按类型分别配置。
    """

    plan: int = 3
    episode: int = 7
    preference: int = 60
    fact: int = 120
    relation: int = 90

    _EXPIRY_DAYS = {"plan": 7, "episode": 30, "preference": 365, "relation": 365}

    def half_life(self, kind: str) -> int:
        """返回该类型的半衰期天数，未知类型按 fact 处理。"""
        return int(getattr(self, str(kind or "fact"), self.fact) or self.fact)

    def expiry_days(self, kind: str) -> int | None:
        """返回该类型的硬过期天数；``fact`` 永不过期。"""
        return self._EXPIRY_DAYS.get(str(kind or "fact"))


@dataclass(frozen=True)
class KindTtlConfig:
    """按记忆类型区分的自适应 TTL 基准天数（r3.3）。

    写入时按 ``base × (0.5 + importance/10) × 强化系数`` 计算有效期；计划类很快
    过期，事实类长期有效；``0`` 表示该类型永不过期（不写 valid_until）。
    """

    plan: int = 2
    episode: int = 7
    preference: int = 60
    relation: int = 90
    fact: int = 180

    def base_days(self, kind: str) -> int:
        """返回该类型的基础有效期天数；未知类型按 fact，0 表示永不过期。"""
        value = getattr(self, str(kind or "fact"), self.fact)
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 180


@dataclass(frozen=True)
class MemoryRagConfig:
    """长期记忆检索及自动写入/抽取策略。

    ``recall_k`` 限制语义召回候选数，``max_return_results`` 限制工具返回上限；
    自动记忆由关键词快速写入和按消息窗口调用模型抽取两条路径组成。抽取配置
    分别控制模型、消息频率、最小间隔、历史窗口、置信度门槛及群/全局作用域。
    """

    enable: bool = False
    embedding_provider_id: str = ""
    auto_extract_provider_id: str = ""
    recall_k: int = 20
    max_return_results: int = 100
    # 记忆生成
    auto_extract_enable: bool = True
    batch_messages: int = 50
    auto_extract_interval_minutes: int = 10
    private_extract_every_n: int = 10
    extract_max_chars: int = 6000
    window_gap_minutes: int = 60
    group_prompt: str = DEFAULT_GROUP_SUMMARY_PROMPT
    private_prompt: str = DEFAULT_PRIVATE_SUMMARY_PROMPT
    auto_extract_min_confidence: float = 0.5
    auto_extract_scope: str = "group_first"
    dedup_threshold: float = 0.90
    conflict_threshold: float = 0.75
    # 可见性：全局记忆随参与者走，私聊默认不跨会话
    cross_group_personal: bool = True
    cross_group_private: bool = False
    # 读取与生命周期
    inject_max_chars: int = 1500
    inject_max_items: int = 5
    multi_atom: bool = True
    atom_max_items: int = 3
    prompt_contract: bool = True
    kind_decay: KindDecayConfig = field(default_factory=KindDecayConfig)
    mmr_enabled: bool = True
    kind_badge: bool = True
    entity_boost: float = 0.10
    online_scan_limit: int = 900
    min_similarity: float = 0.55
    lexical_recall: bool = True
    embedding_cache_ttl: int = 300
    embedding_cache_size: int = 256
    recall_cache_ttl: int = 45
    recall_cache_entries: int = 256
    consolidation_enable: bool = True
    consolidation_interval_days: int = 7
    consolidation_archive_only: bool = True
    archive_enable: bool = True
    archive_days: int = 30
    archive_importance: float = 3.5
    trash_retention_days: int = 30
    # 遗忘机制（r3.3）：自适应 TTL、重要性随时间演化与归档区清理
    ttl_adaptive_enable: bool = True
    ttl_base: KindTtlConfig = field(default_factory=KindTtlConfig)
    # 预留：当前写入时 access_count=0，强化系数恒 1.0；未在 schema 暴露，
    # 避免界面出现无效配置项（r3.3m）。
    ttl_reinforce_step: float = 0.1
    ttl_reinforce_cap: float = 0.5
    importance_decay_enable: bool = True
    importance_decay_rate: float = 0.01
    importance_decay_floor: float = 0.5
    importance_decay_access_window_days: int = 30
    importance_decay_access_max_count: int = 10
    importance_protected_threshold: float = 10.0
    archive_retention_days: int = 0
    # 抽取质量（r3.3）
    extract_min_substance_chars: int = 8
    max_participants: int = 10


@dataclass(frozen=True)
class RuntimeConfig:
    """运行时内存容量及外部模型调用超时配置。"""

    max_origins: int = 500
    image_caption_timeout: float = 45.0
    model_choice_timeout: float = 45.0
    embedding_timeout: float = 8.0
    embedding_write_timeout: float = 12.0
    recall_deadline: float = 3.0


@dataclass(frozen=True)
class PluginConfig:
    """经过规范化的插件配置快照。

    各字段对应一个不可变配置区块，避免业务逻辑直接操作原始 JSON 字典。
    ``default_factory`` 为每个配置对象分别创建缺省子配置，避免实例之间共享状态。
    ``react_enabled`` 等属性是底层开关组合出的便捷视图，
    不会在解析过程中额外存储第二份可变状态。
    """

    active_reply: ActiveReplyConfig = field(default_factory=ActiveReplyConfig)
    group_context: GroupContextConfig = field(default_factory=GroupContextConfig)
    group_reply: GroupReplyConfig = field(default_factory=GroupReplyConfig)
    user_block: UserBlockConfig = field(default_factory=UserBlockConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    memory_rag: MemoryRagConfig = field(default_factory=MemoryRagConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ---- 派生属性 ------------------------------------------------------- #
    @property
    def react_enabled(self) -> bool:
        """群聊回复增强整块是否生效（由 React Pattern 总开关控制）。"""
        return self.group_reply.react_pattern

    @property
    def smart_reply_enabled(self) -> bool:
        """返回 React Pattern 与智能回复子开关同时开启的状态。"""
        # React Pattern 是父级开关；子级智能回复开关不能单独越过它生效。
        return self.react_enabled and self.group_reply.smart_reply

    @property
    def role_display_enabled(self) -> bool:
        """返回 React Pattern 与用户角色显示子开关同时开启的状态。"""
        # 角色提示同样受 React Pattern 总开关约束。
        return self.react_enabled and self.group_reply.role_display


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def parse_plugin_config(raw: dict[str, Any] | None) -> PluginConfig:
    """将 AstrBot 原始配置映射成完整、不可变且范围受限的配置快照。

    每个区块先通过 ``_section`` 确认是字典，再使用对应的类型转换函数读取
    字段；缺失或无法识别的值回退到插件默认值，超出合法范围的数值则被钳制。
    枚举字段会映射到支持值，名单会拆分并去重。此函数不修改传入字典，
    因而调用方可在每次配置更新后重新解析并安全地将结果交给业务代码。

    Args:
        raw: AstrBot 加载的插件配置；``None`` 或非字典区块按默认值处理。

    Returns:
        包含所有配置区块和派生开关的 ``PluginConfig`` 实例。
    """
    raw = raw or {}

    ar_raw = _section(raw, "active_reply")
    mode = _to_str(ar_raw.get("mode"), "probability").lower()
    if mode not in {"probability", "model_choice"}:
        mode = "probability"
    list_mode = _to_str(ar_raw.get("list_mode"), "blacklist").lower()
    if list_mode not in {"blacklist", "whitelist"}:
        list_mode = "blacklist"
    active_reply = ActiveReplyConfig(
        enable=_to_bool(ar_raw.get("enable"), False),
        mode=mode,
        possibility=_to_probability(ar_raw.get("possibility"), 0.1),
        model_history_messages=_to_int(ar_raw.get("model_history_messages"), 0, 0, 100),
        model_choice_provider_id=_to_str(ar_raw.get("model_choice_provider_id")),
        trigger_threshold=_to_int(ar_raw.get("trigger_threshold"), 10, 2, 50),
        list_mode=list_mode,
        blacklist=_parse_str_list(ar_raw.get("blacklist")),
        whitelist=_parse_str_list(ar_raw.get("whitelist")),
    )

    ctx_raw = _section(raw, "group_context_enhancement")
    group_context = GroupContextConfig(
        enable=_to_bool(ctx_raw.get("enable"), True),
        max_messages=_to_int(ctx_raw.get("max_messages"), 200, 10, 1000),
        history_max_chars=_to_int(ctx_raw.get("history_max_chars"), 5000, 500, 20000),
        include_sender_id=_to_bool(ctx_raw.get("include_sender_id"), True),
        include_role_tag=_to_bool(ctx_raw.get("include_role_tag"), True),
        image_caption=_to_bool(ctx_raw.get("image_caption"), False),
        image_caption_provider_id=_to_str(ctx_raw.get("image_caption_provider_id")),
        image_caption_prompt=_to_str(
            ctx_raw.get("image_caption_prompt"), DEFAULT_IMAGE_CAPTION_PROMPT
        ),
    )

    reply_raw = _section(raw, "group_reply_enhancement")
    group_reply = GroupReplyConfig(
        react_pattern=_to_bool(reply_raw.get("react_pattern"), True),
        role_display=_to_bool(reply_raw.get("role_display"), True),
        smart_reply=_to_bool(reply_raw.get("smart_reply"), True),
    )
    block_raw = _section(reply_raw, "user_block")
    has_blacklist = "blacklist" in block_raw
    block_ids = block_raw.get("blacklist")
    if isinstance(block_ids, str):
        block_ids = [part for part in block_ids.replace("\n", ",").split(",")]
    if not isinstance(block_ids, (list, tuple)):
        block_ids = []
    blacklist = tuple(
        str(item).strip()
        for item in block_ids
        if str(item).strip().isalnum()
        and str(item).strip().isascii()
        and len(str(item).strip()) <= 64
    )
    user_block = UserBlockConfig(
        enable=_to_bool(block_raw.get("enable"), True),
        blacklist=blacklist if has_blacklist else UserBlockConfig().blacklist,
        block_hint=_to_bool(block_raw.get("block_hint"), False),
    )

    ws_raw = _section(raw, "web_search")
    request_mode = _to_str(ws_raw.get("request_mode"), "auto").lower()
    if request_mode not in {"auto", "responses", "chat_completions"}:
        request_mode = "auto"
    web_search = WebSearchConfig(
        enable=_to_bool(ws_raw.get("enable"), False),
        provider_id=_to_str(ws_raw.get("provider_id")),
        system_prompt=_to_str(
            ws_raw.get("system_prompt"), DEFAULT_WEB_SEARCH_SYSTEM_PROMPT
        ),
        timeout_sec=_to_float(ws_raw.get("timeout_sec"), 60.0, 30.0, 180.0),
        request_mode=request_mode,
        base_url_override=_to_str(ws_raw.get("base_url_override")),
        show_sources=_to_bool(ws_raw.get("show_sources"), False),
        max_sources=_to_int(ws_raw.get("max_sources"), 5, 1, 20),
    )

    mem_raw = _section(raw, "memory_rag")
    control_raw = _section(mem_raw, "memory_control") or mem_raw
    gen_raw = _section(mem_raw, "memory_generation") or mem_raw
    inj_raw = _section(mem_raw, "memory_injection") or mem_raw
    eval_raw = _section(mem_raw, "memory_evaluation") or mem_raw
    # r3.3n：性能/缓存/高级调参已从「记忆 RAG」迁移到「调用配置 → 记忆检索与缓存调优」；
    # 这里优先读新位置，并回退旧的 memory_rag/memory_injection 等键，保证旧配置兼容。
    tuned_raw = _section(_section(raw, "runtime"), "memory_tuning") or {}

    def _tuned(name: str, *sources: Any) -> Any:
        """按「新位置 → 传入的旧位置」顺序取配置值，全部缺失返回 None。"""
        if isinstance(tuned_raw, dict) and name in tuned_raw:
            return tuned_raw.get(name)
        for source in sources:
            if isinstance(source, dict) and name in source:
                return source.get(name)
        return None

    scope_value = str(
        gen_raw.get("auto_extract_scope")
        or mem_raw.get("auto_extract_scope")
        or "group_first"
    ).strip()
    if scope_value not in {"group_first", "global", "group"}:
        scope_value = "group_first"
    # 旧配置里 memory_kind_decay 可能是字符串/None；先收敛成 dict，
    # 否则下面的 .get() 会在每次事件解析时抛 AttributeError。
    kind_decay_raw = eval_raw.get("memory_kind_decay")
    if not isinstance(kind_decay_raw, dict):
        kind_decay_raw = {}
    # r3.3：自适应 TTL 基准也是嵌套对象，同样先收敛成 dict 再读。
    ttl_base_raw = eval_raw.get("memory_ttl_base") or mem_raw.get("memory_ttl_base")
    if not isinstance(ttl_base_raw, dict):
        ttl_base_raw = {}
    memory_rag = MemoryRagConfig(
        enable=_to_bool(control_raw.get("enable", mem_raw.get("enable")), False),
        cross_group_personal=_to_bool(
            control_raw.get(
                "cross_group_personal", mem_raw.get("cross_group_personal")
            ),
            True,
        ),
        cross_group_private=_to_bool(
            control_raw.get("cross_group_private", mem_raw.get("cross_group_private")),
            False,
        ),
        embedding_provider_id=_to_str(
            control_raw.get(
                "embedding_provider_id", mem_raw.get("embedding_provider_id")
            )
        ),
        auto_extract_provider_id=_to_str(
            control_raw.get(
                "auto_extract_provider_id", mem_raw.get("auto_extract_provider_id")
            )
        ),
        recall_k=_to_int(_tuned("recall_k", inj_raw, mem_raw), 20, 1, 50),
        max_return_results=_to_int(
            _tuned("max_return_results", inj_raw, mem_raw), 100, 1, 300
        ),
        auto_extract_enable=_to_bool(
            gen_raw.get("auto_extract_enable", mem_raw.get("auto_extract_enable")), True
        ),
        batch_messages=_to_int(
            gen_raw.get(
                "memory_batch_messages",
                mem_raw.get("memory_batch_messages")
                or mem_raw.get("auto_extract_window")
                or 50,
            ),
            50,
            10,
            100,
        ),
        auto_extract_interval_minutes=_to_int(
            gen_raw.get(
                "auto_extract_interval_minutes",
                mem_raw.get("auto_extract_interval_minutes"),
            ),
            10,
            1,
            120,
        ),
        private_extract_every_n=_to_int(
            gen_raw.get(
                "private_extract_every_n", mem_raw.get("private_extract_every_n")
            ),
            10,
            3,
            100,
        ),
        extract_max_chars=_to_int(
            gen_raw.get(
                "memory_extract_max_chars", mem_raw.get("memory_extract_max_chars")
            ),
            6000,
            500,
            10000,
        ),
        window_gap_minutes=_to_int(
            _tuned("memory_window_gap_minutes", gen_raw, mem_raw), 60, 0, 240
        ),
        dedup_threshold=_to_float(
            _tuned("memory_dedup_threshold", gen_raw, mem_raw), 0.90, 0.80, 0.99
        ),
        conflict_threshold=_to_float(
            _tuned("memory_conflict_threshold", gen_raw, mem_raw),
            0.75,
            0.50,
            0.89,
        ),
        auto_extract_min_confidence=_to_float(
            gen_raw.get(
                "auto_extract_min_confidence",
                mem_raw.get("auto_extract_min_confidence"),
            ),
            0.5,
            0.0,
            1.0,
        ),
        auto_extract_scope=scope_value,
        group_prompt=_to_str(
            gen_raw.get("memory_group_prompt", mem_raw.get("memory_group_prompt")),
            DEFAULT_GROUP_SUMMARY_PROMPT,
        ),
        private_prompt=_to_str(
            gen_raw.get("memory_private_prompt", mem_raw.get("memory_private_prompt")),
            DEFAULT_PRIVATE_SUMMARY_PROMPT,
        ),
        inject_max_chars=_to_int(
            inj_raw.get(
                "memory_inject_max_chars", mem_raw.get("memory_inject_max_chars")
            ),
            1500,
            500,
            5000,
        ),
        inject_max_items=_to_int(
            inj_raw.get(
                "memory_inject_max_items", mem_raw.get("memory_inject_max_items")
            ),
            5,
            0,
            15,
        ),
        multi_atom=_to_bool(gen_raw.get("memory_multi_atom"), True),
        atom_max_items=_to_int(gen_raw.get("memory_atom_max_items"), 3, 1, 5),
        # 旧版 memory_prompt_contract 已从 schema 移除；契约固定启用，
        # 避免预览与实际抽取行为不一致。
        prompt_contract=True,
        kind_decay=KindDecayConfig(
            plan=_to_int(
                kind_decay_raw.get("plan_half_life_days"),
                3,
                1,
                30,
            ),
            episode=_to_int(
                kind_decay_raw.get("episode_half_life_days"),
                7,
                1,
                60,
            ),
            preference=_to_int(
                kind_decay_raw.get("preference_half_life_days"),
                60,
                7,
                365,
            ),
            fact=_to_int(
                kind_decay_raw.get("fact_half_life_days"),
                120,
                30,
                365,
            ),
            relation=_to_int(
                kind_decay_raw.get("relation_half_life_days"),
                90,
                30,
                365,
            ),
        ),
        mmr_enabled=_to_bool(_tuned("memory_mmr_enabled", inj_raw, mem_raw), True),
        kind_badge=_to_bool(_tuned("memory_kind_badge", inj_raw, mem_raw), True),
        entity_boost=_to_float(
            _tuned("memory_entity_boost", inj_raw, mem_raw), 0.10, 0.0, 0.5
        ),
        online_scan_limit=_to_int(
            _tuned("memory_online_scan_limit", inj_raw, mem_raw), 900, 100, 5000
        ),
        min_similarity=_to_float(
            _tuned("memory_min_similarity", inj_raw, mem_raw), 0.55, 0.0, 1.0
        ),
        lexical_recall=_to_bool(
            _tuned("memory_lexical_recall", inj_raw, mem_raw), True
        ),
        embedding_cache_ttl=_to_int(
            _tuned("memory_embedding_cache_ttl", inj_raw, mem_raw), 300, 0, 3600
        ),
        embedding_cache_size=_to_int(
            _tuned("memory_embedding_cache_size", inj_raw, mem_raw), 256, 0, 2048
        ),
        recall_cache_ttl=_to_int(
            _tuned("memory_recall_cache_ttl_sec", inj_raw, mem_raw), 45, 0, 600
        ),
        recall_cache_entries=_to_int(
            _tuned("memory_recall_cache_entries", inj_raw, mem_raw), 256, 8, 2048
        ),
        consolidation_enable=_to_bool(
            eval_raw.get(
                "memory_consolidation_enable",
                mem_raw.get("memory_consolidation_enable"),
            ),
            True,
        ),
        consolidation_interval_days=_to_int(
            eval_raw.get(
                "memory_consolidation_interval_days",
                mem_raw.get("memory_consolidation_interval_days"),
            ),
            7,
            1,
            30,
        ),
        consolidation_archive_only=_to_bool(
            eval_raw.get(
                "memory_consolidation_archive",
                mem_raw.get("memory_consolidation_archive"),
            ),
            True,
        ),
        archive_enable=_to_bool(
            eval_raw.get("memory_archive_enable", mem_raw.get("memory_archive_enable")),
            True,
        ),
        archive_days=_to_int(
            eval_raw.get("memory_archive_days", mem_raw.get("memory_archive_days")),
            30,
            7,
            120,
        ),
        archive_importance=_to_float(
            eval_raw.get(
                "memory_archive_importance", mem_raw.get("memory_archive_importance")
            ),
            3.5,
            0.0,
            10.0,
        ),
        trash_retention_days=_to_int(
            eval_raw.get(
                "memory_trash_retention_days",
                mem_raw.get("memory_trash_retention_days"),
            ),
            30,
            7,
            365,
        ),
        # r3.3 遗忘机制：自适应 TTL（写入侧）与重要性演化（查询侧）。
        ttl_adaptive_enable=_to_bool(
            eval_raw.get("memory_ttl_adaptive", mem_raw.get("memory_ttl_adaptive")),
            True,
        ),
        ttl_base=KindTtlConfig(
            plan=_to_int(ttl_base_raw.get("plan_days"), 2, 0, 365),
            episode=_to_int(ttl_base_raw.get("episode_days"), 7, 0, 365),
            preference=_to_int(ttl_base_raw.get("preference_days"), 60, 0, 365),
            relation=_to_int(ttl_base_raw.get("relation_days"), 90, 0, 365),
            fact=_to_int(ttl_base_raw.get("fact_days"), 180, 0, 365),
        ),
        ttl_reinforce_step=_to_float(
            eval_raw.get("memory_ttl_reinforce_step"), 0.1, 0.0, 0.5
        ),
        ttl_reinforce_cap=_to_float(
            eval_raw.get("memory_ttl_reinforce_cap"), 0.5, 0.0, 1.0
        ),
        importance_decay_enable=_to_bool(
            _tuned("memory_importance_decay", eval_raw, mem_raw), True
        ),
        importance_decay_rate=_to_float(
            _tuned("memory_importance_decay_rate", eval_raw, mem_raw), 0.01, 0.0, 0.2
        ),
        importance_decay_floor=_to_float(
            _tuned("memory_importance_decay_floor", eval_raw, mem_raw),
            0.5,
            0.0,
            5.0,
        ),
        importance_decay_access_window_days=_to_int(
            _tuned("memory_importance_decay_access_window_days", eval_raw, mem_raw),
            30,
            1,
            365,
        ),
        importance_decay_access_max_count=_to_int(
            _tuned("memory_importance_decay_access_max_count", eval_raw, mem_raw),
            10,
            1,
            100,
        ),
        importance_protected_threshold=_to_float(
            _tuned("memory_importance_protected_threshold", eval_raw, mem_raw),
            10.0,
            5.0,
            10.0,
        ),
        archive_retention_days=_to_int(
            _tuned("memory_archive_retention_days", eval_raw, mem_raw), 0, 0, 3650
        ),
        # r3.3 抽取质量：无实质内容过滤、参与者上限。
        extract_min_substance_chars=_to_int(
            gen_raw.get(
                "memory_extract_min_substance_chars",
                mem_raw.get("memory_extract_min_substance_chars"),
            ),
            8,
            0,
            100,
        ),
        max_participants=_to_int(
            gen_raw.get(
                "memory_max_participants", mem_raw.get("memory_max_participants")
            ),
            10,
            1,
            30,
        ),
    )

    rt_raw = _section(raw, "runtime")
    lru_raw = _section(rt_raw, "lru_cache")
    timeouts_raw = _section(rt_raw, "timeouts")
    runtime = RuntimeConfig(
        max_origins=_to_int(lru_raw.get("max_origins"), 500, 50, 500),
        image_caption_timeout=_to_float(
            timeouts_raw.get("image_caption_sec"), 45.0, 10.0, 120.0
        ),
        model_choice_timeout=_to_float(
            timeouts_raw.get("model_choice_sec"), 45.0, 10.0, 120.0
        ),
        embedding_timeout=_to_float(timeouts_raw.get("embedding_sec"), 8.0, 0.1, 60.0),
        embedding_write_timeout=_to_float(
            timeouts_raw.get("embedding_write_sec"), 12.0, 0.1, 120.0
        ),
        recall_deadline=_to_float(
            timeouts_raw.get("recall_deadline_sec"), 3.0, 0.1, 15.0
        ),
    )

    return PluginConfig(
        active_reply=active_reply,
        group_context=group_context,
        group_reply=group_reply,
        user_block=user_block,
        web_search=web_search,
        memory_rag=memory_rag,
        runtime=runtime,
    )


# --------------------------------------------------------------------------- #
# 每群独立配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroupFieldSpec:
    """每群独立配置中允许覆盖的字段描述。"""

    path: str
    label: str
    kind: str  # bool / enum / int / float / str / text
    options: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    minimum: float = 0
    maximum: float = 0
    step: float = 1
    default: Any = None


@lru_cache(maxsize=1)
def _config_schema() -> dict[str, Any]:
    """读取 schema 一次；缺失时返回空对象，避免配置模块导入失败。"""
    try:
        data = json.loads(
            Path(__file__).with_name("_conf_schema.json").read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _schema_node(path: str) -> dict[str, Any] | None:
    """按点分路径读取 _conf_schema.json 的字段节点。"""
    node: Any = _config_schema()
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        items = node.get("items") if isinstance(node.get("items"), dict) else node
        if part not in items:
            return None
        node = items[part]
    return node if isinstance(node, dict) else None


def _group_spec(path: str) -> GroupFieldSpec:
    """从 schema 派生群字段的标签/类型/范围/枚举，代码只保留路径白名单。"""
    node = _schema_node(path) or {}
    raw_type = str(node.get("type") or "str")
    options = tuple(str(item) for item in (node.get("options") or ()))
    # 普通 string 不强制转 enum：provider ID 等字段没有 options，必须保留 str。
    kind = "enum" if options else raw_type
    if kind not in {"bool", "enum", "int", "float", "str", "text"}:
        kind = "str"
    slider = node.get("slider") if isinstance(node.get("slider"), dict) else {}
    return GroupFieldSpec(
        path=path,
        label=str(node.get("description") or path.rsplit(".", 1)[-1]),
        kind=kind,
        options=options,
        labels=tuple(str(item) for item in (node.get("labels") or ())),
        minimum=float(slider.get("min", 0) or 0),
        maximum=float(slider.get("max", 0) or 0),
        step=float(slider.get("step", 1) or 1),
        default=node.get("default"),
    )


# 说明：只允许覆盖与"群内行为"相关的字段（含记忆 RAG）；搜索/超时属于全局服务配置。
# 路径白名单是唯一需要手写的部分，其余元数据全部来自 _conf_schema.json。
_GROUP_OVERRIDE_PATHS: tuple[str, ...] = (
    "active_reply.enable",
    "active_reply.mode",
    "active_reply.possibility",
    "active_reply.model_history_messages",
    "active_reply.model_choice_provider_id",
    "active_reply.trigger_threshold",
    "group_context_enhancement.enable",
    "group_context_enhancement.max_messages",
    "group_context_enhancement.include_sender_id",
    "group_context_enhancement.include_role_tag",
    "group_context_enhancement.image_caption",
    "group_context_enhancement.image_caption_provider_id",
    "group_context_enhancement.image_caption_prompt",
    "group_reply_enhancement.react_pattern",
    "group_reply_enhancement.role_display",
    "group_reply_enhancement.smart_reply",
    "memory_rag.memory_control.enable",
    "memory_rag.memory_generation.auto_extract_enable",
    "memory_rag.memory_generation.memory_batch_messages",
    "memory_rag.memory_generation.auto_extract_interval_minutes",
    "memory_rag.memory_generation.auto_extract_scope",
    "memory_rag.memory_injection.memory_inject_max_chars",
    "memory_rag.memory_injection.memory_inject_max_items",
)


GROUP_OVERRIDE_FIELDS: tuple[GroupFieldSpec, ...] = tuple(
    _group_spec(path) for path in _GROUP_OVERRIDE_PATHS
)

GROUP_FIELD_BY_PATH = {spec.path: spec for spec in GROUP_OVERRIDE_FIELDS}


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置 dict，overrides 优先。"""
    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_by_path(data: dict[str, Any], path: str) -> Any:
    """沿点分隔路径读取嵌套字典；路径不存在时返回 None。

    Args:
        data: 根配置字典。
        path: 例如 ``active_reply.enable``。
    """
    # 按点分隔的路径读取嵌套配置；路径任一层不存在时返回 None。
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def set_by_path(data: dict[str, Any], path: str, value: Any) -> None:
    """按点分隔路径创建中间字典并写入配置值。"""
    # 按点分隔路径逐层创建字典节点，再写入字段值。
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        node = current.get(key)
        if not isinstance(node, dict):
            node = {}
            current[key] = node
        current = node
    current[keys[-1]] = value


def coerce_group_value(spec: GroupFieldSpec, value: Any) -> Any:
    """按字段描述把页面提交的值转换为合法配置值，非法值回落到默认。"""
    default = _field_default(spec)
    if spec.kind == "bool":
        return _to_bool(value, bool(default))
    if spec.kind == "enum":
        text = _to_str(value, str(default))
        return text if text in spec.options else default
    if spec.kind == "int":
        return _to_int(value, int(default), int(spec.minimum), int(spec.maximum))
    if spec.kind == "float":
        return round(
            _to_float(value, float(default), spec.minimum, spec.maximum),
            3,
        )
    if spec.kind == "text":
        return str(value if value is not None else default).strip() or default
    return _to_str(value, str(default))


# 旧版静态默认值表：仅在 _conf_schema.json 缺失或无法解析时作为兜底，
# 正常路径下默认值一律由 schema 派生（schema 是唯一真相）。
def _field_default(spec: GroupFieldSpec) -> Any:
    """优先返回 schema 派生默认值，缺失时按类型给安全默认。"""
    if spec.default is not None:
        return spec.default
    # 不能返回 ""：int/float 字段随后需要 int()/float()，空串会直接抛错。
    logger.warning(
        "[ActiveReply] schema 缺少群级字段 %s 的默认值，按 %s 类型回退",
        spec.path,
        spec.kind,
    )
    if spec.kind == "bool":
        return False
    if spec.kind == "int":
        return int(spec.minimum)
    if spec.kind == "float":
        return float(spec.minimum)
    if spec.kind == "enum":
        return spec.options[0] if spec.options else ""
    return ""


def build_global_snapshot(cfg: PluginConfig) -> dict[str, Any]:
    """生成每群配置面板需要的"全局值"快照。"""
    raw = {
        "active_reply": {
            "enable": cfg.active_reply.enable,
            "mode": cfg.active_reply.mode,
            "possibility": cfg.active_reply.possibility,
            "model_history_messages": cfg.active_reply.model_history_messages,
            "model_choice_provider_id": cfg.active_reply.model_choice_provider_id,
            "trigger_threshold": cfg.active_reply.trigger_threshold,
        },
        "group_context_enhancement": {
            "enable": cfg.group_context.enable,
            "max_messages": cfg.group_context.max_messages,
            "include_sender_id": cfg.group_context.include_sender_id,
            "include_role_tag": cfg.group_context.include_role_tag,
            "image_caption": cfg.group_context.image_caption,
            "image_caption_provider_id": cfg.group_context.image_caption_provider_id,
            "image_caption_prompt": cfg.group_context.image_caption_prompt,
        },
        "group_reply_enhancement": {
            "react_pattern": cfg.group_reply.react_pattern,
            "role_display": cfg.group_reply.role_display,
            "smart_reply": cfg.group_reply.smart_reply,
        },
        "memory_rag": {
            "memory_control": {"enable": cfg.memory_rag.enable},
            "memory_generation": {
                "auto_extract_enable": cfg.memory_rag.auto_extract_enable,
                "memory_batch_messages": cfg.memory_rag.batch_messages,
                "auto_extract_interval_minutes": cfg.memory_rag.auto_extract_interval_minutes,
                "auto_extract_scope": cfg.memory_rag.auto_extract_scope,
            },
            "memory_injection": {
                "memory_inject_max_chars": cfg.memory_rag.inject_max_chars,
                "memory_inject_max_items": cfg.memory_rag.inject_max_items,
            },
        },
    }
    return {spec.path: get_by_path(raw, spec.path) for spec in GROUP_OVERRIDE_FIELDS}


def active_reply_allowed(group_id: str, cfg: PluginConfig) -> bool:
    """按黑/白名单判断该群是否允许主动回复。"""
    group_id = str(group_id or "").strip()
    ar = cfg.active_reply
    if ar.list_mode == "whitelist":
        return bool(ar.whitelist) and group_id in ar.whitelist
    return group_id not in ar.blacklist


# --------------------------------------------------------------------------- #
# 页面里的"全局设置"表单：按 _conf_schema.json 做类型转换
# --------------------------------------------------------------------------- #
def coerce_schema_leaf(meta: dict[str, Any], raw: Any) -> Any:
    """把一个表单值按 schema 描述转换成合法配置值。"""
    kind = str(meta.get("type") or "string")
    default = meta.get("default")
    slider = meta.get("slider") if isinstance(meta.get("slider"), dict) else {}

    if kind == "bool":
        return _to_bool(raw, bool(default))
    if kind == "int":
        low = int(slider.get("min", -2_147_483_648))
        high = int(slider.get("max", 2_147_483_647))
        return _to_int(raw, int(default or 0), low, high)
    if kind == "float":
        low = float(slider.get("min", -1e18))
        high = float(slider.get("max", 1e18))
        return round(_to_float(raw, float(default or 0.0), low, high), 4)
    if kind == "list":
        return list(_parse_str_list(raw))
    if kind == "dict":
        return raw if isinstance(raw, dict) else (default or {})

    text = str(raw if raw is not None else "").strip()
    options = meta.get("options")
    if isinstance(options, list) and options:
        if text in options:
            return text
        fallback = default if default in options else options[0]
        return str(fallback)
    if not text and default is not None and not text:
        # 允许用户清空字符串（例如 base_url_override），因此不回落到默认值
        return text
    return text


def _lookup_value(data: Any, path: str) -> tuple[bool, Any]:
    """同时支持"扁平点号键"和"嵌套 dict"两种提交格式。"""
    if isinstance(data, dict) and path in data:
        return True, data[path]
    node: Any = data
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False, None
    return True, node


def build_schema_patch(
    schema: dict[str, Any],
    submitted: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """根据 schema 从表单值中挑出合法字段，返回可直接合并进配置的嵌套 dict。

    页面提交的 ``submitted`` 既可以是 ``{"active_reply.enable": True}`` 这种
    扁平点号键，也可以是 ``{"active_reply": {"enable": True}}`` 这种嵌套结构。
    """
    patch: dict[str, Any] = {}
    for key, meta in schema.items():
        if not isinstance(meta, dict):
            continue
        path = f"{prefix}{key}"
        items = meta.get("items")
        if meta.get("type") == "object" and isinstance(items, dict):
            child = build_schema_patch(items, submitted, f"{path}.")
            if child:
                patch[key] = child
            continue
        found, value = _lookup_value(submitted, path)
        if not found:
            continue
        patch[key] = coerce_schema_leaf(meta, value)
    return patch
