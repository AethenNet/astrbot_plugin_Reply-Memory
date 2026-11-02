"""插件配置模型。

把 AstrBot 传入的原始 dict 配置统一解析成不可变 dataclass，
并在这里集中处理默认值、类型兼容与范围钳制，避免业务代码到处 .get()。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    return _to_float(value, default, 0.0, 1.0)


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
    """用户屏蔽（作用于全局，群级记录由 block_store 管理）。"""

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


DEFAULT_GROUP_SUMMARY_PROMPT = """## 总结要求
1. **第一人称视角**：以"我"的视角描述对话，不要使用"bot"、"助手"等第三人称词汇。
2. **讨论内容**：群里讨论了什么。
3. **参与者**：谁参与了对话（包括你自己），**必须使用具体昵称**，明确区分**谁说的**和**说了什么**，不要把别人的发言混为自己的。判断发言者时只看每条消息开头 **[昵称 | ID: 账号 | 时间]** 里的昵称；正文中出现的昵称（引用、@、转述）都不是发言者。正文末尾的（引用 X 的旧消息：「…」）只是当前发言人引用的旧内容，（@ X）只表示提及，都不是当前发言者说的话。查找所有以 "[Bot:" 开头的消息，这些消息是**你自己说的话**。
4. **总结结果**：把整段消息进行总结，每条记忆 1~5 句话、不超过 200 字，**相对时间（今天/昨天/下周）**按消息时间换算成具体日期。

## 时间规则
- 每条消息行头的时间是真实发送时间；总结里的日期和时间必须据此换算。
- 示例中的 2026-10-28 只是格式示意，**不要照抄示例日期**；相对时间必须换算成具体日期。

## 置信度评分标准（0.0-1.0）
"置信度"表示这段对话作为长期记忆的价值，不是事实可信度，也不是对某一条消息打分。
**先判断档位，再在档位区间内取值；默认从低档起评，只有同时满足全部条件才进入高档。**
- 0.90-1.00：重要群体决策。只有群规制定、全员投票结果、正式承诺、涉及金钱/安全/重大日程的决定，或对群体有长期约束的计划，才可以进入本档；并且至少 3 名成员实质参与、有具体可执行的安排。普通约饭、约玩、临时改时间最高 0.60。
- 0.70-0.80：有价值的讨论。多人参与且有明确结论，但影响限于短期或个别成员。
- 0.50-0.60：日常讨论。普通约饭、约玩、临时安排、一般信息交流；即使形成了约定，只要是一次性、临时性的日常安排，就落在本档，**最高 0.60**。有明确时间、地点或参与者的安排取 0.60；其他日常交流取 0.50-0.55。
- 0.30-0.40：简单闲聊。少数人参与、临时状态、当下情绪、玩笑、反讽、推测、传闻。
- 0.00-0.20：没有信息量的灌水。单人刷屏、一次性问答、寒暄、表情、复读。

### 防虚高约束
- 大多数群聊片段应落在 0.30-0.60；0.70 以上只给少数真正有结论、有长期价值的讨论。
- 不要因为"总结写得完整""消息条数多""有人 @ 或引用"而提高评分。
- 无法确认多人达成共识，或只是临时起意的吃饭、看电影、打游戏、生日祝福、日常吐槽时，一律不得高于 0.60。
- 反例：三人临时约好今晚七点吃火锅 → 0.60（daily），不是 0.90。

## 字段规则
- confidence 必须落在上面某个档位区间内，保留两位小数；
- confidence_tier 必须从 major / discussion / daily / chatter / spam 中选择，并与 confidence 所在档位一致；
- importance 是同一口径的长期价值评分，通常等于 confidence；
- topics 最多 5 个，participants 最多 10 个；
- participants 和 role_ids 只填片段中出现过的用户；不确定就留空；
- scope：个人偏好/身份/习惯用 global；只和这个群有关的事用 group；
- quotes 至少 1 条，必须来自当前消息正文并能逐字找到；不要摘录“引用旧消息”里的内容，否则整条记忆会被丢弃。

## 输出格式
只输出一个 JSON 对象，不要数组、解释或代码块：
{
  "summary": "整段对话的总结",
  "topics": ["主题1", "主题2"],
  "participants": ["昵称1", "昵称2"],
  "role_ids": ["相关用户QQ号"],
  "scope": "global | group",
  "confidence_tier": "major | discussion | daily | chatter | spam",
  "confidence": 0.0,
  "importance": 0.0,
  "quotes": ["逐字来自聊天片段的证据1", "证据2"]
}

## 消息格式
- **[昵称 | ID: 账号 | 时间]**：群成员发送的消息，只有方括号开头的昵称是发送者；
- **[Bot: 昵称 | ID: 账号 | 时间]**：你自己（机器人）发送的消息，以 "Bot:" 开头；
- **[图片]**、**[文件: 文件名]**、**[语音]**、**[视频]**、**[表情]**：媒体占位；
- 正文末尾的（引用 X 的旧消息：「…」）不是当前发言人说的话；（@ X）只表示提及。

## 示例输入
- **[事佳 | ID: 1000123 | 2026-10-28 15:01:10]**: 今天谁在摸鱼？
- **[苏琪 | ID: 1000456 | 2026-10-28 15:01:23]**: 刚开完会，脑子还在加载
- **[Bot: AstrBot | ID: 123456789 | 2026-10-28 15:01:26]**: +1
- **[事佳 | ID: 1000123 | 2026-10-28 15:01:52]**: 那就老地方吃火锅，七点？
- **[苏琪 | ID: 1000456 | 2026-10-28 15:02:02]**: 七点可以，我六点半下班
- **[Dew | ID: 1000789 | 2026-10-28 15:02:12]**: 最后一个来的，请大家喝奶茶

## 示例输出
{"summary": "群里约定 2026-10-28 晚上七点在老地方吃火锅；苏琪六点半下班，Dew 说最后一个到场的人请大家喝奶茶。", "topics": ["晚餐约定", "火锅", "下班时间"], "participants": ["事佳", "苏琪", "Dew"], "role_ids": ["1000123", "1000456", "1000789"], "scope": "group", "confidence_tier": "daily", "confidence": 0.60, "importance": 0.60, "quotes": ["那就老地方吃火锅，七点？", "七点可以，我六点半下班", "最后一个来的，请大家喝奶茶"]}

聊天片段：
{messages}"""

DEFAULT_PRIVATE_SUMMARY_PROMPT = """## 总结要求
1. **第一人称视角**：以"我"的视角描述私聊对话，不要使用"bot"、"助手"等第三人称词汇。
2. **聊天内容**：你们聊了什么。
3. **参与者**：只涉及你和对方，**必须使用具体昵称**，明确区分**谁说的**和**说了什么**。判断发言者时只看每条消息开头 **[昵称 | ID: 账号 | 时间]** 里的昵称；正文中出现的昵称（引用、@、转述）都不是发言者。正文末尾的（引用 X 的旧消息：「…」）只是当前发言人引用的旧内容，（@ X）只表示提及，都不是当前发言者说的话。查找所有以 "[Bot:" 开头的消息，这些消息是**你自己说的话**。
4. **总结结果**：把整段消息进行总结，每条记忆 1~5 句话、不超过 200 字，**相对时间（今天/昨天/下周）**按消息时间换算成具体日期。

## 时间规则
- 每条消息行头的时间是真实发送时间；总结里的日期和时间必须据此换算。
- 示例中的 2026-10-28 只是格式示意，**不要照抄示例日期**；相对时间必须换算成具体日期。

## 置信度评分标准（0.0-1.0）
"置信度"表示这段对话作为长期记忆的价值，不是事实可信度，也不是对某一条消息打分。
**先判断档位，再在档位区间内取值；默认从低档起评。**
- 0.90-1.00：正式约定、明确的长期计划或关键个人信息，并且已经确认；普通约饭、约玩、临时看电影最高 0.60。
- 0.70-0.80：有价值的交流，有明确结论，但影响限于短期。
- 0.50-0.60：日常聊天、一般信息、临时安排；普通约饭、约玩、一次性计划最高 0.60。有明确时间、地点的安排取 0.60；其他日常聊天取 0.50-0.55。
- 0.30-0.40：闲聊、临时状态、当下情绪、玩笑、反讽、推测。
- 0.00-0.20：寒暄、表情、复读、没有信息量。

### 防虚高约束
- 大多数私聊片段应落在 0.30-0.60；0.70 以上只给真正有长期价值的交流。
- 不要因为"总结写得完整""消息条数多"而提高评分。
- 临时起意的吃饭、看电影、打游戏、普通祝福、日常吐槽不得高于 0.60。
- 反例：对方临时约我今晚看电影 → 0.60（daily），不是 0.90。

## 字段规则
- confidence 必须落在上面某个档位区间内，保留两位小数；
- confidence_tier 必须从 major / discussion / daily / chatter / spam 中选择，并与 confidence 所在档位一致；
- importance 是同一口径的长期价值评分，通常等于 confidence；
- topics 最多 5 个，participants 最多 10 个；
- participants 和 role_ids 只填片段中出现过的用户；不确定就留空；
- scope：个人偏好/身份/习惯用 global；只和这次私聊有关的事用 group；
- quotes 至少 1 条，必须来自当前消息正文并能逐字找到；不要摘录“引用旧消息”里的内容，否则整条记忆会被丢弃。

## 输出格式
只输出一个 JSON 对象，不要数组、解释或代码块：
{
  "summary": "整段对话的总结",
  "topics": ["主题1", "主题2"],
  "participants": ["昵称1", "昵称2"],
  "role_ids": ["相关用户QQ号"],
  "scope": "global | group",
  "confidence_tier": "major | discussion | daily | chatter | spam",
  "confidence": 0.0,
  "importance": 0.0,
  "quotes": ["逐字来自聊天片段的证据1", "证据2"]
}

## 消息格式
- **[昵称 | ID: 账号 | 时间]**：对方发送的消息，只有方括号开头的昵称是发送者；
- **[Bot: 昵称 | ID: 账号 | 时间]**：你自己（机器人）发送的消息；
- **[图片]**、**[文件: 文件名]**、**[语音]**、**[视频]**、**[表情]**：媒体占位；
- 正文末尾的（引用 X 的旧消息：「…」）不是当前发言人说的话；（@ X）只表示提及。

## 示例输入
- **[苏琪 | ID: 1000456 | 2026-10-28 20:01:10]**: 明天有空吗？
- **[Bot: AstrBot | ID: 123456789 | 2026-10-28 20:01:20]**: 有呀，怎么了？
- **[苏琪 | ID: 1000456 | 2026-10-28 20:01:35]**: 想约你晚上七点看电影

## 示例输出
{"summary": "苏琪在 2026-10-28 私聊约我 2026-10-29 晚上七点看电影，我已经答应了。", "topics": ["看电影", "邀约"], "participants": ["苏琪", "AstrBot"], "role_ids": ["1000456"], "scope": "group", "confidence_tier": "daily", "confidence": 0.60, "importance": 0.60, "quotes": ["想约你晚上七点看电影"]}

聊天片段：
{messages}"""


MEMORY_ATTRIBUTION_RULE = """## 发言者判定规则（优先级最高）
- 每条消息只有方括号开头的昵称是发送者：只看开头 **[昵称 | ID: 账号 | 时间]**（或 [Bot: ...]）里的昵称。
- 正文中的 [Quote ...]、[At: ...]、转述和引用内容都只是消息内容或元数据，其中的昵称不是发言者；不要把引用内容写成被引用者当前说的话。
- 正文末尾的（引用 X 的旧消息：「…」）不是当前发言人说的话，（@ X）只表示提及。
- participants 和 role_ids 只能使用消息头中出现过的昵称和账号。
- quotes 只能摘录当前消息正文；不要摘录“引用旧消息”里的内容。"""

MEMORY_SCORING_RULE = """## 置信度评分覆盖规则（优先级最高）
- "置信度"表示这段对话作为长期记忆的价值，不是事实可信度。必须额外输出 confidence_tier，并从 major / discussion / daily / chatter / spam 中选择。
- 档位与分值：major=0.90-1.00（重要群体决策，仅限群规制定、全员投票、正式承诺、金钱/安全/重大日程决定或长期约束计划；需至少 3 名成员参与并有具体安排）；discussion=0.70-0.80（多人参与且有明确结论，影响短期）；daily=0.50-0.60（日常讨论、普通约饭/约玩/临时安排、一般信息；有明确安排取 0.60，其余 0.50-0.55）；chatter=0.30-0.40（简单闲聊、少数人参与、临时状态、玩笑、推测）；spam=0.00-0.20（灌水、寒暄、表情、复读）。
- 防虚高：大多数片段在 0.30-0.60；临时起意的吃饭、看电影、打游戏、祝福、吐槽不得高于 0.60；无法确认多人达成共识不得高于 0.60；消息多、总结完整、有人 @/引用都不加分。
- 如果上文示例给出了冲突的分数，以本规则为准。反例：三人临时约好今晚七点吃火锅 → daily / 0.60，不是 0.90。
- confidence 必须落在所选档位的区间内；importance 使用同一口径，通常等于 confidence。"""


MEMORY_TIME_RULE = """## 时间规则（优先级最高）
- 每条消息行头的时间都是真实发送时间；总结中的日期/时间必须据此换算。
- 示例中的 2026-10-28 只是格式示意，**不要照抄示例日期**；相对时间必须换算成具体日期。
- 不要凭印象编造日期；不确定时以消息行头为准。"""


def ensure_memory_prompt_rules(prompt: str) -> str:
    """给旧版或自定义记忆提示词补齐发言者归属与置信度评分规则。

    只追加缺失的规则，不覆盖用户自定义内容；插入位置在 {messages}
    占位符之前，确保规则出现在聊天片段前面。旧配置里保存的默认提示词
    含高分示例，因此这里会明确声明覆盖规则优先。
    """
    text = str(prompt or "").strip()
    if not text:
        return text
    needs_scoring = "confidence_tier" not in text
    if needs_scoring and '## 置信度评分标准 "importance"（0.0-1.0）' in text:
        # 旧默认示例里的 0.90 / 0.80 是评分虚高的直接锚点，先替换成日常档示例。
        text = text.replace(
            '"confidence": 0.90, "importance": 0.80',
            '"confidence_tier": "daily", "confidence": 0.60, "importance": 0.60',
        )
    additions: list[str] = []
    if needs_scoring:
        additions.append(MEMORY_SCORING_RULE)
    if "只有方括号开头的昵称是发送者" not in text:
        additions.append(MEMORY_ATTRIBUTION_RULE)
    if "不要照抄示例日期" not in text:
        additions.append(MEMORY_TIME_RULE)
    if not additions:
        return text
    block = "\n\n".join(additions)
    marker = "{messages}"
    if marker not in text:
        return f"{text}\n\n{block}"
    head, tail = text.rsplit(marker, 1)
    head = head.rstrip()
    for heading in ("聊天片段：", "聊天片段:"):
        if head.endswith(heading):
            head = head[: -len(heading)].rstrip()
            break
    return f"{head}\n\n{block}\n\n聊天片段：\n{marker}{tail}"


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
    extract_max_chars: int = 1500
    window_gap_minutes: int = 60
    group_prompt: str = DEFAULT_GROUP_SUMMARY_PROMPT
    private_prompt: str = DEFAULT_PRIVATE_SUMMARY_PROMPT
    auto_extract_min_confidence: float = 0.5
    auto_extract_scope: str = "group_first"
    dedup_threshold: float = 0.90
    conflict_threshold: float = 0.75
    # 读取与生命周期
    decay_half_life_days: int = 10
    inject_max_chars: int = 500
    inject_max_items: int = 5
    consolidation_enable: bool = True
    consolidation_interval_days: int = 7
    archive_enable: bool = True
    archive_days: int = 15
    archive_importance: float = 0.35
    # 旧字段保留，兼容早期配置读取
    auto_extract_every_n: int = 10
    auto_extract_window: int = 20


@dataclass(frozen=True)
class RuntimeConfig:
    """运行时内存容量及外部模型调用超时配置。"""

    max_origins: int = 500
    image_caption_timeout: float = 45.0
    model_choice_timeout: float = 45.0


@dataclass(frozen=True)
class PluginConfig:
    """经过规范化的插件配置快照。

    各字段对应一个不可变配置区块，避免业务逻辑直接操作原始 JSON 字典。
    ``default_factory`` 为每个配置对象分别创建缺省子配置，避免实例之间共享状态。
    ``history_enabled``、``react_enabled`` 等属性是底层开关组合出的便捷视图，
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
    def history_enabled(self) -> bool:
        """返回群聊历史记录与注入功能的总开关状态。"""
        # 兼容调用方使用 history_enabled 命名访问上下文增强总开关。
        return self.group_context.enable

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
        max_messages=_to_int(ctx_raw.get("max_messages"), 200, 50, 1000),
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
        if str(item).strip().isdigit() and len(str(item).strip()) <= 20
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
    scope_raw = _to_str(mem_raw.get("auto_extract_scope")) or "group_first"
    if scope_raw not in {"group_first", "global", "group"}:
        scope_raw = "group_first"
    control_raw = _section(mem_raw, "memory_control") or mem_raw
    gen_raw = _section(mem_raw, "memory_generation") or mem_raw
    inj_raw = _section(mem_raw, "memory_injection") or mem_raw
    eval_raw = _section(mem_raw, "memory_evaluation") or mem_raw
    scope_value = str(
        gen_raw.get("auto_extract_scope")
        or mem_raw.get("auto_extract_scope")
        or "group_first"
    ).strip()
    if scope_value not in {"group_first", "global", "group"}:
        scope_value = "group_first"
    memory_rag = MemoryRagConfig(
        enable=_to_bool(control_raw.get("enable", mem_raw.get("enable")), False),
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
        recall_k=_to_int(inj_raw.get("recall_k", mem_raw.get("recall_k")), 20, 1, 50),
        max_return_results=_to_int(
            inj_raw.get("max_return_results", mem_raw.get("max_return_results")),
            100,
            1,
            300,
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
        dedup_threshold=_to_float(
            gen_raw.get(
                "memory_dedup_threshold", mem_raw.get("memory_dedup_threshold")
            ),
            0.90,
            0.80,
            0.99,
        ),
        conflict_threshold=_to_float(
            gen_raw.get(
                "memory_conflict_threshold", mem_raw.get("memory_conflict_threshold")
            ),
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
        decay_half_life_days=_to_int(
            eval_raw.get(
                "memory_decay_half_life_days",
                mem_raw.get("memory_decay_half_life_days"),
            ),
            10,
            7,
            30,
        ),
        inject_max_chars=_to_int(
            inj_raw.get(
                "memory_inject_max_chars", mem_raw.get("memory_inject_max_chars")
            ),
            500,
            100,
            1200,
        ),
        inject_max_items=_to_int(
            inj_raw.get(
                "memory_inject_max_items", mem_raw.get("memory_inject_max_items")
            ),
            5,
            0,
            15,
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
        archive_enable=_to_bool(
            eval_raw.get("memory_archive_enable", mem_raw.get("memory_archive_enable")),
            True,
        ),
        archive_days=_to_int(
            eval_raw.get("memory_archive_days", mem_raw.get("memory_archive_days")),
            15,
            7,
            120,
        ),
        archive_importance=_to_float(
            eval_raw.get(
                "memory_archive_importance", mem_raw.get("memory_archive_importance")
            ),
            0.35,
            0.0,
            1.0,
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


# 说明：只允许覆盖与"群内行为"相关的字段（含记忆 RAG）；搜索/超时属于全局服务配置。
GROUP_OVERRIDE_FIELDS: tuple[GroupFieldSpec, ...] = (
    GroupFieldSpec("active_reply.enable", "启用主动回复", "bool"),
    GroupFieldSpec(
        "active_reply.mode",
        "回复方式",
        "enum",
        options=("probability", "model_choice"),
        labels=("概率", "模型判定"),
    ),
    GroupFieldSpec(
        "active_reply.possibility", "回复概率", "float", minimum=0, maximum=1, step=0.1
    ),
    GroupFieldSpec(
        "active_reply.model_history_messages",
        "模型判定消息附带",
        "int",
        minimum=0,
        maximum=100,
    ),
    GroupFieldSpec("active_reply.model_choice_provider_id", "模型判定提供商", "str"),
    GroupFieldSpec(
        "active_reply.trigger_threshold",
        "回复触发积累",
        "int",
        minimum=2,
        maximum=50,
    ),
    GroupFieldSpec("group_context_enhancement.enable", "启用群聊上下文增强", "bool"),
    GroupFieldSpec(
        "group_context_enhancement.max_messages",
        "会话消息上限",
        "int",
        minimum=50,
        maximum=1000,
        step=10,
    ),
    GroupFieldSpec(
        "group_context_enhancement.include_sender_id", "包含用户信息", "bool"
    ),
    GroupFieldSpec("group_context_enhancement.include_role_tag", "包含标签", "bool"),
    GroupFieldSpec("group_context_enhancement.image_caption", "图片转述", "bool"),
    GroupFieldSpec(
        "group_context_enhancement.image_caption_provider_id", "图片转述提供商", "str"
    ),
    GroupFieldSpec(
        "group_context_enhancement.image_caption_prompt", "图片转述文本", "text"
    ),
    GroupFieldSpec("group_reply_enhancement.react_pattern", "React Pattern", "bool"),
    GroupFieldSpec("group_reply_enhancement.role_display", "用户角色显示", "bool"),
    GroupFieldSpec("group_reply_enhancement.smart_reply", "智能回复", "bool"),
    GroupFieldSpec("memory_rag.memory_control.enable", "启用记忆 RAG", "bool"),
    GroupFieldSpec(
        "memory_rag.memory_generation.auto_extract_enable", "自动产生记忆", "bool"
    ),
    GroupFieldSpec(
        "memory_rag.memory_generation.memory_batch_messages",
        "累计消息记忆",
        "int",
        minimum=10,
        maximum=100,
        step=1,
    ),
    GroupFieldSpec(
        "memory_rag.memory_generation.auto_extract_interval_minutes",
        "抽取间隔（分钟）",
        "int",
        minimum=1,
        maximum=120,
        step=1,
    ),
    GroupFieldSpec(
        "memory_rag.memory_generation.auto_extract_scope",
        "记忆作用域",
        "enum",
        options=("group_first", "global", "group"),
        labels=("本群优先+全局兜底", "全部全局", "仅本群"),
    ),
    GroupFieldSpec(
        "memory_rag.memory_injection.memory_inject_max_chars",
        "注入字数上限",
        "int",
        minimum=100,
        maximum=1200,
        step=50,
    ),
    GroupFieldSpec(
        "memory_rag.memory_injection.memory_inject_max_items",
        "注入条数上限",
        "int",
        minimum=0,
        maximum=15,
        step=1,
    ),
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


def _field_default(spec: GroupFieldSpec) -> Any:
    """根据群级字段定义返回该项的插件默认值。"""
    # 群级覆盖表单使用独立默认值，避免依赖未加载的全局配置实例。
    defaults: dict[str, Any] = {
        "active_reply.enable": False,
        "active_reply.mode": "probability",
        "active_reply.possibility": 0.1,
        "active_reply.model_history_messages": 0,
        "active_reply.model_choice_provider_id": "",
        "active_reply.trigger_threshold": 10,
        "group_context_enhancement.enable": True,
        "group_context_enhancement.max_messages": 200,
        "group_context_enhancement.include_sender_id": True,
        "group_context_enhancement.include_role_tag": True,
        "group_context_enhancement.image_caption": False,
        "group_context_enhancement.image_caption_provider_id": "",
        "group_context_enhancement.image_caption_prompt": DEFAULT_IMAGE_CAPTION_PROMPT,
        "group_reply_enhancement.react_pattern": True,
        "group_reply_enhancement.role_display": True,
        "group_reply_enhancement.smart_reply": True,
        "memory_rag.memory_control.enable": False,
        "memory_rag.memory_generation.auto_extract_enable": True,
        "memory_rag.memory_generation.memory_batch_messages": 50,
        "memory_rag.memory_generation.auto_extract_interval_minutes": 10,
        "memory_rag.memory_generation.auto_extract_scope": "group_first",
        "memory_rag.memory_injection.memory_inject_max_chars": 500,
        "memory_rag.memory_injection.memory_inject_max_items": 5,
    }
    return defaults.get(spec.path, "")


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
