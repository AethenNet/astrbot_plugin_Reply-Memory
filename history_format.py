"""聊天历史行格式与身份解析。

会话历史（session_chats）和记忆历史（memory_chats）共用同一套行格式：

* 用户：``["昵称"/用户ID/时间](角色) #msg消息ID: 正文``
* 机器人：``[SELF/机器人ID/时间](bot): 正文``

昵称总在双引号里并被清洗，用户把名字改成 ``Bot``、``SELF`` 或塞进
``/ [ ] " #msg`` 都造不出第二种行首；解析与渲染只认角色词和 ID，
不再依赖"行首第一个词是什么"。

旧格式（``[昵称/ID/时间]``、``[Bot/ID/时间]``）仍可解析，用于历史数据展示；
但旧格式里昵称叫 ``Bot`` 的人类无法与机器人区分，这是升级前数据的固有限制。
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

SELF_TOKEN = "SELF"
USER_ROLE = "群成员"
SELF_ROLE = "你自己"

_MAX_NAME = 24
_SAFE_NAME_MAP = str.maketrans(
    {
        "/": "／",
        "[": "［",
        "]": "］",
        '"': "＂",
        "\\": "＼",
        "|": "｜",
        ":": "：",
        "#": "＃",
        "\n": " ",
        "\r": " ",
        "\t": " ",
    }
)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_SELF_RE = re.compile(
    r"^\[SELF/([^/\]]*)/([^\]]*)\](?:\(([^)]*)\))?\s*:?\s*"
    r"(?:#msg([^:]+):)?\s*(.*)$",
    re.DOTALL,
)
_USER_NEW_RE = re.compile(
    r'^\["([^"]*)"/([^/]*)/([^\]]*)\](?:\(([^)]*)\))?\s*:?\s*'
    r"(?:#msg([^:]+):)?\s*(.*)$",
    re.DOTALL,
)
_LEGACY_SELF_RE = re.compile(
    r"^\[Bot/([^/\]]*)/([^\]]*)\](?:\(([^)]*)\))?\s*:?\s*"
    r"(?:#msg([^:]+):)?\s*(.*)$",
    re.DOTALL,
)
_LEGACY_USER_RE = re.compile(
    r"^\[([^/\]]+)/([^/\]]*)/([^\]]+)\](?:\(([^)]*)\))?\s*:?\s*"
    r"(?:#msg([^:]+):)?\s*(.*)$",
    re.DOTALL,
)
_RENDERED_SELF_RE = re.compile(
    r"^-\s*\*\*\[(?:你自己|SELF)\s*\|\s*ID:\s*([^\s|]*)\s*\|\s*([^\]]*)\]\*\*:?\s*(.*)$",
    re.DOTALL,
)
_RENDERED_USER_RE = re.compile(
    r'^-\s*\*\*\[群成员\s*\|\s*ID:\s*([^\s|]*)\s*\|\s*昵称:\s*"([^"]*)"\s*'
    r"\|\s*([^\]]*)\]\*\*:?\s*(.*)$",
    re.DOTALL,
)
_HEADER_FALLBACK_RE = re.compile(
    r"^\[[^\]]*\](?:\([^)]*\))?\s*:?\s*(?:#msg[^:]+:\s*)?",
)


def sanitize_name(name: Any, limit: int = _MAX_NAME) -> str:
    """把用户可控昵称收敛成单行、不可破坏行格式的文本。"""
    text = _CONTROL_RE.sub(" ", str(name or ""))
    text = text.translate(_SAFE_NAME_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "未知"
    limit = max(1, int(limit))
    return text[:limit]


_PRIVATE_ORIGIN_MARKS = frozenset({"friend", "friendmessage", "private"})
_INLINE_MAX_LEN = 1000


def sanitize_inline(text: Any, max_len: int = _INLINE_MAX_LEN) -> str:
    """把用户可控正文压成单行，避免伪造历史行或引用注记。

    换行/制表符等控制字符统一变成空格并折叠连续空白；``[``/``]`` 换成
    全角括号，用户无法在正文里拼出 ``[SELF/...]``、``[At: ...]`` 这类
    结构标记；超长时截断并补省略号。调用方负责只对**用户原始文本**调用，
    不要替换 :func:`format_user_line` 自己生成的结构标记。
    """
    value = _CONTROL_RE.sub(" ", str(text or ""))
    value = re.sub(r"\s+", " ", value)
    value = value.replace("[", "［").replace("]", "］").strip()
    limit = max(1, int(max_len))
    if len(value) > limit:
        value = value[: max(1, limit - 1)].rstrip() + "…"
    return value


def is_private_origin(origin: Any) -> bool:
    """判断 ``unified_msg_origin`` 是否私聊会话。

    AstrBot 的来源形如 ``平台:消息类型:会话ID``；私聊的消息类型段可能是
    ``FriendMessage`` / ``friend`` / ``private``。后台扫描与抽取都必须先
    区分私聊，否则用户 ID 会被当成群号写出可跨群召回的作用域（F01）。
    """
    parts = str(origin or "").strip().split(":")
    if len(parts) < 2:
        return False
    return any(part.strip().lower() in _PRIVATE_ORIGIN_MARKS for part in parts[1:])


def _sanitize_id(value: Any) -> str:
    """ID 只保留字母数字与下划线；空值写成 ``-`` 占位。"""
    text = re.sub(r"[^0-9A-Za-z_\-]", "", str(value or "").strip())
    return text or "-"


def _sanitize_token(value: Any, limit: int = 24) -> str:
    """角色/消息号等结构性字段只保留安全字符。"""
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "", str(value or ""))[:limit]


@dataclass(frozen=True)
class HistoryLine:
    """解析后的一条聊天历史。"""

    is_self: bool
    sender_id: str
    name: str
    stamp: str
    body: str
    role_tag: str = ""
    msg_id: str = ""


def format_user_line(
    name: Any,
    sender_id: Any,
    stamp: Any,
    *,
    role_tag: str = "",
    msg_id: str = "",
    body: str = "",
    include_uid: bool = True,
) -> str:
    """生成用户行：``["昵称"/ID/时间](角色) #msgID: 正文``。"""
    safe_name = sanitize_name(name)
    uid = _sanitize_id(sender_id) if include_uid else "-"
    safe_stamp = str(stamp or "").strip().replace("]", "）").replace("[", "（")
    header = f'["{safe_name}"/{uid}/{safe_stamp}]'
    tag = _sanitize_token(role_tag)
    if tag:
        header += f"({tag})"
    safe_msg = _sanitize_token(msg_id)
    if safe_msg:
        header += f" #msg{safe_msg}:"
    if not body:
        return header
    # 有消息号时 header 已经以 ":" 结尾，正文用空格衔接，避免出现 "::"。
    return f"{header} {body}" if safe_msg else f"{header}: {body}"


def format_self_line(
    self_id: Any,
    stamp: Any,
    *,
    body: str = "",
    msg_id: str = "",
    role_tag: str = "bot",
) -> str:
    """生成机器人行：``[SELF/ID/时间](bot): 正文``；昵称不参与格式。"""
    uid = _sanitize_id(self_id)
    safe_stamp = str(stamp or "").strip().replace("]", "）").replace("[", "（")
    header = f"[{SELF_TOKEN}/{uid}/{safe_stamp}]"
    tag = _sanitize_token(role_tag)
    if tag:
        header += f"({tag})"
    safe_msg = _sanitize_token(msg_id)
    if safe_msg:
        header += f" #msg{safe_msg}:"
    if not body:
        return header
    # 有消息号时 header 已经以 ":" 结尾，正文用空格衔接，避免出现 "::"。
    return f"{header} {body}" if safe_msg else f"{header}: {body}"


def parse_line(line: Any) -> HistoryLine | None:
    """解析新格式或旧格式的历史行；无法识别时返回 None。"""
    text = str(line or "")
    match = _SELF_RE.match(text)
    if match:
        uid, stamp, tag, msg_id, body = match.groups()
        return HistoryLine(
            True, _clean_id(uid), "", stamp.strip(), body, tag or "", msg_id or ""
        )
    match = _USER_NEW_RE.match(text)
    if match:
        name, uid, stamp, tag, msg_id, body = match.groups()
        return HistoryLine(
            False, _clean_id(uid), name, stamp.strip(), body, tag or "", msg_id or ""
        )
    match = _LEGACY_SELF_RE.match(text)
    if match:
        uid, stamp, tag, msg_id, body = match.groups()
        return HistoryLine(
            True, _clean_id(uid), "", stamp.strip(), body, tag or "", msg_id or ""
        )
    match = _LEGACY_USER_RE.match(text)
    if match:
        name, uid, stamp, tag, msg_id, body = match.groups()
        return HistoryLine(
            False, _clean_id(uid), name, stamp.strip(), body, tag or "", msg_id or ""
        )
    return None


def parse_rendered(line: Any) -> HistoryLine | None:
    """解析渲染给抽取模型的 ``- **[群成员 | ...]**`` 行。"""
    text = str(line or "")
    match = _RENDERED_SELF_RE.match(text)
    if match:
        uid, stamp, body = match.groups()
        return HistoryLine(True, _clean_id(uid), "", stamp.strip(), body)
    match = _RENDERED_USER_RE.match(text)
    if match:
        uid, name, stamp, body = match.groups()
        return HistoryLine(False, _clean_id(uid), name, stamp.strip(), body)
    return None


def _clean_id(value: Any) -> str:
    """把占位 ID ``-`` 归一成空串。"""
    text = str(value or "").strip()
    return "" if text in {"", "-"} else text


def body_of(line: Any) -> str:
    """取一条历史（原始或渲染后）的正文；识别不了时原样返回。"""
    parsed = parse_line(line) or parse_rendered(line)
    if parsed is not None:
        return parsed.body
    text = str(line or "")
    header = _HEADER_FALLBACK_RE.match(text)
    return text[header.end() :] if header else text


def is_self_line(line: Any) -> bool:
    """判断历史行是否机器人自己发的。"""
    parsed = parse_line(line) or parse_rendered(line)
    return bool(parsed and parsed.is_self)


def message_id_of(line: Any) -> str:
    """取历史行行头的消息 ID。"""
    parsed = parse_line(line)
    return parsed.msg_id if parsed else ""


def render_for_extract(
    line: Any, *, body_transform: Callable[[str], str] | None = None
) -> str:
    """把历史行渲染成抽取模型看到的固定角色词格式。

    ``body_transform`` 用于在解析后改写正文（如把 [Quote ...] / [At: ...]
    转成可读说明），避免调用方重新实现一套行头正则。
    """
    parsed = parse_line(line) or parse_rendered(line)
    if parsed is None:
        return str(line or "")
    body = body_transform(parsed.body) if body_transform else parsed.body
    stamp = parsed.stamp
    if parsed.is_self:
        return f"- **[{SELF_ROLE} | ID: {parsed.sender_id or '0'} | {stamp}]**: {body}"
    name = sanitize_name(parsed.name)
    return (
        f'- **[{USER_ROLE} | ID: {parsed.sender_id or "0"} | 昵称: "{name}" '
        f"| {stamp}]**: {body}"
    )


def _normalized(text: Any) -> str:
    """去掉空白，用于引文匹配。"""
    return re.sub(r"\s+", "", str(text or ""))


def classify_quotes(
    quotes: Iterable[str], lines: Sequence[str]
) -> tuple[list[str], list[str]]:
    """按引文来源把证据拆成 ``(真人引文, 机器人引文)``。"""
    human: list[str] = []
    self_quotes: list[str] = []
    parsed = [(parse_line(line), _normalized(body_of(line))) for line in lines]
    for quote in quotes:
        text = str(quote or "").strip()
        target = _normalized(text)
        if not target:
            continue
        matched = False
        for item, body in parsed:
            if item is not None and target in body:
                if item.is_self:
                    self_quotes.append(text)
                else:
                    human.append(text)
                matched = True
                break
        if not matched:
            human.append(text)
    return human, self_quotes


def build_participant_identities(
    role_ids: Iterable[Any],
    lines: Sequence[str],
    self_id: str = "",
) -> list[dict[str, Any]]:
    """从历史行构建参与者稳定身份（供记忆注入/详情渲染）。

    昵称可能被修改或与他人同名，因此同时保留 sender_id 与写入窗口内出现过
    的所有昵称（aliases）。机器人自身不进入参与者列表；调用方渲染时用
    ``昵称#账号后四位``，避免把完整账号写进模型上下文。

    Args:
        role_ids: 已校验的参与者账号 ID。
        lines: 抽取窗口内的原始历史行。
        self_id: 机器人自身账号，命中时跳过。

    Returns:
        形如 ``[{"identity_key": ..., "sender_id": ..., "display_name": ...,
        "aliases": [...], "is_bot": False}]`` 的列表。
    """
    parsed = [item for item in (parse_line(line) for line in lines) if item is not None]
    names_by_id: dict[str, list[str]] = {}
    for item in parsed:
        if item.is_self or not item.sender_id:
            continue
        name = sanitize_name(item.name)
        if not name or name == "未知":
            continue
        bucket = names_by_id.setdefault(item.sender_id, [])
        if name not in bucket:
            bucket.append(name)

    bot_id = str(self_id or "").strip()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in role_ids or ():
        sender_id = str(raw or "").strip()
        if not sender_id or sender_id == bot_id or sender_id in seen:
            continue
        aliases = names_by_id.get(sender_id, [])
        result.append(
            {
                "identity_key": sender_id,
                "sender_id": sender_id,
                "display_name": aliases[-1] if aliases else sender_id,
                "aliases": list(aliases),
                "is_bot": False,
            }
        )
        seen.add(sender_id)
    return result


def resolve_role_ids(
    model_ids: Iterable[Any],
    lines: Sequence[str],
    quotes: Iterable[str] = (),
    fallback: str = "",
    participants: Iterable[Any] = (),
    evidence_lines: Sequence[str] = (),
    limit: int = 8,
) -> list[str]:
    """合并出参与者角色 ID，永不含机器人 ID（r3.3 扩充）。

    实测多人窗口里模型常只挂一个角色，因此按固定顺序合并去重：
    ① 模型 ``role_ids`` ∩ 窗口内真人；② ``participants`` 里的昵称在窗口历史
    行的 name 中匹配到的人；③ ``evidence_lines``（含引文的原始行）的发送者；
    ④ 窗口内所有真人发送者（``human_ids`` 顺序）。bot/SELF 永不进入结果，
    最多保留 ``limit`` 个，且绝不包含窗口外的人。
    """
    parsed = [item for item in (parse_line(line) for line in lines) if item is not None]
    human_ids = list(
        dict.fromkeys(
            item.sender_id for item in parsed if not item.is_self and item.sender_id
        )
    )
    human_set = set(human_ids)
    cap = max(1, int(limit or 8))
    result: list[str] = []

    def add(value: Any) -> None:
        """把窗口内真人 ID 收进结果（去重、限长、排除 bot/SELF）。"""
        text = str(value or "").strip()
        if text and text in human_set and text not in result and len(result) < cap:
            result.append(text)

    # ① 模型给出的 role_ids（只认窗口内真实出现过的真人）。
    for raw in model_ids or ():
        add(raw)
    # ② 模型给出的参与者昵称 → 窗口历史行里的 sender_id。
    name_to_id = {
        _normalized(item.name): item.sender_id
        for item in parsed
        if not item.is_self and item.sender_id and _normalized(item.name)
    }
    for raw in participants or ():
        sender = name_to_id.get(_normalized(raw))
        if sender:
            add(sender)
    # ③ 证据行（含引文的原始行）的发送者；旧调用只给引文时按正文匹配兜底。
    evidence = list(evidence_lines or ())
    for line in evidence:
        item = parse_line(line)
        if item is not None and not item.is_self:
            add(item.sender_id)
    if not evidence:
        quote_targets = [
            _normalized(quote) for quote in quotes or () if _normalized(quote)
        ]
        for item in parsed:
            if item.is_self or not item.sender_id:
                continue
            body = _normalized(item.body)
            if any(target in body for target in quote_targets):
                add(item.sender_id)
    # ④ 窗口内所有真人发送者（顺序与消息里首次出现一致）。
    for value in human_ids:
        add(value)
    if result:
        return result
    # 兜底：窗口里没有可识别的真人时，沿用调用方提供的发送者。
    value = str(fallback or "").strip()
    return [value] if value else []


def normalize_media(text: str) -> str:
    """把历史里的媒体占位符统一成中文标记。"""
    value = (
        str(text or "")
        .replace("[Image]", "[图片]")
        .replace("[Voice]", "[语音]")
        .replace("[Video]", "[视频]")
        .replace("[Sticker]", "[表情]")
    )
    return re.sub(r"\[File: ([^\]]+)\]", r"[文件: \1]", value)


def select_recent_lines(lines: Sequence[str], max_chars: int) -> list[str]:
    """从最新往前取，直到字数预算用完；用于详情页展示原始窗口。"""
    budget = max(300, int(max_chars))
    selected: list[str] = []
    used = 0
    for line in reversed(list(lines)):
        size = len(str(line)) + 1
        if selected and used + size > budget:
            break
        selected.append(str(line))
        used += size
    selected.reverse()
    return selected


def render_extract_lines(
    lines: Sequence[str],
    max_chars: int,
    *,
    body_transform: Callable[[str], str] | None = None,
) -> list[str]:
    """渲染抽取输入：固定角色词 + 媒体标记 + 字数预算（保留最近消息）。"""
    rendered = [
        normalize_media(render_for_extract(line, body_transform=body_transform))
        for line in lines
    ]
    rendered = [text for text in rendered if text.strip()]
    budget = max(300, int(max_chars))
    selected: list[str] = []
    used = 0
    for text in reversed(rendered):
        if selected and used + len(text) > budget:
            break
        selected.append(text)
        used += len(text)
    selected.reverse()
    return selected


def resolve_self_id(event: Any) -> str:
    """尽量取机器人自身 ID；取不到返回 ``"0"``（绝不退回发送者）。"""
    candidates: list[object] = []
    getter = getattr(event, "get_self_id", None)
    if callable(getter):
        with contextlib.suppress(Exception):
            candidates.append(getter())
    message_obj = getattr(event, "message_obj", None)
    for source in (message_obj, getattr(message_obj, "raw_message", None)):
        if source is None:
            continue
        for key in ("self_id", "bot_id"):
            try:
                candidates.append(getattr(source, key, None))
            except Exception:
                continue
    for value in candidates:
        text = str(value or "").strip()
        if text and text != "0":
            return text
    return "0"


def find_pending_start(history: Sequence[str], last_msg: str) -> int:
    """返回上次总结之后的第一条消息下标；找不到水位时从头开始。"""
    if not last_msg:
        return 0
    for index in range(len(history) - 1, -1, -1):
        if message_id_of(history[index]) == last_msg:
            return index + 1
    return 0


def cooldown_ok(now: float, last_extract_at: float, interval_minutes: float) -> bool:
    """抽取间隔是否已经过去（首次运行为真）。"""
    interval = max(0.0, float(interval_minutes)) * 60
    return now - float(last_extract_at) >= interval


def default_prompt_sample() -> str:
    """提示词试跑用的内置样例（新行格式）。"""
    return "\n".join(
        [
            format_user_line(
                "甲",
                "10001",
                "2026-11-03 19:02:11",
                role_tag="member",
                msg_id="9001",
                body="周六去吃那家火锅？",
            ),
            format_user_line(
                "乙",
                "10002",
                "2026-11-03 19:02:26",
                role_tag="member",
                msg_id="9002",
                body="行，六点半老地方",
            ),
            format_user_line(
                "甲",
                "10001",
                "2026-11-03 19:02:40",
                role_tag="member",
                msg_id="9003",
                body="我最近在戒辣，点微辣",
            ),
            format_self_line(
                "999", "2026-11-03 19:02:52", msg_id="9004", body="记下了"
            ),
        ]
    )


__all__ = [
    "HistoryLine",
    "SELF_ROLE",
    "SELF_TOKEN",
    "USER_ROLE",
    "body_of",
    "classify_quotes",
    "cooldown_ok",
    "default_prompt_sample",
    "find_pending_start",
    "format_self_line",
    "format_user_line",
    "is_private_origin",
    "is_self_line",
    "message_id_of",
    "parse_line",
    "parse_rendered",
    "normalize_media",
    "render_extract_lines",
    "render_for_extract",
    "resolve_role_ids",
    "resolve_self_id",
    "select_recent_lines",
    "sanitize_inline",
    "sanitize_name",
]
