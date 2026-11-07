"""记忆抽取的触发与窗口规划（纯函数，便于单测）。

三条触发规则：

1. 批量：未总结消息达到 ``batch`` 且冷却结束；
2. 冷却期满：冷却结束时积压仍超过 ``batch``，由后台扫把积压总结掉；
3. 空闲兜底：会话空闲超过 ``gap_minutes`` 且至少 3 条未总结消息。

窗口一律从最早的未总结消息开始取，避免积压时丢最旧的消息。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

MIN_MESSAGES = 3


def plan_batch(pending: int, batch: int, cooldown_ok: bool) -> bool:
    """是否满足「攒满 batch 且冷却结束」的批量触发。"""
    return bool(cooldown_ok and pending >= max(1, int(batch)))


def plan_cooldown_flush(
    *,
    now: float,
    last_extract_at: float,
    interval_minutes: float,
    pending: int,
    batch: int,
) -> bool:
    """冷却期满时是否应把积压（>= batch）总结掉。"""
    if pending < max(1, int(batch)):
        return False
    return now - float(last_extract_at) >= max(0.0, float(interval_minutes)) * 60


def plan_idle(
    *,
    now: float,
    last_chat: float,
    gap_minutes: float,
    pending: int,
    last_flush_at: float,
) -> bool:
    """是否满足空闲兜底触发。

    采用 ``last_flush_at < last_chat`` 做去重：同一次空闲只触发一次，
    新消息到达（``last_chat`` 变大）后才会重新计时。
    """
    gap_minutes = float(gap_minutes or 0)
    if gap_minutes <= 0 or pending < MIN_MESSAGES or not last_chat:
        return False
    if last_flush_at >= last_chat:
        return False
    return now - float(last_chat) >= gap_minutes * 60


def select_chunk(
    lines: Sequence[str],
    budget: int,
    measure: Callable[[str], int],
) -> tuple[list[str], list[str]]:
    """从最早开始取一段不超过字数预算的消息，返回 ``(chunk, rest)``。

    至少返回 1 条（单条超预算也保留），避免某条长消息把抽取卡死。
    """
    limit = max(300, int(budget))
    used = 0
    index = 0
    total = len(lines)
    while index < total:
        size = max(1, int(measure(lines[index])))
        if index > 0 and used + size > limit:
            break
        used += size
        index += 1
    chunk = list(lines[:index])
    return chunk, list(lines[index:])


def plan_sweep(
    *,
    now: float,
    last_chat: float,
    last_flush_at: float,
    gap_minutes: float,
    pending: int,
    batch: int,
    last_extract_at: float,
    interval_minutes: float,
) -> str:
    """返回后台扫描应采用的触发原因：``idle`` / ``cooldown`` / 空串。"""
    if plan_idle(
        now=now,
        last_chat=last_chat,
        gap_minutes=gap_minutes,
        pending=pending,
        last_flush_at=last_flush_at,
    ):
        return "idle"
    if plan_cooldown_flush(
        now=now,
        last_extract_at=last_extract_at,
        interval_minutes=interval_minutes,
        pending=pending,
        batch=batch,
    ):
        return "cooldown"
    return ""


__all__ = [
    "MIN_MESSAGES",
    "plan_batch",
    "plan_cooldown_flush",
    "plan_idle",
    "plan_sweep",
    "select_chunk",
]
