"""记忆工具的单请求预算。

模型可以在一次回复里连续调用多次记忆工具。没有预算时，一句“帮我多查几次”
就能把回复拖成十几秒。这里把调用次数、累计耗时和返回长度限制挂在当前事件
上：同一次用户请求内共享一份预算，请求结束自然失效。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

_MAX_CALLS = 4
_TIME_BUDGET_SECONDS = 20.0
_MAX_RESULT_CHARS = 12000
_EXTRA_KEY = "_areply_recall_budget"


@dataclass
class RecallBudget:
    """一次用户请求内共享的记忆工具预算。"""

    max_calls: int = _MAX_CALLS
    time_budget: float = _TIME_BUDGET_SECONDS
    max_result_chars: int = _MAX_RESULT_CHARS
    calls: int = 0
    elapsed: float = 0.0

    @property
    def remaining_calls(self) -> int:
        """剩余可用调用次数。"""
        return max(0, self.max_calls - self.calls)

    @property
    def remaining_seconds(self) -> float:
        """剩余可用耗时（秒）。"""
        return max(0.0, self.time_budget - self.elapsed)

    def take(self) -> bool:
        """占用一次调用名额；次数或时间耗尽时返回 False。"""
        if self.calls >= self.max_calls or self.remaining_seconds <= 0:
            return False
        self.calls += 1
        return True

    def charge(self, seconds: float) -> None:
        """累计一次调用的实际耗时。"""
        if seconds and seconds > 0:
            self.elapsed += float(seconds)

    def clip(self, text: str) -> str:
        """把返回内容截断到长度上限。"""
        value = str(text or "")
        if len(value) <= self.max_result_chars:
            return value
        return value[: self.max_result_chars] + "…（已截断）"

    def status(self) -> dict[str, Any]:
        """给日志/调试用的预算状态。"""
        return {
            "calls": self.calls,
            "max_calls": self.max_calls,
            "elapsed": round(self.elapsed, 3),
            "remaining_seconds": round(self.remaining_seconds, 3),
        }


def get_budget(event: Any) -> RecallBudget:
    """取出当前事件共享的预算；没有时创建并挂到事件 extra 上。"""
    existing = event.get_extra(_EXTRA_KEY)
    if isinstance(existing, RecallBudget):
        return existing
    budget = RecallBudget()
    event.set_extra(_EXTRA_KEY, budget)
    return budget


class budgeted:
    """``with budgeted(event) as b:`` 的轻量上下文管理器。

    名额不足时把 ``allowed`` 置为 False，调用方自行返回提示；正常结束或抛错
    都会把实际耗时记进预算。
    """

    def __init__(self, event: Any) -> None:
        self.budget = get_budget(event)
        self.allowed = False
        self._started = 0.0

    def __enter__(self) -> budgeted:
        self.allowed = self.budget.take()
        self._started = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.budget.charge(time.monotonic() - self._started)
        return False


__all__ = ["RecallBudget", "budgeted", "get_budget"]
