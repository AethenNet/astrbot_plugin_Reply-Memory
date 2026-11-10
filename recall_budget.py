"""记忆工具的单请求预算。

模型可以在一次回复里连续调用多次记忆工具。没有预算时，一句“帮我多查几次”
就能把回复拖成十几秒。这里把调用次数、累计耗时和返回长度限制挂在当前事件
上：同一次用户请求内共享一份预算，请求结束自然失效。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

_MAX_CALLS = 4
_TIME_BUDGET_SECONDS = 20.0
_MAX_RESULT_CHARS = 12000
_EXTRA_KEY = "_areply_recall_budget"


def dump_trimmed_json(
    payload: dict[str, Any], max_chars: int, list_key: str = "records"
) -> str:
    """按记录裁剪超长 JSON，始终返回合法对象，不直接截断字符串。"""
    original_count = 0
    original_length = 0
    if not isinstance(payload, dict):
        payload = {}
    else:
        # 纯函数不修改调用方对象；列表裁剪也只作用于本地副本。
        payload = dict(payload)
        if isinstance(payload.get(list_key), list):
            payload[list_key] = list(payload[list_key])
    try:
        original_count = int(payload.get("count") or 0)
    except (TypeError, ValueError, OverflowError):
        pass
    records = payload.get(list_key)
    if isinstance(records, list):
        original_length = len(records)
    try:
        limit = max(1, int(max_chars))
    except (TypeError, ValueError, OverflowError):
        limit = _MAX_RESULT_CHARS
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError):
        original_length = max(original_length, original_count)
        return json.dumps(
            {
                "status": "ok",
                "count": original_count,
                "truncated": True,
                "omitted": original_length,
                "note": "结果过长，已省略正文；可用 read_memory 按 id 读取单条。",
            },
            ensure_ascii=False,
        )
    if len(text) <= limit:
        return text
    records = payload.get(list_key)
    if isinstance(records, list):
        while records:
            records.pop()
            payload["truncated"] = True
            payload["omitted"] = original_length - len(records)
            try:
                text = json.dumps(payload, ensure_ascii=False)
            except (TypeError, ValueError, OverflowError):
                break
            if len(text) <= limit:
                return text
    return json.dumps(
        {
            "status": "ok",
            "count": original_count,
            "truncated": True,
            "omitted": original_length,
            "note": "结果过长，已省略正文；可用 read_memory 按 id 读取单条。",
        },
        ensure_ascii=False,
    )


@dataclass
class RecallBudget:
    """一次用户请求内共享的记忆工具预算。"""

    max_calls: int = _MAX_CALLS
    time_budget: float = _TIME_BUDGET_SECONDS
    max_result_chars: int = _MAX_RESULT_CHARS
    calls: int = 0
    elapsed: float = 0.0

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


__all__ = ["RecallBudget", "budgeted", "dump_trimmed_json", "get_budget"]
