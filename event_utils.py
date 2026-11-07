"""事件类型辅助：识别 AstrBot 的合成唤醒轮次。

定时任务、后台任务与插件唤醒不是用户对话轮次。新版 AstrBot 用
``agent_wakeup`` 事件承载这类唤醒（平台为目标会话平台），4.29.4 及更早
版本的定时任务事件平台为 ``cron``。把这些轮次写入聊天历史/记忆会污染
抽取窗口与总结游标，因此统一在这里判断。
"""

from __future__ import annotations

from typing import Any


def is_synthetic_event(event: Any) -> bool:
    """判断事件是否为框架合成的唤醒轮次。

    Args:
        event: AstrBot 消息事件；允许没有 ``event_kind`` 属性的旧版本。

    Returns:
        是合成唤醒返回 True；普通用户消息返回 False。
    """
    try:
        raw_kind = getattr(event, "event_kind", "")
        if hasattr(raw_kind, "value"):
            raw_kind = raw_kind.value
        kind = str(raw_kind or "").strip().lower()
        if kind == "agent_wakeup":
            return True
        getter = getattr(event, "get_platform_name", None)
        platform = str(getter() if callable(getter) else "").strip().lower()
        return platform == "cron"
    except Exception:
        # 事件对象异常时按普通消息处理，避免误吞真实用户消息。
        return False
