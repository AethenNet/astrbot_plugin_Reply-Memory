"""访问计数的批量缓冲。

召回命中后需要更新 ``last_access`` / ``access_count``，但一次请求里可能命中
十几条记忆，逐次写库会把请求路径拖长。本模块只在内存里累积命中的记忆 ID，
由后台任务（``main.py`` 的 3 秒窗口）调用 :meth:`AccessBuffer.flush` 合并成
一次 ``bump_access``；``aclose`` 负责最后一次落库。

设计取舍：不做定时器、不持有任务、不依赖 AstrBot；定时与生命周期由调用方
管理，便于测试与在插件卸载时确定性地收尾。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from astrbot.api import logger


class AccessBuffer:
    """把短时间内的多次访问计数合并成一次数据库写入。

    Args:
        store: 提供同步 ``bump_access(memory_ids)`` 的存储对象（通常是
            :class:`MemoryStore`）。
        max_pending: 缓冲内最多保留的记忆 ID 数；超出后丢弃新的 ID，保证
            长时间不 flush 时内存仍有上限。
    """

    def __init__(self, store: Any, *, max_pending: int = 4096) -> None:
        self._store = store
        self._max_pending = max(1, int(max_pending))
        self._pending: set[int] = set()
        self._closed = False

    def add(self, memory_ids: Sequence[int]) -> None:
        """累积记忆 ID（去重、忽略非正数），不阻塞、不写库。"""
        if self._closed:
            return
        for raw in memory_ids or ():
            try:
                memory_id = int(raw)
            except (TypeError, ValueError):
                continue
            if memory_id <= 0 or memory_id in self._pending:
                continue
            if len(self._pending) >= self._max_pending:
                # 缓冲区满：宁可少记一次访问，也不无限吃内存。
                continue
            self._pending.add(memory_id)

    async def flush(self) -> int:
        """把缓冲的 ID 排序后一次性写库，返回成功写入的条数。

        写入在 ``asyncio.to_thread`` 中执行，不阻塞事件循环。F53：先取快照、
        写库成功后再清空缓冲；写库失败时把这批 ID 并回缓冲（受 ``max_pending``
        约束），避免「先 clear 再写库」失败后整批访问计数永久丢失。失败仍按
        旧契约返回 0，由调用方按需重试（``aclose`` 会再 flush 一次）。
        """
        if not self._pending:
            return 0
        memory_ids = sorted(self._pending)
        # 先摘掉快照；flush 期间 add() 进来的新 ID 不会被这次写入影响。
        self._pending.difference_update(memory_ids)
        try:
            await asyncio.to_thread(self._store.bump_access, memory_ids)
        except Exception as exc:
            # 把这批 ID 并回缓冲重试；仍受 max_pending 上限约束，避免无限吃内存。
            restored = set(memory_ids)
            restored.update(self._pending)
            if len(restored) > self._max_pending:
                restored = set(sorted(restored)[: self._max_pending])
            self._pending = restored
            logger.warning("[ActiveReply] 访问计数批量写入失败: %s", exc)
            return 0
        return len(memory_ids)

    async def aclose(self) -> None:
        """停止接收新 ID 并做最后一次 flush（幂等，可重复调用）。"""
        self._closed = True
        await self.flush()


__all__ = ["AccessBuffer"]
