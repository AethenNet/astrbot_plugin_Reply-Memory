"""短期检索缓存。

同一句话在短时间内的重复检索（连续追问、Agent 一轮内的多次 LLM 调用）
直接复用上次的注入块，省掉 embedding 请求与数据库扫描。缓存只保存渲染好的
卡片文本和命中的记忆编号，不保存用户消息本身。

写库 / 删除 / 巩固等数据变更后必须调用 :meth:`RecallCache.invalidate`：
``generation`` 随之递增，调用方（召回路径）在写回缓存前比较代际，避免把
失效前算出的旧卡片在失效后又写回去（M-05 / N-11）。

``ttl_seconds <= 0`` 表示关闭缓存（``enabled`` 为 False，get/put 均直接返回）。
只在事件循环里使用：``get`` / ``put`` 都是同步方法，没有 await 中断点。
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

DEFAULT_TTL_SECONDS = 45.0
DEFAULT_MAX_ENTRIES = 256


class RecallCache:
    """带 TTL 与 LRU 上限的检索结果缓存。

    Args:
        ttl_seconds: 条目存活秒数；``<=0`` 表示关闭缓存（不再回落默认值）。
        max_entries: 最大条目数；过小或非法值回落到默认值。
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """初始化缓存；参数非法时收敛到安全默认值。"""
        self._items: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        # 代际号：每次主动失效 +1；召回路径写回缓存前用它判断是否已被清空。
        self.generation = 0
        self.configure(ttl_seconds, max_entries)

    def configure(self, ttl_seconds: float, max_entries: int) -> None:
        """更新 TTL/容量（配置重载时调用）；``ttl<=0`` 关闭缓存并清空。"""
        try:
            ttl = float(ttl_seconds)
        except (TypeError, ValueError):
            ttl = DEFAULT_TTL_SECONDS
        self.enabled = ttl > 0
        self.ttl_seconds = max(1.0, ttl) if self.enabled else 0.0
        try:
            entries = int(max_entries)
        except (TypeError, ValueError):
            entries = DEFAULT_MAX_ENTRIES
        self.max_entries = max(8, entries or DEFAULT_MAX_ENTRIES)
        if not self.enabled:
            self._items.clear()

    @staticmethod
    def make_key(*parts: Any) -> str:
        """把检索条件拼成稳定的缓存键。"""
        raw = "\x1f".join(str(part) for part in parts)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        """读取未过期的缓存项；缓存关闭、未命中或已过期返回 None。"""
        if not self.enabled or not key:
            return None
        item = self._items.get(key)
        if item is None:
            return None
        value, created_at = item
        if time.time() - created_at > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: str, value: Any) -> None:
        """写入缓存并按容量淘汰最久未使用的条目；缓存关闭时丢弃。"""
        if not self.enabled or not key:
            return
        self._items[key] = (value, time.time())
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def invalidate(self) -> None:
        """清空缓存并递增代际号（数据变更后调用）。"""
        self._items.clear()
        self.generation += 1


__all__ = ["DEFAULT_MAX_ENTRIES", "DEFAULT_TTL_SECONDS", "RecallCache"]
