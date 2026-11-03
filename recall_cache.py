"""短期检索缓存。

同一句话在短时间内的重复检索（连续追问、Agent 一轮内的多次 LLM 调用）
直接复用上次的注入块，省掉 embedding 请求与数据库扫描。缓存只保存渲染好的
卡片文本和命中的记忆编号，不保存用户消息本身。

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
    """带 TTL 与 LRU 上限的检索结果缓存。"""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """初始化缓存；参数非法时收敛到安全默认值。"""
        self.ttl_seconds = max(1.0, float(ttl_seconds or DEFAULT_TTL_SECONDS))
        self.max_entries = max(8, int(max_entries or DEFAULT_MAX_ENTRIES))
        self._items: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(*parts: Any) -> str:
        """把检索条件拼成稳定的缓存键。"""
        raw = "\x1f".join(str(part) for part in parts)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        """读取未过期的缓存项；未命中或已过期返回 None。"""
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        value, created_at = item
        if time.time() - created_at > self.ttl_seconds:
            self._items.pop(key, None)
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        """写入缓存并按容量淘汰最久未使用的条目。"""
        if not key:
            return
        self._items[key] = (value, time.time())
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        """清空缓存（配置切换或数据目录变化时调用）。"""
        self._items.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        """当前缓存条目数，便于状态页展示。"""
        return len(self._items)


__all__ = ["DEFAULT_MAX_ENTRIES", "DEFAULT_TTL_SECONDS", "RecallCache"]
