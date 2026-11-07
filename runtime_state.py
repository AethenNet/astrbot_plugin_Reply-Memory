"""进程内运行时状态。

全部状态按 ``unified_msg_origin`` 分桶，并用一个 LRU 队列限制来源数量，
避免长期运行时内存无界增长。所有状态都是内存态，重启即清空。
"""

from __future__ import annotations

import time
from collections import OrderedDict, defaultdict

# F48：调试开关集合的容量上限。它与 origin_lru 独立，因为用户手动开启的
# 调试状态不应因其它会话活跃而被静默关闭；超过上限时淘汰最久未开启的来源。
_DEBUG_ORIGINS_LIMIT = 200


def _new_turn_usage() -> dict[str, int]:
    """本轮 token/召回指标的空白桶。"""
    return {
        "input_cached": 0,
        "input_other": 0,
        "output": 0,
        "calls": 0,
        "memory_hits": 0,
        "lexical_hits": 0,
        "recall_ms": 0,
        "embedding_cache_hit": 0,
    }


class RuntimeState:
    """管理进程内短期数据，并按 AstrBot 会话来源隔离所有记录。

    该对象保存群历史、消息发送者索引、图片缓存、用量、调试日志和自动记忆
    抽取节流状态，不负责长期持久化。来源进入 LRU 队列后，超过配置上限时
    会连同各映射中的该来源数据一起淘汰；消息索引与昵称索引另有局部上限，
    防止单个长期活跃群持续增长。
    """

    def __init__(self) -> None:
        """创建所有以会话来源为键的进程内状态映射。"""
        # 按 unified_msg_origin 隔离会话相关数据，避免不同群的状态相互混用。
        # 群聊历史文本（每条一行，已格式化）——供对话/主动回复使用
        self.session_chats: dict[str, list[str]] = defaultdict(list)
        # 记忆专用历史：始终包含完整日期、发送者 ID 与原始引用/@ 标记，
        # 不随对话上下文的格式开关变化，避免记忆时间/发言者被改写。
        self.memory_chats: dict[str, list[str]] = defaultdict(list)
        # 消息 ID -> 发送者 ID，用于校验 @ / 引用的目标是否真实存在
        self.message_senders: dict[str, OrderedDict[str, str]] = defaultdict(
            OrderedDict
        )
        # 消息 ID -> {urls: [...], captions: {index: 描述}, resolved: [...]}
        self.image_registry: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
        # QQ 号 -> 昵称，用于把模型输出的 "@昵称" 还原成真实 @
        self.sender_names: dict[str, dict[str, str]] = defaultdict(dict)
        # 主动回复触发计数
        self.reply_counters: dict[str, int] = defaultdict(int)
        # 本轮 token 用量（按来源累计）
        self.turn_usage: dict[str, dict[str, int]] = defaultdict(_new_turn_usage)
        # 查询向量缓存：query hash -> (vector, 写入时间)。全局共享，按 TTL 与条数淘汰，
        # 同一条问题在短时间内重复出现时不必再请求 embedding 服务。
        self.embedding_cache: OrderedDict[str, tuple[list[float], float]] = (
            OrderedDict()
        )
        # 回复调试开关（按来源）；F48：有界 LRU，长期运行不会无界增长
        self.debug_origins: OrderedDict[str, None] = OrderedDict()
        # 待发送的调试报告
        self.pending_report: dict[str, dict] = {}
        # 本轮执行日志
        self.turn_logs: dict[str, list[str]] = defaultdict(list)
        # 最近一条消息时间，用于按对话间隔切窗
        self.last_chat_at: dict[str, float] = {}
        # 上次已总结到的最后一条消息 ID，避免窗口重叠重复写入
        self.last_extract_msg: dict[str, str] = {}
        # 本轮开始时间（用于截取插件运行日志）
        self.turn_started_at: dict[str, float] = {}
        # 上一轮 system 段指纹，用于判断前缀缓存是否还能命中
        self.last_system_hash: dict[str, int] = {}
        # 空闲兜底上次触发时间，避免同一次空闲反复总结
        self.gap_flushed_at: dict[str, float] = {}
        # 自动记忆：抽取计数与上次抽取时间
        self.extract_counters: dict[str, int] = defaultdict(int)
        self.last_extract_at: dict[str, float] = {}
        # 抽取失败后的重试次数，成功后清零
        self.extract_retries: dict[str, int] = defaultdict(int)
        # LRU：最左侧最久未使用
        self.origin_lru: OrderedDict[str, None] = OrderedDict()

    # ------------------------------------------------------------------ #
    # 来源生命周期
    # ------------------------------------------------------------------ #
    def _evict_origin(self, origin: str) -> None:
        """从各状态容器中移除指定来源的数据，不触碰长期存储。"""
        # 清除来源关联的所有短期数据；debug_origins 的生命周期由插件管理。
        self.session_chats.pop(origin, None)
        self.memory_chats.pop(origin, None)
        self.message_senders.pop(origin, None)
        self.image_registry.pop(origin, None)
        self.sender_names.pop(origin, None)
        self.reply_counters.pop(origin, None)
        self.turn_usage.pop(origin, None)
        self.turn_logs.pop(origin, None)
        self.turn_started_at.pop(origin, None)
        self.pending_report.pop(origin, None)
        self.extract_counters.pop(origin, None)
        self.last_extract_at.pop(origin, None)
        self.extract_retries.pop(origin, None)
        # 这两个映射同样是按来源无限增长的，必须一起清，否则 LRU 只清了一半。
        self.last_chat_at.pop(origin, None)
        self.last_extract_msg.pop(origin, None)
        self.last_system_hash.pop(origin, None)
        self.gap_flushed_at.pop(origin, None)

    def touch_origin(self, origin: str, max_origins: int) -> None:
        """刷新来源的最近使用顺序，并按来源总数上限执行 LRU 淘汰。

        ``OrderedDict`` 左端代表最久未使用的会话。每次触达会先移除旧位置，
        再插入到队尾；当来源数超限时从队首逐个移出，并调用 ``_evict_origin``
        清理其在所有短期状态映射中的数据。限制至少为一个，避免把当前来源
        也立即淘汰掉。
        """
        if not origin:
            return
        self.origin_lru.pop(origin, None)
        self.origin_lru[origin] = None
        limit = max(1, int(max_origins))
        while len(self.origin_lru) > limit:
            oldest, _ = self.origin_lru.popitem(last=False)
            self._evict_origin(oldest)

    # ------------------------------------------------------------------ #
    # 一轮对话的用量 / 日志
    # ------------------------------------------------------------------ #
    def reset_turn(self, origin: str) -> None:
        """新的一轮开始：清空用量与执行日志。"""
        self.turn_started_at[origin] = time.time()
        self.turn_usage[origin] = _new_turn_usage()
        self.turn_logs[origin] = []

    def bump_metric(self, origin: str, key: str, value: int = 1) -> None:
        """累加本轮的自定义指标（召回命中、耗时、缓存命中）。

        后台写入可能发生在 ``reset_turn`` 之前或该来源已被 LRU 淘汰之后，
        因此缺桶时创建一个只含该指标的桶，而不是抛 ``KeyError``。
        """
        bucket = self.turn_usage.get(origin)
        if bucket is None:
            bucket = self.turn_usage[origin] = {}
        bucket[key] = int(bucket.get(key, 0)) + int(value)

    # ------------------------------------------------------------------ #
    # 查询向量缓存
    # ------------------------------------------------------------------ #
    def cache_get_embedding(self, key: str, ttl_seconds: int) -> list[float] | None:
        """读取未过期的查询向量；过期或未命中返回 None。"""
        if not key or ttl_seconds <= 0:
            return None
        item = self.embedding_cache.get(key)
        if item is None:
            return None
        vector, created_at = item
        if time.time() - created_at > max(1, int(ttl_seconds)):
            self.embedding_cache.pop(key, None)
            return None
        self.embedding_cache.move_to_end(key)
        return vector

    def cache_put_embedding(self, key: str, vector: list[float], max_size: int) -> None:
        """写入查询向量，并按容量上限淘汰最久未使用的条目。"""
        if not key or max_size <= 0 or not vector:
            return
        self.embedding_cache[key] = (list(vector), time.time())
        self.embedding_cache.move_to_end(key)
        while len(self.embedding_cache) > max(1, int(max_size)):
            self.embedding_cache.popitem(last=False)

    def add_usage_totals(
        self, origin: str, cached: int, other: int, output: int
    ) -> None:
        """按已归一的三个数值累加 token 用量。"""
        bucket = self.turn_usage[origin]
        bucket["input_cached"] += max(int(cached or 0), 0)
        bucket["input_other"] += max(int(other or 0), 0)
        bucket["output"] += max(int(output or 0), 0)
        bucket["calls"] += 1

    def set_debug(self, origin: str, enabled: bool) -> None:
        """启用或关闭指定来源的回复用量调试。"""
        # 调整该来源的 token 调试状态；该集合只存在于当前进程。
        if not origin:
            return
        if enabled:
            self.debug_origins.pop(origin, None)
            self.debug_origins[origin] = None
            # F48：与 origin_lru 解耦的调试集合同样需要容量上限。
            while len(self.debug_origins) > _DEBUG_ORIGINS_LIMIT:
                self.debug_origins.popitem(last=False)
        else:
            self.debug_origins.pop(origin, None)

    def is_debug(self, origin: str) -> bool:
        """判断指定来源是否开启用量调试。"""
        # 查询当前来源是否启用回复调试报告。
        return origin in self.debug_origins

    # ------------------------------------------------------------------ #
    # 便捷读写
    # ------------------------------------------------------------------ #
    def append_memory_chat(self, origin: str, line: str, max_messages: int) -> None:
        """追加记忆专用历史，并只保留最近 max_messages 条。"""
        chats = self.memory_chats[origin]
        chats.append(line)
        limit = max(10, int(max_messages))
        if len(chats) > limit:
            del chats[: len(chats) - limit]

    def memory_history(self, origin: str) -> list[str]:
        """返回记忆历史；升级后尚未产生新记忆行时回退到对话历史。"""
        return self.memory_chats.get(origin) or self.session_chats.get(origin) or []

    def append_chat(self, origin: str, line: str, max_messages: int) -> str:
        """追加一条已格式化的历史行，并将列表裁剪到指定长度。

        如果新增后超过上限，会从最旧的一端持续弹出记录。返回最后一条被移除
        的文本，方便调用方在需要时记录或检查被淘汰的历史；没有发生裁剪则
        返回空字符串。
        """
        chats = self.session_chats[origin]
        chats.append(line)
        removed = ""
        while len(chats) > max(1, int(max_messages)):
            removed = chats.pop(0)
        return removed

    def remember_message(self, origin: str, msg_id: str, sender_id: str) -> None:
        """登记消息发送者，供输出标签引用/目标校验使用。"""
        # 保存消息与发送者关联供标签校验，并限制映射最多保留 2000 条。
        if not msg_id:
            return
        registry = self.message_senders[origin]
        registry[msg_id] = str(sender_id or "")
        registry.move_to_end(msg_id)
        while len(registry) > 2000:
            registry.popitem(last=False)

    def remember_images(
        self,
        origin: str,
        msg_id: str,
        payload: dict[str, object],
        *,
        max_entries: int = 300,
    ) -> None:
        """登记一条消息的图片缓存，并限制单个来源的条目数。

        图片条目只有在该消息行被对话历史上限挤出时才回收；关闭群聊上下文
        增强时历史不会裁剪，缓存就会无限增长，因此这里再加一层硬上限。
        """
        if not msg_id:
            return
        registry = self.image_registry[origin]
        registry[msg_id] = payload
        while len(registry) > max(1, int(max_entries)):
            registry.pop(next(iter(registry)))

    def remember_sender(self, origin: str, sender_id: str, nickname: str) -> None:
        """记录 QQ 号与昵称的对应关系（用于无标签的 "@昵称" 兜底）。"""
        sender_id = str(sender_id or "").strip()
        nickname = str(nickname or "").strip()
        if not sender_id or not nickname:
            return
        names = self.sender_names[origin]
        names[sender_id] = nickname
        while len(names) > 500:
            names.pop(next(iter(names)))

    def name_map(self, origin: str) -> dict[str, str]:
        """昵称 -> QQ 号，只返回在本群唯一对应一个 QQ 的昵称。"""
        buckets: dict[str, list[str]] = {}
        for qq, name in self.sender_names.get(origin, {}).items():
            buckets.setdefault(name, []).append(qq)
        return {
            name: qq_list[0] for name, qq_list in buckets.items() if len(qq_list) == 1
        }

    def reset_counter(self, origin: str) -> None:
        """重置指定来源累计的主动回复消息数。"""
        # 清零当前会话的主动回复累计值。
        self.reply_counters[origin] = 0

    def bump_counter(self, origin: str) -> int:
        """将来源的主动回复计数加一并返回新计数。"""
        # 累加当前会话的主动回复计数并返回更新后的值。
        self.reply_counters[origin] = self.reply_counters.get(origin, 0) + 1
        return self.reply_counters[origin]
