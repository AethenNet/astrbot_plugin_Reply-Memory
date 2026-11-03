"""相似记忆的写入决策（纯函数，不依赖 AstrBot 与数据库）。

向量相似只说明「话题接近」，不说明「同一件事」。因此 merge / duplicate /
supersede 都必须先确认类型一致，避免把「喜欢火锅」(偏好) 与「周五吃火锅」
(计划) 合并、丢弃或互相取代。
"""

from __future__ import annotations

from typing import Any, Literal

SimilarAction = Literal["merge", "duplicate", "supersede", "insert"]


def normalize_kind(kind: Any) -> str:
    """类型规整：去空白、转小写；空值视为空串。"""
    return str(kind or "").strip().lower()


def entity_keys(entities: Any) -> set[str]:
    """把实体列表规整为可比较的集合（去空白、忽略大小写）。"""
    if not isinstance(entities, (list, tuple, set)):
        return set()
    return {str(item).strip().casefold() for item in entities if str(item).strip()}


def same_kind(old_kind: Any, new_kind: Any) -> bool:
    """两条记忆的类型是否一致。"""
    return normalize_kind(old_kind) == normalize_kind(new_kind)


def entities_overlap(old_entities: Any, new_entities: Any) -> bool:
    """实体集合是否有交集；任何一方为空都视为无法确认指向同一对象。"""
    return bool(entity_keys(old_entities) & entity_keys(new_entities))


def decide_similar_action(
    same_scope: bool,
    similarity: float,
    old_kind: Any,
    new_kind: Any,
    old_entities: Any,
    new_entities: Any,
    dedup_threshold: float,
    merge_threshold: float,
) -> SimilarAction:
    """根据最相似旧记忆决定新记忆如何处理。

    规则（按顺序）：
    1. 跨作用域：一律 ``insert``，群里的总结不能改写全局记忆或别的群的记忆；
    2. 类型不同：一律 ``insert``，两条都保留；
    3. 同类型且 ``similarity >= merge_threshold``：``merge``（更新旧行正文）；
    4. 同类型且 ``similarity >= dedup_threshold``：``duplicate``（丢弃新记忆）；
    5. 同类型且实体有交集：``supersede``（旧记忆被取代）；
    6. 其余：``insert``。

    调用方保证传入的旧记忆已达到冲突阈值（相似查找已按阈值筛过）。
    """
    if not same_scope:
        return "insert"
    if not same_kind(old_kind, new_kind):
        return "insert"
    if similarity >= merge_threshold:
        return "merge"
    if similarity >= dedup_threshold:
        return "duplicate"
    if entities_overlap(old_entities, new_entities):
        return "supersede"
    return "insert"
