"""相似记忆的写入决策（纯函数，不依赖 AstrBot 与数据库）。

向量相似只说明「话题接近」，不说明「同一件事」。因此 merge / duplicate
都必须先确认类型一致，避免把「喜欢火锅」(偏好) 与「周五吃火锅」(计划)
合并或丢弃。

写入路径不再产生 ``supersede``：同类型 + 实体交集不足以证明两条记忆互相
矛盾（「小明喜欢猫」和「小明养了一只狗」可以同时为真），旧规则在中文短句
上会把仍然有效的记忆静默藏起来。现在未达 dedup 阈值的一律 ``insert``，
两条都保留。``SimilarAction`` 仍保留 ``"supersede"`` 字面量，兼容外部引用。
"""

from __future__ import annotations

from typing import Any, Literal

SimilarAction = Literal["merge", "duplicate", "supersede", "insert"]


def normalize_kind(kind: Any) -> str:
    """类型规整：去空白、转小写；空值视为空串。"""
    return str(kind or "").strip().lower()


def same_kind(old_kind: Any, new_kind: Any) -> bool:
    """两条记忆的类型是否一致。"""
    return normalize_kind(old_kind) == normalize_kind(new_kind)


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
    """根据最相似旧记忆决定新记忆如何处理（实体参数仅为兼容保留）。

    规则（按顺序）：
    1. 跨作用域：一律 ``insert``，群里的总结不能改写全局记忆或别的群的记忆；
    2. 类型不同：一律 ``insert``，两条都保留；
    3. 同类型且 ``similarity >= merge_threshold``：``merge``（更新旧行正文）；
    4. 同类型且 ``similarity >= dedup_threshold``：``duplicate``（丢弃新记忆）；
    5. 其余（含同类型 + 实体交集但未达 dedup 阈值）：``insert``，两条都保留。

    同类型 + 实体交集不再返回 ``supersede``：中文短句的 embedding 相似度普遍
    偏高，仅凭「类型相同、提到同一个实体」无法证明两条记忆互斥。旧记忆被
    标记 ``superseded_by`` 后不再参与召回，会造成静默丢记忆（M-01）。

    调用方保证传入的旧记忆已达到冲突阈值（相似查找已按阈值筛过）。
    ``old_entities`` / ``new_entities`` 仅为签名兼容保留。
    """
    if not same_scope:
        return "insert"
    if not same_kind(old_kind, new_kind):
        return "insert"
    if similarity >= merge_threshold:
        return "merge"
    if similarity >= dedup_threshold:
        return "duplicate"
    return "insert"
