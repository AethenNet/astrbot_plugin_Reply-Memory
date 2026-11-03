"""相似记忆决策、相似查找与 kind 守卫（不依赖 AstrBot 运行时）。"""

from __future__ import annotations

import asyncio

import pytest
from _helpers import (  # noqa: I001
    add,
    load_store_memory,
    make_plugin,
    store_one,
    vec_with_sim,
)
from astrbot_plugin_active_reply.memory_conflict import (
    decide_similar_action,
    entities_overlap,
    same_kind,
)
from astrbot_plugin_active_reply.memory_store import MemoryStore

DEDUP, MERGE = 0.90, 0.95
REGIONS = {"merge": 0.97, "dup": 0.92, "low": 0.80}


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


# ---- 决策矩阵：同/跨作用域 × 同/异 kind × 有无实体 × 三个阈值区间 ---------


def _oracle(same_scope: bool, kind_same: bool, overlap: bool, region: str) -> str:
    if not same_scope:
        return "insert"
    if not kind_same:
        return "insert"
    if region == "merge":
        return "merge"
    if region == "dup":
        return "duplicate"
    return "supersede" if overlap else "insert"


@pytest.mark.parametrize("region", ["merge", "dup", "low"])
@pytest.mark.parametrize("overlap", [True, False], ids=["entity", "no-entity"])
@pytest.mark.parametrize("kind_same", [True, False], ids=["same-kind", "diff-kind"])
@pytest.mark.parametrize("same_scope", [True, False], ids=["same-scope", "cross-scope"])
def test_decision_matrix(same_scope, kind_same, overlap, region):
    action = decide_similar_action(
        same_scope=same_scope,
        similarity=REGIONS[region],
        old_kind="preference",
        new_kind="preference" if kind_same else "plan",
        old_entities=["小明"],
        new_entities=["小明", "火锅"] if overlap else ["小红"],
        dedup_threshold=DEDUP,
        merge_threshold=MERGE,
    )
    assert action == _oracle(same_scope, kind_same, overlap, region)


@pytest.mark.parametrize(
    "similarity,expected",
    [(MERGE, "merge"), (MERGE - 1e-9, "duplicate"), (DEDUP, "duplicate")],
)
def test_threshold_boundaries(similarity, expected):
    assert (
        decide_similar_action(True, similarity, "fact", "fact", [], [], DEDUP, MERGE)
        == expected
    )


def test_below_dedup_without_entities_inserts():
    assert (
        decide_similar_action(True, 0.8, "fact", "fact", [], [], DEDUP, MERGE)
        == "insert"
    )


def test_kind_compare_is_case_and_space_insensitive():
    assert same_kind(" Fact ", "fact")
    assert same_kind(None, "")
    assert not same_kind("fact", "plan")


def test_entities_overlap_normalizes():
    assert entities_overlap(["Alice "], ["alice"])
    assert not entities_overlap([], ["a"])
    assert not entities_overlap(["a"], None)


@pytest.mark.parametrize(
    "old_kind,old_ents,new_kind,new_ents,expected",
    [
        ("preference", ["小明"], "preference", ["小明"], True),
        ("preference", ["小明"], "plan", ["小明"], False),
        ("fact", ["小明"], "fact", ["小红"], False),
        ("fact", [], "fact", ["小明"], False),
        ("fact", ["小明"], "fact", [], False),
    ],
)
def test_store_should_supersede_delegates(
    old_kind, old_ents, new_kind, new_ents, expected
):
    assert (
        MemoryStore.should_supersede(old_kind, old_ents, new_kind, new_ents) is expected
    )


# ---- find_similar_memory ---------------------------------------------------


def test_similar_returns_kind_and_entities(store):
    mid = add(store, "小明喜欢火锅", [1, 0, 0], kind="preference", entities=["小明"])
    hit = store.find_similar_memory([1, 0.05, 0], group_scope="g1", threshold=0.75)
    assert hit["memory_id"] == mid
    assert (hit["kind"], hit["entities"]) == ("preference", ["小明"])


def test_similar_excludes_archived(store):
    mid = add(store, "旧计划", [1, 0, 0])
    assert store.archive_expired(now=0) == 0
    with store._connect() as conn:  # noqa: SLF001
        conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (mid,))
        conn.commit()
    assert store.find_similar_memory([1, 0, 0], group_scope="g1", threshold=0.5) is None


def test_similar_excludes_superseded(store):
    old = add(store, "旧", [1, 0, 0])
    new = add(store, "新", [0, 1, 0])
    store.supersede(old, new)
    assert store.find_similar_memory([1, 0, 0], group_scope="g1", threshold=0.5) is None
    assert (
        store.find_similar_memory([0, 1, 0], group_scope="g1", threshold=0.5)[
            "memory_id"
        ]
        == new
    )


def test_similar_scope_isolation(store):
    add(store, "别的群", [1, 0, 0], scope="g2")
    assert store.find_similar_memory([1, 0, 0], group_scope="g1", threshold=0.5) is None
    gid = add(store, "全局", [1, 0, 0], scope="")
    assert (
        store.find_similar_memory([1, 0, 0], group_scope="g1", threshold=0.5)[
            "memory_id"
        ]
        == gid
    )
    assert (
        store.find_similar_memory([1, 0, 0], group_scope="", threshold=0.5)["memory_id"]
        == gid
    )


def test_similar_dimension_mismatch_skipped(store):
    add(store, "四维", [1, 0, 0, 0])
    assert store.find_similar_memory([1, 0, 0], group_scope="g1", threshold=0.5) is None


def test_similar_below_threshold_none(store):
    add(store, "a", [1, 0, 0])
    assert (
        store.find_similar_memory([0, 1, 0], group_scope="g1", threshold=0.75) is None
    )


def test_similar_picks_best(store):
    add(store, "远", [0, 1, 0])
    best = add(store, "近", [1, 0.01, 0])
    add(store, "中", [1, 1, 0])
    assert (
        store.find_similar_memory([1, 0, 0], group_scope="g1", threshold=0.5)[
            "memory_id"
        ]
        == best
    )


def test_similar_without_numpy(store, monkeypatch):
    from astrbot_plugin_active_reply import vector_math

    monkeypatch.setattr(vector_math, "_np", None)
    mid = add(store, "x", [1, 0, 0])
    assert (
        store.find_similar_memory([1, 0, 0], group_scope="g1", threshold=0.9)[
            "memory_id"
        ]
        == mid
    )


# ---- _store_memory 的 kind 守卫（真实函数源码，伪造外部依赖） --------------


def _setup(store, *, old_kind, new_kind, sim, old_ents=(), new_ents=(), scope="g1"):
    fn = load_store_memory()
    old_id = add(
        store, "旧正文", [1.0, 0.0], scope=scope, kind=old_kind, entities=list(old_ents)
    )
    plugin = make_plugin(store, {"新正文": vec_with_sim(sim)})
    result = asyncio.run(
        store_one(
            plugin, fn, "新正文", kind=new_kind, entities=list(new_ents), scope=scope
        )
    )
    return old_id, result


def _rows(store):
    with store._connect() as conn:  # noqa: SLF001
        return [
            (r["id"], r["content"], r["superseded_by"])
            for r in conn.execute(
                "SELECT id, content, superseded_by FROM memories ORDER BY id"
            )
        ]


@pytest.mark.parametrize(
    "sim", [0.97, 0.92, 0.80], ids=["merge-zone", "dup-zone", "low-zone"]
)
def test_different_kind_never_merge_dedup_or_supersede(store, sim):
    old_id, (new_id, dup_id) = _setup(
        store,
        old_kind="preference",
        new_kind="plan",
        sim=sim,
        old_ents=["小明"],
        new_ents=["小明"],
    )
    assert new_id is not None and dup_id is None
    rows = _rows(store)
    assert [r[1] for r in rows] == ["旧正文", "新正文"]  # 旧正文未被改写
    assert all(r[2] is None for r in rows)  # 无人被取代


def test_same_kind_merge_rewrites_old(store):
    old_id, (new_id, dup_id) = _setup(store, old_kind="fact", new_kind="fact", sim=0.97)
    assert new_id is None and dup_id == old_id
    assert _rows(store) == [(old_id, "新正文", None)]


def test_same_kind_duplicate_drops_new(store):
    old_id, (new_id, dup_id) = _setup(store, old_kind="fact", new_kind="fact", sim=0.92)
    assert new_id is None and dup_id == old_id
    assert _rows(store) == [(old_id, "旧正文", None)]


def test_same_kind_with_entity_supersedes(store):
    old_id, (new_id, dup_id) = _setup(
        store,
        old_kind="fact",
        new_kind="fact",
        sim=0.80,
        old_ents=["小明"],
        new_ents=["小明"],
    )
    assert new_id is not None and dup_id is None
    assert _rows(store) == [(old_id, "旧正文", new_id), (new_id, "新正文", None)]


def test_same_kind_without_entity_keeps_both(store):
    old_id, (new_id, _) = _setup(store, old_kind="fact", new_kind="fact", sim=0.80)
    assert _rows(store) == [(old_id, "旧正文", None), (new_id, "新正文", None)]


def test_cross_scope_never_touches_other_scope(store):
    """群里的相似总结不能改写、取代或吞掉全局记忆。"""
    fn = load_store_memory()
    old_id = add(
        store, "全局旧正文", [1.0, 0.0], scope="", kind="fact", entities=["小明"]
    )
    plugin = make_plugin(store, {"群里新正文": vec_with_sim(0.97)})
    new_id, dup = asyncio.run(
        store_one(plugin, fn, "群里新正文", scope="g1", kind="fact", entities=["小明"])
    )
    assert new_id is not None and dup is None
    assert _rows(store) == [(old_id, "全局旧正文", None), (new_id, "群里新正文", None)]
