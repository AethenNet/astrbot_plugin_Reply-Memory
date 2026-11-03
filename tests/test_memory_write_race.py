"""并发写入：锁内串行，不出现双写（真实 _store_memory 源码）。"""

from __future__ import annotations

import asyncio

import pytest
from _helpers import load_store_memory, make_plugin, store_one
from astrbot_plugin_active_reply.memory_store import MemoryStore

SAME_VEC = [1.0, 0.0]


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _count(store):
    with store._connect() as conn:  # noqa: SLF001
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


async def _race(plugin, fn, n=2):
    return await asyncio.gather(
        *(store_one(plugin, fn, f"同一件事-{i}") for i in range(n))
    )


def _vectors(n):
    return {f"同一件事-{i}": SAME_VEC for i in range(n)}


def test_two_concurrent_writes_do_not_double_write(store):
    fn = load_store_memory()
    plugin = make_plugin(store, _vectors(2))
    results = asyncio.run(_race(plugin, fn, 2))
    assert _count(store) == 1
    assert sorted(r[0] is None for r in results) == [False, True]  # 一写一合并


def test_many_concurrent_writes_collapse_to_one(store):
    fn = load_store_memory()
    plugin = make_plugin(store, _vectors(8))
    asyncio.run(_race(plugin, fn, 8))
    assert _count(store) == 1


def test_without_lock_the_race_is_real(store):
    """对照：去掉锁后同样的并发会双写，说明上面的测试确实能抓到竞态。"""
    fn = load_store_memory()
    plugin = make_plugin(store, _vectors(2), with_lock=False)
    asyncio.run(_race(plugin, fn, 2))
    assert _count(store) == 2


def test_different_kinds_concurrent_both_kept(store):
    fn = load_store_memory()
    plugin = make_plugin(store, _vectors(2))

    async def run():
        return await asyncio.gather(
            store_one(plugin, fn, "同一件事-0", kind="preference"),
            store_one(plugin, fn, "同一件事-1", kind="plan"),
        )

    asyncio.run(run())
    assert _count(store) == 2


def test_exact_duplicate_short_circuits_before_embedding(store):
    fn = load_store_memory()
    plugin = make_plugin(store, {"x": SAME_VEC})
    first = asyncio.run(store_one(plugin, fn, "x"))
    second = asyncio.run(store_one(plugin, fn, "x"))
    assert first[0] is not None and second == (None, first[0])
    assert _count(store) == 1
