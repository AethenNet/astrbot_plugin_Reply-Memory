"""删除不再重排 ID：批量删除不能误删。"""

from __future__ import annotations

import time

import pytest
from _helpers import add
from astrbot_plugin_active_reply.memory_store import MemoryStore


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(tmp_path / "m.db")


def _fill(store, n):
    return [add(store, f"记忆{i}", [1.0, float(i), 0.0]) for i in range(1, n + 1)]


def _snapshot(store):
    with store._connect() as conn:  # noqa: SLF001
        return {
            r["id"]: r["content"]
            for r in conn.execute("SELECT id, content FROM memories")
        }


def test_resequence_api_is_gone():
    assert not hasattr(MemoryStore, "resequence_ids")


def test_delete_5_then_10_only_removes_those(store):
    ids = _fill(store, 12)
    before = _snapshot(store)
    for target in (5, 10):  # 巩固流程：按 cluster 的旧 ID 连续删除
        assert store.delete_memory(target) is True
    after = _snapshot(store)
    assert set(before) - set(after) == {5, 10}
    assert all(after[i] == before[i] for i in after)  # 其余 ID 与内容不变
    assert len(after) == 10 and ids == list(range(1, 13))


def test_delete_in_reverse_order_same_result(store):
    _fill(store, 12)
    before = _snapshot(store)
    for target in (10, 5):
        store.delete_memory(target)
    assert set(before) - set(_snapshot(store)) == {5, 10}


def test_delete_missing_or_invalid_returns_false(store):
    _fill(store, 3)
    assert store.delete_memory(99) is False
    assert store.delete_memory(0) is False
    assert store.delete_memory("abc") is False  # type: ignore[arg-type]
    assert len(_snapshot(store)) == 3


def test_deleted_id_not_reused(store):
    _fill(store, 12)
    store.delete_memory(12)
    assert add(store, "新的", [1, 0, 0]) == 13


def test_ids_keep_gaps_after_delete(store):
    _fill(store, 5)
    store.delete_memory(2)
    assert sorted(_snapshot(store)) == [1, 3, 4, 5]


def test_superseded_by_stays_valid_after_unrelated_delete(store):
    _fill(store, 6)
    store.supersede(5, 6)
    store.delete_memory(2)
    with store._connect() as conn:  # noqa: SLF001
        row = conn.execute("SELECT superseded_by FROM memories WHERE id = 5").fetchone()
    assert row["superseded_by"] == 6


def test_roles_follow_their_rows_after_delete(store):
    for i in range(1, 6):
        store.add_memory(content=f"m{i}", embedding=[1, 0], role_ids=[f"u{i}"])
    store.delete_memory(2)
    with store._connect() as conn:  # noqa: SLF001
        rows = {
            r["memory_id"]: r["role_id"]
            for r in conn.execute("SELECT memory_id, role_id FROM memory_roles")
        }
    assert rows == {1: "u1", 3: "u3", 4: "u4", 5: "u5"}


def test_fts_stays_consistent_after_delete(store):
    if not store.fts_enabled:
        pytest.skip("FTS5 不可用")
    ids = [add(store, f"独特词{i}号记忆", [1.0, float(i), 0.0]) for i in range(1, 6)]
    store.delete_memory(ids[1])
    with store._connect() as conn:  # noqa: SLF001
        hits = {
            r["rowid"]
            for r in conn.execute(
                "SELECT rowid FROM memory_fts WHERE memory_fts MATCH '独特词'"
            )
        }
    assert ids[1] not in hits
    assert {ids[0], ids[2], ids[3], ids[4]} <= hits or hits <= set(ids)


def test_trash_restore_creates_new_id_without_disturbing_others(store):
    _fill(store, 6)
    store.delete_memory(3)
    trash = store.latest_trash_id()
    new_id = store.restore_memory(trash)
    snap = _snapshot(store)
    assert new_id == 7 and snap[7] == "记忆3"
    assert all(snap[i] == f"记忆{i}" for i in (1, 2, 4, 5, 6))


def test_delete_goes_to_trash(store):
    _fill(store, 3)
    store.delete_memory(2)
    assert store.latest_trash_id() is not None


def test_delete_is_fast_on_large_db(store):
    """修复前 3000 条库删一条约 862ms（全表重排 + 全量重建 FTS）。"""
    with store._connect() as conn:  # noqa: SLF001
        conn.executemany(
            "INSERT INTO memories (content, embedding, embedding_dim, memory_time, "
            "created_at, updated_at) VALUES (?, '', 2, 1, 1, 1)",
            [(f"批量{i}",) for i in range(3000)],
        )
        conn.commit()
    started = time.perf_counter()
    assert store.delete_memory(1500) is True
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 100, f"删除耗时 {elapsed_ms:.1f}ms"
