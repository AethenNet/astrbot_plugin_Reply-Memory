"""存储层测试：作用域去重、BLOB 写入、合并与回填。"""

from __future__ import annotations

import time

import pytest
from astrbot_plugin_active_reply.memory_scope import ScopeFilter
from astrbot_plugin_active_reply.memory_store import MemoryStore

VECTOR = [0.1] * 8


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    """临时目录里的空记忆库。"""
    return MemoryStore(tmp_path / "memory.db")


def add(store: MemoryStore, content: str, scope: str, sender: str, **kwargs) -> int:
    """写入一条测试记忆。"""
    return store.add_memory(
        content=content,
        embedding=VECTOR,
        role_ids=[sender],
        group_scope=scope,
        group_id="111" if scope else "",
        platform_id="qq",
        sender_id=sender,
        **kwargs,
    )


def test_exact_dedup_is_scope_specific(store: MemoryStore) -> None:
    first = add(store, "同一句话", "qq:111", "alice")
    second = add(store, "同一句话", "qq:222", "bob")
    assert first != second
    assert store.find_exact_memory("同一句话", "qq:111") == first
    assert store.find_exact_memory("同一句话", "qq:222") == second
    assert store.find_exact_memory("同一句话", "qq:333") is None


def test_vectors_are_stored_as_blob_only(store: MemoryStore) -> None:
    memory_id = add(store, "只存 BLOB", "qq:111", "alice")
    with store._connect() as conn:
        row = conn.execute(
            "SELECT embedding, length(embedding_blob) AS blob_len "
            "FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
    assert row["embedding"] == ""
    assert row["blob_len"]


def test_search_returns_similarity_and_respects_visibility(store: MemoryStore) -> None:
    add(store, "群A 火锅", "qq:111", "alice")
    add(store, "群B 火锅", "qq:222", "bob")
    add(store, "全局 火锅", "", "alice")
    flt = ScopeFilter(mode="group_first", current_scope="qq:111", sender_id="alice")
    hits = store.search_memories(query_embedding=VECTOR, recall_k=10, scope_filter=flt)
    contents = [hit["content"] for hit in hits]
    assert "群A 火锅" in contents
    assert "全局 火锅" in contents
    assert "群B 火锅" not in contents
    assert all(hit["similarity"] is not None for hit in hits)


def test_merge_keeps_the_later_expiry(store: MemoryStore) -> None:
    memory_id = add(store, "计划", "qq:111", "alice")
    later = time.time() + 1000
    earlier = time.time() + 100
    store.merge_memory(memory_id, content="计划v2", embedding=VECTOR, valid_until=later)
    store.merge_memory(
        memory_id, content="计划v3", embedding=VECTOR, valid_until=earlier
    )
    record = store.get_memory(memory_id)
    assert record is not None
    assert record["valid_until"] == pytest.approx(later, rel=1e-6)
    assert record["content"] == "计划v3"


def test_backfill_roles_from_sender(store: MemoryStore) -> None:
    memory_id = add(store, "旧记忆", "", "alice")
    with store._connect() as conn:
        conn.execute("DELETE FROM memory_roles WHERE memory_id = ?", (memory_id,))
    filled = store.backfill_roles_from_sender()
    assert filled == 1
    record = store.get_memory(memory_id)
    assert record is not None
    assert "alice" in record["related_role_ids"]
    assert store.backfill_roles_from_sender() == 0
