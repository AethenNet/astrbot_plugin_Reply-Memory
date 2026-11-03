"""可见性规则测试：本会话 ∪ 本人参与。"""

from __future__ import annotations

import sqlite3

import pytest
from astrbot_plugin_active_reply.memory_scope import ScopeFilter, normalize_mode

ROWS = [
    (1, "群A-爱丽丝", "qq:111", "alice"),
    (2, "群B-鲍勃", "qq:222", "bob"),
    (3, "全局-爱丽丝", "", "alice"),
    (4, "私聊-爱丽丝", "qq:private:alice", "alice"),
    (5, "群A-鲍勃", "qq:111", "bob"),
]


@pytest.fixture()
def db() -> sqlite3.Connection:
    """内存库：两条群内记忆、一条全局、一条私聊。"""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE memories(
            id INTEGER PRIMARY KEY, content TEXT, group_scope TEXT, sender_id TEXT
        );
        CREATE TABLE memory_roles(memory_id INTEGER, role_id TEXT);
        """
    )
    conn.executemany("INSERT INTO memories VALUES (?,?,?,?)", ROWS)
    conn.executemany(
        "INSERT INTO memory_roles VALUES (?,?)",
        [(1, "alice"), (2, "bob"), (3, "alice"), (4, "alice"), (5, "bob")],
    )
    return conn


def visible(conn: sqlite3.Connection, flt: ScopeFilter) -> list[str]:
    """执行可见性条件，返回命中的记忆正文。"""
    clause, params = flt.clause("m")
    sql = f"SELECT m.content FROM memories AS m WHERE {clause} ORDER BY m.id"
    return [row[0] for row in conn.execute(sql, params)]


def test_auto_sees_current_group_and_personal_global(db: sqlite3.Connection) -> None:
    flt = ScopeFilter(mode="group_first", current_scope="qq:111", sender_id="alice")
    assert visible(db, flt) == ["群A-爱丽丝", "全局-爱丽丝", "群A-鲍勃"]


def test_other_group_memories_are_isolated(db: sqlite3.Connection) -> None:
    flt = ScopeFilter(mode="group_first", current_scope="qq:111", sender_id="carol")
    assert visible(db, flt) == ["群A-爱丽丝", "群A-鲍勃"]


def test_cross_group_only_returns_participated(db: sqlite3.Connection) -> None:
    flt = ScopeFilter(mode="group_first", current_scope="", sender_id="bob")
    # Bob 在群 A 与群 B 都参与过；没有当前会话时只按参与者取。
    assert visible(db, flt) == ["群B-鲍勃", "群A-鲍勃"]


def test_global_mode_is_participant_filtered(db: sqlite3.Connection) -> None:
    flt = ScopeFilter(mode="global", current_scope="qq:111", sender_id="alice")
    assert visible(db, flt) == ["全局-爱丽丝"]


def test_private_scope_excluded_from_group_by_default(db: sqlite3.Connection) -> None:
    flt = ScopeFilter(mode="group_first", current_scope="qq:111", sender_id="alice")
    assert "私聊-爱丽丝" not in visible(db, flt)


def test_private_scope_included_when_allowed(db: sqlite3.Connection) -> None:
    flt = ScopeFilter(
        mode="group_first",
        current_scope="qq:111",
        sender_id="alice",
        allow_private=True,
    )
    assert "私聊-爱丽丝" in visible(db, flt)


def test_personal_channel_can_be_disabled(db: sqlite3.Connection) -> None:
    flt = ScopeFilter(
        mode="group_first",
        current_scope="qq:111",
        sender_id="alice",
        personal_enabled=False,
    )
    assert visible(db, flt) == ["群A-爱丽丝", "群A-鲍勃"]


def test_python_check_agrees_with_sql(db: sqlite3.Connection) -> None:
    flt = ScopeFilter(mode="group_first", current_scope="qq:111", sender_id="alice")
    records = [
        {
            "content": content,
            "group_scope": scope,
            "sender_id": sender,
            "related_role_ids": [sender],
        }
        for _, content, scope, sender in ROWS
    ]
    python_side = sorted(row["content"] for row in records if flt.is_visible(row))
    assert sorted(visible(db, flt)) == python_side


def test_unknown_mode_falls_back_to_auto() -> None:
    assert normalize_mode("乱填") == "group_first"
    assert normalize_mode("global") == "global"
    assert normalize_mode(None) == "group_first"


def test_empty_scope_and_sender_matches_nothing() -> None:
    clause, params = ScopeFilter(mode="group_first").clause()
    assert clause == "0"
    assert params == []
