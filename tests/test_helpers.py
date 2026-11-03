"""辅助模块测试：向量计算、检索缓存、工具预算、配置解析。"""

from __future__ import annotations

import json
import time
from array import array

import astrbot_plugin_active_reply.vector_math as vector_math
import pytest
from astrbot_plugin_active_reply.config_model import parse_plugin_config
from astrbot_plugin_active_reply.recall_budget import RecallBudget, get_budget
from astrbot_plugin_active_reply.recall_cache import RecallCache


def test_batch_cosine_matches_python_fallback() -> None:
    query = [0.6, 0.8, 0.0]
    vectors = [
        array("f", [0.6, 0.8, 0.0]).tobytes(),
        json.dumps([0.0, 1.0, 0.0]),
        [1.0, 2.0],  # 维度不一致，记 -1
    ]
    fast = vector_math.cosine_scores(query, vectors)
    saved = vector_math._np
    try:
        vector_math._np = None
        slow = vector_math.cosine_scores(query, vectors)
    finally:
        vector_math._np = saved
    assert fast == pytest.approx(slow, abs=1e-5)
    assert fast[2] == -1.0


def test_recall_cache_lru_and_ttl() -> None:
    cache = RecallCache(ttl_seconds=1.0, max_entries=10)
    for index in range(12):
        cache.put(f"k{index}", index)
    assert len(cache) == 10
    assert cache.get("k0") is None
    cache.put("ttl", 1)
    time.sleep(1.1)
    assert cache.get("ttl") is None


def test_recall_budget_is_shared_per_event_and_clipped() -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self.extras: dict[str, object] = {}

        def get_extra(self, key: str, default=None):
            return self.extras.get(key, default)

        def set_extra(self, key: str, value) -> None:
            self.extras[key] = value

    event = FakeEvent()
    first = get_budget(event)
    assert get_budget(event) is first
    assert [first.take() for _ in range(5)] == [True, True, True, True, False]
    clipped = RecallBudget().clip("x" * 20000)
    assert len(clipped) < 20000


def test_config_defaults_for_visibility_switches() -> None:
    cfg = parse_plugin_config({"memory_rag": {"memory_control": {"enable": True}}})
    assert cfg.memory_rag.cross_group_personal is True
    assert cfg.memory_rag.cross_group_private is False


def test_config_tolerates_broken_kind_decay() -> None:
    cfg = parse_plugin_config(
        {"memory_rag": {"memory_evaluation": {"memory_kind_decay": "坏数据"}}}
    )
    assert cfg.memory_rag.kind_decay.fact == 120
    assert cfg.memory_rag.kind_decay.plan == 3


def test_scope_labels_are_renamed() -> None:
    from astrbot_plugin_active_reply.config_model import GROUP_OVERRIDE_FIELDS

    spec = next(
        item
        for item in GROUP_OVERRIDE_FIELDS
        if item.path == "memory_rag.memory_generation.auto_extract_scope"
    )
    assert spec.labels[0] == "自动"
    assert spec.labels[1] == "仅全局"
