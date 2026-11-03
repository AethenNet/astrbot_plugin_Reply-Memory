"""测试公用工具（不依赖 AstrBot 运行时）。"""

from __future__ import annotations

import ast
import asyncio
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve()
# 两种目录布局都支持：tests/ 与插件目录同级，或 tests/ 放在插件目录内部。
_SIBLING = HERE.parents[1] / "astrbot_plugin_active_reply"
if _SIBLING.is_dir():
    PLUGIN_DIR = _SIBLING
    PACKAGE_PARENT = HERE.parents[1]
else:
    PLUGIN_DIR = HERE.parents[1]
    PACKAGE_PARENT = PLUGIN_DIR.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


def vec_with_sim(sim: float) -> list[float]:
    """与 (1, 0) 余弦相似度恰为 sim 的二维单位向量。"""
    return [sim, math.sqrt(max(0.0, 1.0 - sim * sim))]


def add(store, content, vec, *, scope="g1", kind="fact", entities=None, **kw):
    return store.add_memory(
        content=content,
        embedding=vec,
        role_ids=["u1"],
        group_scope=scope,
        kind=kind,
        entities=entities or [],
        **kw,
    )


def load_store_memory():
    """从 main.py 源码里抽出真实的 ``_store_memory`` 协程函数。

    main.py 在模块顶层导入 AstrBot，测试环境没有它；这里用 ast 只取这一个
    函数的源码并在受控命名空间里编译，测到的仍是插件里的真实代码。
    """
    from astrbot_plugin_active_reply.memory_conflict import decide_similar_action

    source = (PLUGIN_DIR / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_store_memory"
    )
    code = ast.get_source_segment(source, node)
    import textwrap

    code = textwrap.dedent(code)
    namespace: dict[str, Any] = {
        "asyncio": asyncio,
        "time": time,
        "Any": Any,
        "PluginConfig": object,
        "decide_similar_action": decide_similar_action,
        "logger": logging.getLogger("test_store_memory"),
    }
    exec(compile(code, "main.py::_store_memory", "exec"), namespace)  # noqa: S102
    return namespace["_store_memory"]


class FakeProvider:
    """按正文返回预设向量；每次调用让出事件循环，便于暴露并发竞态。"""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    async def get_embedding(self, text: str) -> list[float]:
        await asyncio.sleep(0)
        return self.vectors[text]


class _NoLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def make_plugin(store, vectors, *, with_lock=True, dedup=0.9, conflict=0.75):
    cfg = SimpleNamespace(
        memory_rag=SimpleNamespace(dedup_threshold=dedup, conflict_threshold=conflict)
    )
    return SimpleNamespace(
        memory_store=store,
        _memory_write_lock=asyncio.Lock() if with_lock else _NoLock(),
        _embedding_provider=lambda: FakeProvider(vectors),
        _global_cfg=lambda: cfg,
        _journal=lambda *a, **k: None,
    )


async def store_one(plugin, fn, content, **kw):
    params = dict(
        content=content,
        role_ids=["u1"],
        scope="g1",
        group_id="g1",
        platform_id="p",
        source="test",
        origin="o",
        kind="fact",
        entities=[],
        importance=0.5,
    )
    params.update(kw)
    return await fn(plugin, **params)
