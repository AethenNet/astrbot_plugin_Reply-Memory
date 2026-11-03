"""记忆召回基准。

用法：
    python tests/benchmark_recall.py            # 100 / 600 / 1200 条
    python tests/benchmark_recall.py 5000       # 自定义规模

只依赖标准库 + 插件自身；用随机向量填充临时库，测的是检索路径本身的耗时。
"""

from __future__ import annotations

import random
import sqlite3
import statistics
import sys
import tempfile
import time
from array import array
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR.parent))

from astrbot_plugin_active_reply.memory_scope import ScopeFilter  # noqa: E402
from astrbot_plugin_active_reply.memory_store import MemoryStore  # noqa: E402

DIM = 1024
RUNS = 20


def build_store(size: int) -> tuple[MemoryStore, list[float]]:
    """构造一个指定规模的临时记忆库，并返回查询向量。"""
    store = MemoryStore(Path(tempfile.mkdtemp()) / "bench.db")
    rng = random.Random(42)
    templates = [[rng.random() for _ in range(DIM)] for _ in range(min(50, size))]
    now = time.time()
    rows = []
    for index in range(size):
        vector = templates[index % len(templates)]
        rows.append(
            (
                f"记忆{index} 火锅 聚餐 讨论",
                "",
                DIM,
                now - index,
                "qq:111",
                "111",
                "qq",
                "{}",
                now,
                now,
                array("f", vector).tobytes(),
                1.0,
                "火锅 聚餐 讨论",
            )
        )
    with sqlite3.connect(store.db_path) as conn:
        conn.executemany(
            "INSERT INTO memories(content, embedding, embedding_dim, memory_time, "
            "group_scope, group_id, platform_id, extra_metadata, created_at, "
            "updated_at, embedding_blob, embedding_norm, grams) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    query = [rng.random() for _ in range(DIM)]
    return store, query


def measure(
    store: MemoryStore, query: list[float], scope: ScopeFilter
) -> tuple[float, float]:
    """返回 (P50, P95) 毫秒。"""
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        store.search_memories(
            query_embedding=query,
            recall_k=20,
            max_results=10,
            scope_filter=scope,
            scan_limit=5000,
        )
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return statistics.median(samples), samples[int(len(samples) * 0.95) - 1]


def main() -> None:
    """按规模逐个跑基准并打印结果。"""
    sizes = [int(sys.argv[1])] if len(sys.argv) > 1 else [100, 600, 1200]
    scope = ScopeFilter(mode="group_first", current_scope="qq:111", sender_id="alice")
    print(f"{'规模':>6} {'P50(ms)':>9} {'P95(ms)':>9}")
    for size in sizes:
        store, query = build_store(size)
        p50, p95 = measure(store, query, scope)
        print(f"{size:>6} {p50:>9.1f} {p95:>9.1f}")


if __name__ == "__main__":
    main()
