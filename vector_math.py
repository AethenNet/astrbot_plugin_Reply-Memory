"""批量余弦相似度。

记忆向量在库里有两种表示：新数据是 ``float32`` BLOB，旧数据是 JSON 文本。
numpy 可用时把一批候选拼成矩阵做一次矩阵乘法；不可用时退回纯 Python 循环，
保证没有 numpy 的环境行为不变。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

try:  # numpy 是 AstrBot/FAISS 的既有依赖，但没有它也要能跑
    import numpy as _np
except Exception:  # pragma: no cover - 取决于运行环境
    _np = None


def numpy_available() -> bool:
    """当前环境是否可用 numpy 加速。"""
    return _np is not None


def _to_array(value: Any) -> Any:
    """把 BLOB / JSON / 列表统一成 float32 向量；无法解析时返回 None。"""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if not raw or len(raw) % 4 != 0:
            return None
        return _np.frombuffer(raw, dtype=_np.float32)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list) or not parsed:
            return None
        return _np.asarray(parsed, dtype=_np.float32)
    try:
        return _np.asarray(value, dtype=_np.float32)
    except (TypeError, ValueError):
        return None


def cosine_scores(query: Sequence[float], vectors: Sequence[Any]) -> list[float]:
    """计算查询向量与一批记忆向量的余弦相似度。

    Args:
        query: 查询向量。
        vectors: 每项是 ``float32`` BLOB、JSON 文本或数值列表；顺序与返回一致。

    Returns:
        与 ``vectors`` 等长的分数列表；向量非法或维度不一致记 ``-1.0``。
        返回 ``-1.0`` 的条目调用方应视为不可比较，而不是“完全无关”。
    """
    if not query or not vectors:
        return []
    if _np is None:
        return [_python_cosine(query, item) for item in vectors]
    query_array = _np.asarray(query, dtype=_np.float32)
    if query_array.size == 0:
        return [-1.0] * len(vectors)
    query_norm = float(_np.linalg.norm(query_array))
    if query_norm <= 0:
        return [-1.0] * len(vectors)
    prepared = [_to_array(item) for item in vectors]
    dim = int(query_array.size)
    usable = [
        i for i, item in enumerate(prepared) if item is not None and item.size == dim
    ]
    scores = [-1.0] * len(prepared)
    if not usable:
        return scores
    matrix = _np.stack([prepared[i] for i in usable])
    norms = _np.linalg.norm(matrix, axis=1)
    dots = matrix @ query_array
    with _np.errstate(divide="ignore", invalid="ignore"):
        values = dots / (norms * query_norm)
    for offset, index in enumerate(usable):
        norm = float(norms[offset])
        scores[index] = float(values[offset]) if norm > 0 else -1.0
    return scores


def _python_cosine(query: Sequence[float], value: Any) -> float:
    """无 numpy 时的单条余弦；与 numpy 路径保持一致的口径。"""
    vector = value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            from array import array

            values = array("f")
            values.frombytes(bytes(value))
            vector = list(values)
        except (TypeError, ValueError):
            return -1.0
    elif isinstance(value, str):
        try:
            vector = [float(item) for item in json.loads(value)]
        except (TypeError, ValueError, json.JSONDecodeError):
            return -1.0
    try:
        left = [float(item) for item in query]
        right = [float(item) for item in (vector or [])]
    except (TypeError, ValueError):
        return -1.0
    if not left or len(left) != len(right):
        return -1.0
    dot = norm_left = norm_right = 0.0
    for a, b in zip(left, right, strict=False):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left <= 0 or norm_right <= 0:
        return -1.0
    return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))


__all__ = ["cosine_scores", "numpy_available"]
