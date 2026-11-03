"""插件初始化接线检查。

`main.py` 在模块顶层导入 AstrBot，测试环境无法直接 import；但「只导入、
忘记在 __init__ 里赋值」这类错误会让运行时在 except 里静默失败（例如
`self._recall_cache` 缺失会导致自动回忆整体失效）。这里用 AST 兜住这一类。
"""

from __future__ import annotations

import ast
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"

# 运行时会被 self.xxx 读写、必须有初始化的属性
REQUIRED_ATTRS = {
    "_recall_cache",
    "_memory_write_lock",
    "_background_tasks",
    "_maintenance_task",
    "_last_consolidation_at",
    "state",
    "memory_store",
    "group_store",
    "block_store",
}


def _self_assignments(tree: ast.AST) -> set[str]:
    """收集所有 ``self.xxx = ...`` / ``self.xxx: T = ...`` 的属性名。"""
    assigned: set[str] = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned.add(target.attr)
    return assigned


def test_required_attributes_are_initialized() -> None:
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    assigned = _self_assignments(tree)
    missing = sorted(REQUIRED_ATTRS - assigned)
    assert not missing, f"main.py 没有在 __init__ 里赋值: {missing}"


def test_imported_names_are_used() -> None:
    """新增模块必须真的被用起来，避免「导入但未接线」。"""
    source = MAIN.read_text(encoding="utf-8")
    for name in ("RecallCache", "decide_similar_action", "ScopeFilter"):
        assert name in source
        # 至少有除 import 行以外的使用
        uses = [
            line
            for line in source.splitlines()
            if name in line and not line.lstrip().startswith(("import ", "from "))
        ]
        assert uses, f"{name} 只被导入、没有被使用"
