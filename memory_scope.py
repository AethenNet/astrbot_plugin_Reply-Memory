"""记忆可见性规则（唯一真相来源）。

一条记忆对当前发言人可见，当且仅当：

* 它属于当前会话作用域（``scope == 当前会话``），或
* 当前发言人是这条记忆的参与者（``memory_roles`` 命中）。

全局作用域（``group_scope == ''``）不再是“所有群都能读”，而是“随参与者走”：
谁参与过，谁才能在别的会话里召回。私聊作用域默认不跨到群聊。

本模块只依赖标准库，``clause()`` 生成的 SQL 供存储层使用，``is_visible()``
供运行时对象判断，两条路径共用同一条规则，避免“召回一套、工具一套”。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

GLOBAL_SCOPE = ""
PRIVATE_MARK = ":private:"

MODE_AUTO = "group_first"
MODE_GLOBAL = "global"
MODE_GROUP = "group"
VALID_MODES = frozenset({MODE_AUTO, MODE_GLOBAL, MODE_GROUP})


def normalize_mode(value: Any) -> str:
    """把配置里的作用域模式收敛成三种合法值，未知值按“自动”处理。"""
    mode = str(value or "").strip().lower()
    return mode if mode in VALID_MODES else MODE_AUTO


def is_private_scope(scope: str) -> bool:
    """判断一个作用域是否是私聊（``platform:private:user``）。"""
    return PRIVATE_MARK in str(scope or "").lower()


def _participant_sql(alias: str) -> str:
    """参与者判定子查询；参数由调用方按顺序绑定。"""
    return (
        f"EXISTS (SELECT 1 FROM memory_roles AS r "
        f"WHERE r.memory_id = {alias}.id AND r.role_id = ?)"
    )


@dataclass(frozen=True)
class ScopeFilter:
    """一次记忆检索的可见性条件。

    Args:
        mode: ``group_first``（自动：本会话 + 本人参与）、``global``（仅全局层
            且本人参与）、``group``（仅本会话）。
        current_scope: 当前会话的规范作用域；空串表示没有会话作用域。
        sender_id: 当前发言人 ID；空串表示无法判断参与者。
        allow_private: 是否允许把私聊作用域的记忆带到其它会话，默认关闭。
        personal_enabled: 是否允许“本人参与”这条通道；关闭后只保留本会话。
    """

    mode: str = MODE_AUTO
    current_scope: str = ""
    sender_id: str = ""
    allow_private: bool = False
    personal_enabled: bool = True

    def __post_init__(self) -> None:
        """规范化入参，保证外部随意传值也不会生成奇怪的条件。"""
        object.__setattr__(self, "mode", normalize_mode(self.mode))
        object.__setattr__(self, "current_scope", str(self.current_scope or "").strip())
        object.__setattr__(self, "sender_id", str(self.sender_id or "").strip())

    @property
    def uses_personal(self) -> bool:
        """当前条件是否包含“本人参与”通道。"""
        return bool(self.personal_enabled and self.sender_id)

    def clause(self, alias: str = "m") -> tuple[str, list[Any]]:
        """生成 SQL 条件与绑定参数；无法命中任何记忆时返回 ``("0", [])``。

        返回的条件恒非空，调用方可以直接 ``AND`` 进自己的 WHERE，不需要再判断。
        """
        alias = str(alias or "m")
        params: list[Any] = []
        if self.mode == MODE_GROUP:
            if not self.current_scope:
                return "0", []
            return f"{alias}.group_scope = ?", [self.current_scope]

        if self.mode == MODE_GLOBAL:
            if not self.uses_personal:
                return "0", []
            params.append(self.sender_id)
            return (
                f"({alias}.group_scope = '' AND {_participant_sql(alias)})",
                params,
            )

        branches: list[str] = []
        if self.current_scope:
            branches.append(f"{alias}.group_scope = ?")
            params.append(self.current_scope)
        if self.uses_personal:
            participant = _participant_sql(alias)
            if not self.allow_private:
                # 私聊记忆只在它自己的会话里可见：当前会话命中已由上一个分支处理。
                participant = (
                    f"({participant} AND instr(lower({alias}.group_scope), "
                    f"'{PRIVATE_MARK}') = 0)"
                )
            branches.append(f"({participant})")
            params.append(self.sender_id)
        if not branches:
            return "0", []
        return "(" + " OR ".join(branches) + ")", params

    def is_visible(self, record: Mapping[str, Any]) -> bool:
        """按同一条规则判断一条已取出的记忆是否可见。"""
        scope = str(record.get("group_scope") or "").strip()
        if scope and scope == self.current_scope:
            return True
        if not self.uses_personal:
            return False
        if self.mode == MODE_GROUP:
            return False
        if self.mode == MODE_GLOBAL and scope:
            return False
        if scope and not self.allow_private and is_private_scope(scope):
            return False
        roles = record.get("related_role_ids") or record.get("role_ids") or ()
        try:
            if self.sender_id in {str(role) for role in roles}:
                return True
        except TypeError:
            pass
        return str(record.get("sender_id") or "").strip() == self.sender_id

    def describe(self) -> str:
        """生成日志用的简短描述，避免把完整参数打进日志。"""
        scope = self.current_scope or "(无会话作用域)"
        personal = "开" if self.uses_personal else "关"
        private = "允许" if self.allow_private else "禁止"
        return f"mode={self.mode} scope={scope} personal={personal} private={private}"


__all__ = [
    "GLOBAL_SCOPE",
    "MODE_AUTO",
    "MODE_GLOBAL",
    "MODE_GROUP",
    "ScopeFilter",
    "is_private_scope",
    "normalize_mode",
]
