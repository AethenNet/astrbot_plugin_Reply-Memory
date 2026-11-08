"""每个群的独立配置存储。

覆盖项只包含"群内行为"相关的字段（主动回复 / 群聊上下文 / 群聊回复增强 / 记忆 RAG）；
联网搜索、超时等属于全局服务配置，不在群级覆盖范围内。

数据保存在 ``data/plugin_data/astrbot_plugin_active_reply/groups.json``，
与 AstrBot 的插件配置分开，避免 Schema 变更时丢失群配置。
"""

from __future__ import annotations

from astrbot.api import logger

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

from .config_model import GROUP_OVERRIDE_FIELDS, get_by_path, set_by_path


STORE_VERSION = 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把任意 JSON 值转成有限浮点数；非法值回退默认值（F20）。"""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _as_text(value: Any) -> str:
    """把 JSON 标量转成清理后的文本；对象/数组等结构返回空串（F20）。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _nested_config(config: dict[str, Any]) -> dict[str, Any]:
    """把扁平点号键（page_api 的提交格式）归一成嵌套结构。"""
    nested: dict[str, Any] = {}
    for key, value in (config or {}).items():
        if isinstance(key, str) and "." in key:
            set_by_path(nested, key, value)
        else:
            nested[key] = value
    return nested


class GroupStore:
    """将各群独立配置以 JSON 文件持久化，并提供全局值合并视图。

    磁盘记录只保存明确允许群级覆盖的字段，不复制全局配置；读取面板数据时
    再将覆盖项与最新全局快照合并。这样全局配置调整后，未覆盖字段会自动
    跟随新值。写操作由可重入锁保护，并通过同目录临时文件替换避免写入中断
    留下半截 JSON。
    """

    def __init__(self, path: Path | str) -> None:
        """创建群配置存储并从指定 JSON 文件载入已有记录。

        Args:
            path: 群级覆盖配置文件路径。
        """
        # 初始化线程安全的内存快照，并立即尝试从 JSON 文件加载现有配置。
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # 读写
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """从磁盘恢复群记录，并逐条清洗坏数据（F20）。

        文件缺失、读取失败或 JSON 语法损坏时初始化为空配置，不因本地群配置
        文件问题阻断插件启动；群条目必须是 dict，``config`` 必须是 dict，
        ``updated_at`` 必须是数字，否则忽略坏值并告警。
        """
        # 读取旧配置；文件不存在或损坏时从空映射启动，不阻塞插件启动。
        self._data = {}
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("[ActiveReply] 群配置读取失败，按空配置启动: %s", exc)
            return
        groups = payload.get("groups") if isinstance(payload, dict) else None
        if not isinstance(groups, dict):
            if payload is not None:
                logger.warning("[ActiveReply] groups.json 的 groups 不是对象，已忽略")
            return
        cleaned: dict[str, dict[str, Any]] = {}
        for group_id, entry in groups.items():
            normalized = self._normalize_entry(entry, str(group_id))
            if normalized is not None:
                cleaned[str(group_id)] = normalized
        self._data = cleaned

    @staticmethod
    def _normalize_entry(entry: Any, group_id: str) -> dict[str, Any] | None:
        """清洗单个群条目；返回 None 表示丢弃（F20）。"""
        if not isinstance(entry, dict):
            logger.warning(
                "[ActiveReply] 群配置条目 %s 不是对象，已忽略", group_id[:64]
            )
            return None
        config = entry.get("config")
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            logger.warning(
                "[ActiveReply] 群 %s 的 config 不是对象，已按空覆盖处理", group_id[:64]
            )
            config = {}
        # 旧版/页面提交可能使用扁平点号键，载入时统一成嵌套结构。
        config = _nested_config(config)
        # 保留未知顶层键，避免升级/降级时静默丢失未来字段。
        normalized = dict(entry)
        normalized["independent"] = bool(entry.get("independent"))
        normalized["config"] = config
        normalized["note"] = _as_text(entry.get("note"))
        updated_at = _safe_float(entry.get("updated_at"), 0.0)
        if "updated_at" in entry and entry.get("updated_at") not in (None, ""):
            if updated_at <= 0:
                logger.warning(
                    "[ActiveReply] 群 %s 的 updated_at 非法，已回退 0", group_id[:64]
                )
        normalized["updated_at"] = updated_at
        return normalized

    def _save(self) -> None:
        """将当前群配置快照以临时文件替换方式持久化。"""
        # 先写同目录临时文件再替换目标，降低进程中断导致 JSON 被截断的风险。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": STORE_VERSION, "groups": self._data}
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(self.path)

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def get_overrides(self, group_id: str) -> dict[str, Any]:
        """返回该群生效的覆盖值（嵌套 dict），坏值一律按空 dict 处理（F20）。"""
        with self._lock:
            entry = self._data.get(str(group_id))
            if not entry or not entry.get("independent"):
                return {}
            overrides = entry.get("config")
            if not isinstance(overrides, dict):
                if overrides is not None:
                    logger.warning(
                        "[ActiveReply] 群 %s 的覆盖值不是对象，已按空处理",
                        str(group_id)[:64],
                    )
                return {}
            return copy.deepcopy(overrides)

    def get_meta(self, group_id: str) -> dict[str, Any]:
        """获取面板展示所需的群状态、更新时间和备注。"""
        # 返回面板需要的状态、最近更新时间与可读备注，不暴露内部配置对象。
        with self._lock:
            entry = self._data.get(str(group_id)) or {}
            if not isinstance(entry, dict):
                return {"independent": False, "updated_at": 0.0, "note": ""}
            return {
                "independent": bool(entry.get("independent")),
                "updated_at": _safe_float(entry.get("updated_at"), 0.0),
                "note": _as_text(entry.get("note")),
            }

    def known_groups(self) -> list[str]:
        """列出本地存储中出现过的全部群号。"""
        # 返回曾保存过元数据或覆盖值的群号，保证面板可重新找到这些群。
        return sorted(self._data.keys())

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def save_group_config(
        self,
        group_id: str,
        values: dict[str, Any],
        *,
        independent: bool,
        note: str = "",
    ) -> dict[str, Any]:
        """一次提交群配置（覆盖值 + 独立状态 + 备注），失败回滚内存。"""
        group_id = str(group_id or "").strip()
        if not group_id:
            raise ValueError("group_id 不能为空")
        with self._lock:
            previous = copy.deepcopy(self._data)
            try:
                if independent:
                    self._data[group_id] = {
                        "independent": True,
                        "config": _nested_config(values or {}),
                        "note": str(note or "").strip(),
                        "updated_at": time.time(),
                    }
                else:
                    entry = self._data.setdefault(group_id, {})
                    entry["independent"] = False
                    if note:
                        entry["note"] = str(note).strip()
                    entry["updated_at"] = time.time()
                    # 跟随全局也可能只是保存群备注；有 note 时不能删记录。
                    if not entry.get("config") and not entry.get("note"):
                        self._data.pop(group_id, None)
                self._save()
            except Exception:
                self._data = previous
                raise
        return _nested_config(values or {}) if independent else {}

    def remove(self, group_id: str) -> bool:
        """删除指定群的记录，并返回是否实际删除了数据。"""
        # 删除群的全部保存数据；写盘失败时恢复内存快照，避免本次运行丢配置。
        with self._lock:
            previous = copy.deepcopy(self._data)
            removed = self._data.pop(str(group_id), None) is not None
            if removed:
                try:
                    self._save()
                except Exception:
                    self._data = previous
                    raise
            return removed

    # ------------------------------------------------------------------ #
    # 面板展示
    # ------------------------------------------------------------------ #
    def effective_values(
        self, group_id: str, global_values: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """为指定群构造面板使用的完整字段快照。

        未启用独立模式时返回全局快照的浅拷贝；独立模式下只遍历白名单字段，
        优先取群配置中非 ``None`` 的值，缺失字段继续回落到全局配置。返回值
        使用扁平点号路径作为键，便于前端按字段元数据生成表单。

        Returns:
            ``(字段路径到有效值的映射, 是否处于独立配置模式)``。
        """
        with self._lock:
            entry = self._data.get(str(group_id)) or {}
            independent = (
                bool(entry.get("independent")) if isinstance(entry, dict) else False
            )
            if not independent:
                return dict(global_values), False
            overrides = self.get_overrides(group_id)
            values: dict[str, Any] = {}
            for spec in GROUP_OVERRIDE_FIELDS:
                value = get_by_path(overrides, spec.path)
                values[spec.path] = (
                    value if value is not None else global_values.get(spec.path)
                )
            return values, True
