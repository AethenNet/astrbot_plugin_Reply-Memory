"""用户屏蔽记录的持久化存储。

全局屏蔽（scope=``all``）对所有会话生效；群级屏蔽（scope=``group``）只对
对应群生效。记录以 JSON 保存，时间为 0 表示永久，否则为到期时间戳。
"""

from __future__ import annotations

from astrbot.api import logger

import copy
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any


STORE_VERSION = 1
USER_ID_RE = re.compile(r"^[A-Za-z0-9]+$")

# 默认全局屏蔽用户与原因（与插件配置里的默认黑名单保持一致）
DEFAULT_REASONS: dict[str, str] = {
    "2854196310": "Q群管家",
    "2854197266": "AL_1S",
    "3889239358": "chaob",
    "3889001246": "幽幽子",
}


def default_reason(user_id: str) -> str:
    return DEFAULT_REASONS.get(str(user_id), "配置黑名单")


def parse_until(text: str) -> float | None:
    """解析屏蔽时长，返回绝对到期时间；0 表示永久；非法返回 None。

    支持：``0``/``永久``、``1d2h30m``、``30m``、``2h``、裸数字（按分钟）。
    """
    raw = str(text or "").strip().lower()
    if raw in {"0", "永久", "永久屏蔽", "forever", "perm", "none", ""}:
        return 0.0
    match = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", raw)
    if match and any(match.groups()):
        days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
        total = days * 86400 + hours * 3600 + minutes * 60 + seconds
        return time.time() + total if total > 0 else 0.0
    if raw.isdigit():
        return time.time() + int(raw) * 60
    return None


def format_remaining(until: float) -> str:
    """把到期时间格式化为「永久 / 剩余 X」的易读文本。"""
    if not until:
        return "永久"
    try:
        remaining = float(until) - time.time()
    except (TypeError, ValueError):
        return "永久"
    if remaining <= 0:
        return "已过期"
    total = int(remaining)
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    if minutes and len(parts) < 2:
        parts.append(f"{minutes}分钟")
    if not parts:
        parts.append(f"{seconds}秒")
    return "剩余" + "".join(parts)


def _coerce_float(value: Any, default: float = 0.0) -> tuple[float, bool]:
    """返回 ``(有限浮点数, 是否合法)``；非法/非有限值回退默认值（F19）。"""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default, False
    if not math.isfinite(number):
        return default, False
    return number, True


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把任意 JSON 值转成有限浮点数；非法值回退默认值（F19）。"""
    return _coerce_float(value, default)[0]


def _as_text(value: Any) -> str:
    """把 JSON 标量转成清理后的文本；对象/数组等结构返回空串。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


class BlockStore:
    """用户屏蔽记录的读写与查询。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._global: list[dict[str, Any]] = []
        self._groups: dict[str, list[dict[str, Any]]] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """从 JSON 文件恢复屏蔽记录；任何坏数据只告警、不阻塞插件加载。"""
        self._global = []
        self._groups = {}
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[ActiveReply] 屏蔽记录文件读取失败，按空配置启动: %s", exc)
            return
        if not isinstance(data, dict):
            logger.warning("[ActiveReply] 屏蔽记录文件顶层不是对象，按空配置启动")
            return
        try:
            raw_global = data.get("global")
            if isinstance(raw_global, list):
                self._global = [
                    item for item in map(self._normalize, raw_global) if item
                ]
            raw_groups = data.get("groups")
            if isinstance(raw_groups, dict):
                for group_id, entries in raw_groups.items():
                    if not isinstance(entries, list):
                        logger.warning(
                            "[ActiveReply] 屏蔽记录群 %s 的条目不是数组，已跳过",
                            str(group_id)[:64],
                        )
                        continue
                    # 注意：这里不把外层 key 注入条目的 group_id。既有测试
                    # ``test_bad_records_are_filtered`` 约定「scope=group 但缺
                    # group_id 的条目」视为坏数据丢弃，保持旧行为兼容。
                    cleaned = [item for item in map(self._normalize, entries) if item]
                    if cleaned:
                        self._groups[str(group_id)] = cleaned
        except Exception as exc:  # 兜底：坏文件结构也不能让插件加载失败
            logger.warning("[ActiveReply] 屏蔽记录载入异常，按空配置启动: %s", exc)
            self._global = []
            self._groups = {}

    @staticmethod
    def _normalize(item: Any) -> dict[str, Any] | None:
        """清洗单条屏蔽记录；结构或字段非法的条目返回 None（F19）。"""
        if not isinstance(item, dict):
            logger.warning("[ActiveReply] 屏蔽记录不是对象，已跳过")
            return None
        user_id = _as_text(item.get("user_id"))
        if not USER_ID_RE.fullmatch(user_id):
            logger.warning("[ActiveReply] 屏蔽记录用户 ID 非法，已跳过")
            return None
        scope = _as_text(item.get("scope")).lower() or "all"
        if scope not in {"all", "group"}:
            scope = "all"
        group_id = _as_text(item.get("group_id"))
        if scope == "group" and not group_id:
            logger.warning(
                "[ActiveReply] 群级屏蔽记录缺少群号，已跳过 | user=%s", user_id
            )
            return None
        # created_at/until 必须容错：手改或旧版本的坏 JSON 曾让 float() 直接
        # 抛错，进而在插件 __init__ 里把整个插件加载带崩。
        raw_until = item.get("until")
        until, until_ok = _coerce_float(raw_until, 0.0)
        if not until_ok and raw_until not in (None, ""):
            logger.warning(
                "[ActiveReply] 屏蔽记录 until 非法，按永久处理 | user=%s", user_id
            )
        raw_created = item.get("created_at")
        created_at, created_ok = _coerce_float(raw_created, 0.0)
        if not created_ok and raw_created not in (None, ""):
            logger.warning(
                "[ActiveReply] 屏蔽记录 created_at 非法，按当前时间处理 | user=%s",
                user_id,
            )
        if created_at <= 0:
            created_at = time.time()
        return {
            "user_id": user_id,
            "reason": _as_text(item.get("reason")) or default_reason(user_id),
            "scope": scope,
            "group_id": group_id,
            "until": until,
            "created_at": created_at,
            "source": _as_text(item.get("source")) or "page",
        }

    def _save(self) -> None:
        payload = {
            "version": STORE_VERSION,
            "global": self._global,
            "groups": self._groups,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)

    @staticmethod
    def _active(entry: dict[str, Any]) -> bool:
        until = _safe_float(entry.get("until"), 0.0)
        return until == 0 or until > time.time()

    # ------------------------------------------------------------------ #
    # 配置同步
    # ------------------------------------------------------------------ #
    def sync_from_config(self, user_ids: tuple[str, ...] | list[str]) -> None:
        """把配置里的黑名单同步为全局屏蔽记录。

        只回收 ``source=config`` 的旧配置条目；页面与指令记录带原因、时长
        和来源，不因配置缺失而被删除。写盘失败时恢复内存并抛出。
        """
        # 配置里的重复 ID 只处理一次；只把“仍有效”的记录当作已存在，
        # 避免过期页面临时屏蔽占位导致配置永久条目补不进来。
        wanted = list(
            dict.fromkeys(str(uid).strip() for uid in user_ids if str(uid).strip())
        )
        wanted_set = set(wanted)
        with self._lock:
            previous = copy.deepcopy((self._global, self._groups))
            existing = {
                entry["user_id"] for entry in self._global if self._active(entry)
            }
            changed = False
            # 配置镜像只增不删会让 /remove-all 或全局设置清空无效：
            # 只回收 source=config 的条目，页面/指令记录保留。
            kept = [
                entry
                for entry in self._global
                if not (
                    entry.get("source") == "config"
                    and entry.get("user_id") not in wanted_set
                )
            ]
            if len(kept) != len(self._global):
                self._global = kept
                existing = {entry["user_id"] for entry in self._global}
                changed = True
            for user_id in wanted:
                if user_id not in existing:
                    # 同一 ID 的过期临时记录让位给配置源永久记录。
                    self._global = [
                        entry
                        for entry in self._global
                        if not (
                            entry.get("user_id") == user_id
                            and not self._active(entry)
                        )
                    ]
                    self._global.append(
                        {
                            "user_id": user_id,
                            "reason": default_reason(user_id),
                            "scope": "all",
                            "group_id": "",
                            "until": 0.0,
                            "created_at": time.time(),
                            "source": "config",
                        }
                    )
                    changed = True
            if changed:
                try:
                    self._save()
                except Exception:
                    self._global, self._groups = previous
                    raise

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def global_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._global if self._active(item)]

    def group_entries(self, group_id: str) -> list[dict[str, Any]]:
        group_id = str(group_id or "").strip()
        if not group_id:
            return []
        with self._lock:
            entries = self._groups.get(group_id, [])
            return [dict(item) for item in entries if self._active(item)]

    def effective_entries(self, group_id: str) -> list[dict[str, Any]]:
        entries = self.global_entries()
        entries.extend(self.group_entries(group_id))
        return entries

    def is_blocked(self, user_id: str, group_id: str) -> dict[str, Any] | None:
        user_id = str(user_id or "").strip()
        if not user_id:
            return None
        for entry in self.effective_entries(group_id):
            if entry["user_id"] == user_id:
                return entry
        return None

    # ------------------------------------------------------------------ #
    # 写入 / 删除
    # ------------------------------------------------------------------ #
    def add(
        self,
        user_id: str,
        *,
        reason: str = "",
        scope: str = "all",
        until: float = 0.0,
        group_id: str = "",
        source: str = "page",
    ) -> dict[str, Any]:
        user_id = str(user_id or "").strip()
        if not USER_ID_RE.fullmatch(user_id):
            raise ValueError("用户 ID 只能包含英文字母和数字")
        scope = "group" if str(scope).lower() == "group" else "all"
        group_id = str(group_id or "").strip()
        if scope == "group" and not group_id:
            raise ValueError("群级屏蔽缺少群号")
        entry = {
            "user_id": user_id,
            "reason": str(reason or "").strip() or default_reason(user_id),
            "scope": scope,
            "group_id": group_id if scope == "group" else "",
            "until": _safe_float(until, 0.0),
            "created_at": time.time(),
            "source": str(source or "page"),
        }
        with self._lock:
            previous = copy.deepcopy((self._global, self._groups))
            try:
                if scope == "all":
                    self._global = [e for e in self._global if e["user_id"] != user_id]
                    self._global.append(entry)
                else:
                    entries = [
                        e for e in self._groups.get(group_id, [])
                        if e["user_id"] != user_id
                    ]
                    entries.append(entry)
                    self._groups[group_id] = entries
                self._save()
            except Exception:
                self._global, self._groups = previous
                raise
        return dict(entry)

    def remove(self, user_id: str, *, scope: str = "all", group_id: str = "") -> bool:
        user_id = str(user_id or "").strip()
        scope = "group" if str(scope).lower() == "group" else "all"
        group_id = str(group_id or "").strip()
        removed = False
        with self._lock:
            previous = copy.deepcopy((self._global, self._groups))
            try:
                if scope == "all":
                    before = len(self._global)
                    self._global = [e for e in self._global if e["user_id"] != user_id]
                    removed = len(self._global) != before
                else:
                    entries = self._groups.get(group_id, [])
                    new_entries = [e for e in entries if e["user_id"] != user_id]
                    removed = len(new_entries) != len(entries)
                    if new_entries:
                        self._groups[group_id] = new_entries
                    else:
                        self._groups.pop(group_id, None)
                if removed:
                    self._save()
            except Exception:
                self._global, self._groups = previous
                raise
        return removed

    def remove_all_for_user(self, user_id: str) -> int:
        """删除某用户的全部记录（全局 + 群级），返回删除数量。"""
        user_id = str(user_id or "").strip()
        removed = 0
        with self._lock:
            previous = copy.deepcopy((self._global, self._groups))
            try:
                before = len(self._global)
                self._global = [e for e in self._global if e["user_id"] != user_id]
                removed += before - len(self._global)
                for group_id in list(self._groups):
                    entries = self._groups[group_id]
                    new_entries = [e for e in entries if e["user_id"] != user_id]
                    if len(new_entries) != len(entries):
                        removed += len(entries) - len(new_entries)
                        if new_entries:
                            self._groups[group_id] = new_entries
                        else:
                            self._groups.pop(group_id, None)
                if removed:
                    self._save()
            except Exception:
                self._global, self._groups = previous
                raise
        return removed
