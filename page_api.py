"""插件控制台页面的后端接口与数据整形层。

各 handler 通过 ``web_compat`` 读取请求并返回统一 JSON，不直接操作 HTTP
响应对象。全局设置依据 ``_conf_schema.json`` 过滤并转换提交字段；群设置只允许
配置模型声明的覆盖字段；记忆管理接口则将 SQLite 存储结果加工成控制台需要的
分页、统计、自检和清理数据。前端由 ``views/console`` 下的脚本通过 AstrBot
插件视图 bridge 调用这些处理器。
"""

from __future__ import annotations

import asyncio
import copy
import datetime
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .block_store import default_reason, format_remaining, parse_until
from .config_model import (
    DEFAULT_GROUP_SUMMARY_PROMPT,
    DEFAULT_PRIVATE_SUMMARY_PROMPT,
    GROUP_FIELD_BY_PATH,
    GROUP_OVERRIDE_FIELDS,
    build_global_snapshot,
    build_schema_patch,
    coerce_group_value,
    deep_merge,
    get_by_path,
)
from .constants import IMPORTANCE_DEFAULT, IMPORTANCE_MAX
from .memory_store import compute_effective_importance
from .web_compat import JsonBodyError, fail, header_get, ok, query_get, read_json

# 屏蔽接口的群号白名单：纯数字群号，或「平台:群号」形式；长度另设上限。
_GROUP_ID_RE = re.compile(r"^(?:[A-Za-z0-9_.\-]{1,32}:)?\d{1,20}$")
_GROUP_ID_MAX_LEN = 64

# 记忆列表接口只回传轻量元数据：证据类大字段留给详情接口，避免一页
# 200 条各带数 KB 正文把页面拖慢（F40）。
_RAG_LIST_BULK_KEYS = ("source_messages", "quotes", "evidence")
_RAG_LIST_METADATA_MAX_BYTES = 1024
_RAG_LIST_SCALAR_MAX_CHARS = 200

# r3.3 开发者模式：密钥只保存加盐哈希，令牌只存在进程内存、30 分钟后过期。
_DEV_KEY_SALT = "ActiveReply::devkey::v1::"
# 内置密钥同样只存加盐 SHA-256；建议用环境变量 ACTIVE_REPLY_DEV_KEY 覆盖。
_DEV_KEY_SHA256 = "c370539e9b712e62c73e78847ef2ff0e7cc1d2bc204f15f058fe9d7899de6a9e"
_DEV_KEY_ENV = "ACTIVE_REPLY_DEV_KEY"
_DEV_TOKEN_TTL_SEC = 1800
_DEV_MAX_FAILS = 5
_DEV_FAIL_WINDOW_SEC = 600
_DEV_INVALID_MSG = "开发者密钥无效或已过期，请重新进入开发者模式"


def _dev_key_sha256() -> str:
    """解析当前生效的开发者密钥哈希：优先环境变量，其次内置哈希。

    环境变量 ``ACTIVE_REPLY_DEV_KEY`` 存明文密钥，这里只在内存中计算
    ``sha256(salt + env)``，不打印明文；未设置环境变量时使用内置哈希，
    并在模块加载时告警一次（不逐请求刷）。
    """
    env_key = os.environ.get(_DEV_KEY_ENV, "")
    if env_key:
        return hashlib.sha256((_DEV_KEY_SALT + env_key).encode("utf-8")).hexdigest()
    logger.warning(
        "[ActiveReply] 未设置环境变量 %s，使用内置开发者密钥；建议部署时改用环境变量。",
        _DEV_KEY_ENV,
    )
    return _DEV_KEY_SHA256


# 模块加载时解析一次，供 dev_login 做常量时间比较。
_DEV_KEY_EFFECTIVE_SHA256 = _dev_key_sha256()


async def _read_payload() -> tuple[dict[str, Any], str]:
    """读取 JSON 请求体；解析失败时返回错误文案而不是静默当作空对象。

    Returns:
        ``(payload, error)``；``error`` 非空表示请求体语法错误，调用方应
        直接返回 ``fail(error)``（F22）。
    """
    try:
        return await read_json(), ""
    except JsonBodyError as exc:
        return {}, str(exc)


class PageApi:
    """将控制台路由操作映射到插件配置、存储和运行状态的适配器。

    实例不拥有独立业务状态，而是引用插件对象并委托给其配置模型、群配置仓库、
    记忆存储和运行态。响应使用 ``ok``/``fail`` 统一包装，避免每个端点自行
    决定错误结构或重复实现配置校验。
    """

    def __init__(self, plugin: Any) -> None:
        """保存插件实例，以便各页面处理器访问配置、存储和运行状态。"""
        # 保留插件实例，以共享配置、存储、提供商列表与运行时状态。
        self.plugin = plugin
        # r3.3：开发者令牌与失败时间戳都只放进程内存，插件重载后自然失效。
        self._dev_tokens: dict[str, float] = {}
        self._dev_fail_times: list[float] = []

    # ------------------------------------------------------------------ #
    # 展示与校验辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _display_data_dir(path: Any) -> str:
        """只回传数据目录名，避免向控制台暴露服务器绝对路径（F43）。"""
        try:
            name = Path(path).name
        except (TypeError, ValueError):
            return ""
        return name or "."

    @staticmethod
    def _safe_until(value: Any) -> float:
        """把屏蔽记录的 ``until`` 安全转成浮点，坏值按永久处理（F24）。"""
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _valid_group_id(value: Any) -> str:
        """校验屏蔽接口的 ``group_id``，非法时返回空字符串（F24）。"""
        text = str(value or "").strip()
        if not text or len(text) > _GROUP_ID_MAX_LEN:
            return ""
        if not _GROUP_ID_RE.fullmatch(text):
            return ""
        return text

    @classmethod
    def _compact_list_metadata(cls, meta: Any) -> dict[str, Any]:
        """裁剪列表接口的 ``extra_metadata``，大字段只在详情接口返回（F40）。

        ``source_messages`` / ``quotes`` / ``evidence`` 往往是完整的消息正文，
        列表页并不展示；超过序列化上限后再退化为只保留短标量字段。
        """
        if not isinstance(meta, dict):
            return {}
        compact = {
            key: value
            for key, value in meta.items()
            if key not in _RAG_LIST_BULK_KEYS
        }
        try:
            size = len(json.dumps(compact, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            return {}
        if size <= _RAG_LIST_METADATA_MAX_BYTES:
            return compact
        trimmed: dict[str, Any] = {}
        for key, value in compact.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                if len(str(value)) <= _RAG_LIST_SCALAR_MAX_CHARS:
                    trimmed[key] = value
        return trimmed

    @staticmethod
    def _plugin_version() -> str:
        """复用 main.PLUGIN_VERSION，避免两处 metadata 解析漂移。"""
        try:
            from .main import PLUGIN_VERSION

            return PLUGIN_VERSION
        except Exception:
            return "unknown"

    def _embedding_options(self) -> list[dict[str, str]]:
        """列出当前可用的 embedding 提供商，方便用户复制 ID / 模型名。"""
        options: list[dict[str, str]] = []
        try:
            providers = self.plugin.context.get_all_embedding_providers() or []
        except Exception:
            return options
        for provider in providers:
            provider_id, model = self.plugin._provider_meta_label(provider)
            if provider_id or model:
                options.append({"id": provider_id, "model": model})
        return options

    # ------------------------------------------------------------------ #
    # 全局配置
    # ------------------------------------------------------------------ #
    def _schema_path(self) -> Path:
        """返回插件目录中的 JSON 配置 schema 文件路径。"""
        # schema 与插件源码同目录，避免依赖 AstrBot 当前工作目录。
        return Path(__file__).with_name("_conf_schema.json")

    def _load_schema(self) -> dict[str, Any]:
        """读取页面表单 schema；失败时记录错误并返回空对象。"""
        # 读取全局配置表单定义；失败时记录日志并返回空 schema。
        try:
            return json.loads(self._schema_path().read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[ActiveReply] 读取配置 schema 失败: {exc}")
            return {}

    def _inject_memory_tuning_values(self, values: dict[str, Any]) -> None:
        """把解析后的记忆调优值填到新路径 runtime.memory_tuning（r3.3n）。

        这些配置从 memory_rag 迁到「调用配置」，旧配置仍保留在原路径；
        这里把当前生效值写到新路径，保证页面显示的是实际值、保存后也不会
        用 schema 默认值覆盖用户历史设置。
        """
        try:
            rag = self.plugin._global_cfg().memory_rag
        except Exception:
            return
        mapping = {
            "recall_k": rag.recall_k,
            "max_return_results": rag.max_return_results,
            "memory_online_scan_limit": rag.online_scan_limit,
            "memory_min_similarity": rag.min_similarity,
            "memory_mmr_enabled": rag.mmr_enabled,
            "memory_kind_badge": rag.kind_badge,
            "memory_entity_boost": rag.entity_boost,
            "memory_lexical_recall": rag.lexical_recall,
            "memory_embedding_cache_ttl": rag.embedding_cache_ttl,
            "memory_embedding_cache_size": rag.embedding_cache_size,
            "memory_recall_cache_ttl_sec": rag.recall_cache_ttl,
            "memory_recall_cache_entries": rag.recall_cache_entries,
            "memory_dedup_threshold": rag.dedup_threshold,
            "memory_conflict_threshold": rag.conflict_threshold,
            "memory_window_gap_minutes": rag.window_gap_minutes,
            "memory_importance_decay": rag.importance_decay_enable,
            "memory_importance_decay_rate": rag.importance_decay_rate,
            "memory_importance_decay_floor": rag.importance_decay_floor,
            "memory_importance_decay_access_window_days": (
                rag.importance_decay_access_window_days
            ),
            "memory_importance_decay_access_max_count": (
                rag.importance_decay_access_max_count
            ),
            "memory_importance_protected_threshold": rag.importance_protected_threshold,
            "memory_archive_retention_days": rag.archive_retention_days,
        }
        runtime = values.get("runtime")
        if not isinstance(runtime, dict):
            runtime = {}
            values["runtime"] = runtime
        tuning = runtime.get("memory_tuning")
        if not isinstance(tuning, dict):
            tuning = {}
            runtime["memory_tuning"] = tuning
        for key, value in mapping.items():
            tuning[key] = value

    async def get_config(self) -> dict[str, Any]:
        """返回配置 schema 与当前配置值，供页面渲染全局设置表单。"""
        values = copy.deepcopy(dict(self.plugin.config))
        self._inject_memory_tuning_values(values)
        return ok(
            {
                "schema": self._load_schema(),
                "values": values,
                "providers": self._provider_options(),
                "embedding_providers": self._embedding_options(),
            }
        )

    def _apply_config_patch(
        self,
        patch: dict[str, Any],
        *,
        error_prefix: str,
        require_saver: bool = True,
    ) -> tuple[str | None, str]:
        """统一配置补丁：快照 → 合并 → 写内存/磁盘 → 失败回滚。

        返回 ``(error, warning)``；成功时 error 为 None。``require_saver=False``
        允许宿主不支持写盘，仅本次运行生效。
        """
        previous = copy.deepcopy(dict(self.plugin.config))
        merged = deep_merge(copy.deepcopy(previous), patch)
        warning = ""
        try:
            self.plugin.config.clear()
            self.plugin.config.update(merged)
            saver = getattr(self.plugin, "_save_config", None)
            if not callable(saver):
                if require_saver:
                    raise RuntimeError("当前 AstrBot 版本不支持自动保存")
                warning = "当前 AstrBot 版本不支持自动保存，配置仅在本次运行生效"
            else:
                saver()
        except Exception as exc:
            logger.error(f"[ActiveReply] {error_prefix}: {exc}", exc_info=True)
            self._restore_config(previous)
            self.plugin.invalidate_config_cache()
            return f"{error_prefix}：{exc}", ""
        self.plugin.invalidate_config_cache()
        return None, warning

    async def save_config(self) -> dict[str, Any]:
        """按 schema 校验并保存全局配置。"""
        payload, error = await _read_payload()
        if error:
            return fail(error)
        values = (
            payload.get("values") if isinstance(payload.get("values"), dict) else {}
        )
        if not values:
            return fail("没有需要保存的配置项")

        schema = self._load_schema()
        if not schema:
            return fail("配置 schema 不可用")

        patch = build_schema_patch(schema, values)
        if not patch:
            return fail("没有可保存的配置项（提交的值与 schema 不匹配）")
        error, warning = self._apply_config_patch(
            patch, error_prefix="保存配置失败", require_saver=False
        )
        if error:
            return fail(error)
        logger.info(
            "[ActiveReply] 页面保存全局配置 | 字段数=%s 写盘=%s",
            len(patch),
            "成功" if not warning else "失败",
        )
        return ok({"saved": True, "warning": warning, "applied": patch})

    # ------------------------------------------------------------------ #
    # 概览
    # ------------------------------------------------------------------ #
    def _memory_stats(self) -> dict[str, Any]:
        """统一的记忆统计入口：总数、来源与今日新增，供两个页面复用。

        本方法是同步纯函数，内部所有 SQLite 访问（``get_stats`` /
        ``source_counts`` / ``count_since``）都集中在这里；调用方必须通过
        ``await asyncio.to_thread(self._memory_stats)`` 执行，避免同步数据库
        操作阻塞事件循环。
        """
        store = self.plugin.memory_store
        if store is None:
            return {}
        stats = store.get_stats()
        counts = store.source_counts()
        now = datetime.datetime.now(self.plugin._tzinfo())
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        stats["source_counts"] = {
            "keyword": int(counts.get("keyword", 0)),
            "model": int(counts.get("model", 0)),
            "manual": int(counts.get("manual", 0)),
            "consolidation": int(counts.get("consolidation", 0)),
        }
        stats["today"] = int(store.count_since(today_start))
        # r3.3k：遗忘可见性统计（即将过期 / 已过期未归档 / 已归档）。
        try:
            stats.update(store.forgetting_counts(now=time.time()))
        except Exception as exc:  # 统计失败不影响记忆列表
            logger.debug(f"[ActiveReply] 读取遗忘统计失败: {exc}")
        # r3.3l：最近 7 天新归档数量与原因分布（每周遗忘汇总）。
        try:
            stats.update(store.weekly_forgetting_counts(now=time.time()))
        except Exception as exc:
            logger.debug(f"[ActiveReply] 读取每周遗忘统计失败: {exc}")
        return stats

    # 召回指标键名：与 main.py 主流程 bump_metric 写入 RuntimeState.turn_usage
    # 的键保持一致（recall_deadline_hit / embedding_timeout / recall_degraded 等）。
    _RECALL_METRIC_KEYS = (
        "recall_ms",
        "memory_hits",
        "lexical_hits",
        "embedding_cache_hit",
        "recall_cache_hit",
        "recall_deadline_hit",
        "embedding_timeout",
        "recall_degraded",
        # N-06：并行取舍与分阶段耗时也要在控制台可见。
        "embedding_wasted",
        "lexical_ms",
        "embedding_ms",
        "vector_ms",
        # P1-3：词法超时后转向量兜底的次数（与整体降级区分）。
        "lexical_timeout",
    )

    def _runtime_metrics(self) -> dict[str, Any]:
        """读取最近一轮的记忆召回指标，供状态页展示耗时与降级情况。

        只读进程内的 ``RuntimeState``（纯内存，不触碰存储），因此无需线程池。
        运行态或轮次数据缺失时返回空 dict；存在运行态时保证指标键稳定出现，
        缺失的计数器按 0 返回，而不是让整个概览接口失败。
        """
        state = getattr(self.plugin, "state", None)
        if state is None:
            return {}
        usage_map = getattr(state, "turn_usage", None) or {}
        started_map = getattr(state, "turn_started_at", None) or {}
        # 取最近开始的一轮；没有轮次时间时退回第一个有数据的来源。
        origin = ""
        latest = -1.0
        for key, stamp in started_map.items():
            try:
                value = float(stamp)
            except (TypeError, ValueError):
                continue
            if value >= latest:
                latest = value
                origin = str(key)
        if not origin and usage_map:
            origin = str(next(iter(usage_map)))
        bucket = usage_map.get(origin) or {}
        metrics: dict[str, Any] = {"origin": origin}
        for key in self._RECALL_METRIC_KEYS:
            try:
                metrics[key] = int(bucket.get(key, 0) or 0)
            except (TypeError, ValueError):
                metrics[key] = 0
        metrics["degraded"] = bool(
            metrics["recall_deadline_hit"]
            or metrics["embedding_timeout"]
            or metrics["recall_degraded"]
        )
        metrics["auto_extract"] = self._auto_extract_metrics(state)
        return metrics

    @staticmethod
    def _auto_extract_metrics(state: Any) -> dict[str, Any]:
        """汇总后台记忆抽取的运行态计数；结构缺失时返回空 dict。

        运行态目前只按来源记录待抽取消息数、最近抽取时间和重试次数，没有
        「累计抽取次数」字段，因此这里不臆造该数字；前端如需展示可先用
        ``origins`` 表示有过抽取的来源数。
        """
        counters = getattr(state, "extract_counters", None)
        last_at = getattr(state, "last_extract_at", None)
        retries = getattr(state, "extract_retries", None)
        if not any(isinstance(item, dict) for item in (counters, last_at, retries)):
            return {}
        try:
            pending = sum(int(value or 0) for value in (counters or {}).values())
            retry_total = sum(int(value or 0) for value in (retries or {}).values())
            last_value = max(
                (float(value) for value in (last_at or {}).values()), default=0.0
            )
        except (TypeError, ValueError):
            return {}
        return {
            "pending_messages": pending,
            "retries": retry_total,
            "last_at": last_value,
            "origins": len(counters or {}),
        }

    async def overview(self) -> dict[str, Any]:
        """汇总插件版本、群组、记忆统计、提供商和功能开关。"""
        # 汇总版本、群、记忆、Embedding 和全局开关供状态页一次渲染。
        plugin = self.plugin
        cfg = plugin._global_cfg()
        force = bool(query_get("force", 0, int))
        try:
            groups = await plugin.collect_groups(force=force)
        except Exception as exc:
            logger.error(f"[ActiveReply] 获取群列表失败: {exc}", exc_info=True)
            groups = []

        stats: dict[str, Any] = {}
        try:
            stats = await asyncio.to_thread(self._memory_stats)
        except Exception as exc:
            logger.error(f"[ActiveReply] 读取记忆统计失败: {exc}", exc_info=True)

        metrics: dict[str, Any] = {}
        try:
            metrics = self._runtime_metrics()
        except Exception as exc:
            logger.error(f"[ActiveReply] 读取召回指标失败: {exc}", exc_info=True)

        embedding_options = self._embedding_options()
        embedding = {
            "available": bool(embedding_options),
            "count": len(embedding_options),
            "configured": bool(cfg.memory_rag.embedding_provider_id),
            "providers": embedding_options,
        }

        return ok(
            {
                "version": self._plugin_version(),
                "timezone": plugin._resolve_timezone_name(),
                # 只回传目录名：绝对路径属于服务器内部信息（F43）。
                "data_dir": self._display_data_dir(plugin.data_dir),
                "groups": groups,
                "stats": stats,
                "metrics": metrics,
                "embedding": embedding,
                "global": build_global_snapshot(cfg),
                "switches": {
                    "active_reply": cfg.active_reply.enable,
                    "mode": cfg.active_reply.mode,
                    "context": cfg.group_context.enable,
                    "react_pattern": cfg.group_reply.react_pattern,
                    "smart_reply": cfg.smart_reply_enabled,
                    "web_search": cfg.web_search.enable,
                    "memory_rag": cfg.memory_rag.enable,
                },
                "memory_engine": {
                    "fts_enabled": bool(
                        getattr(plugin.memory_store, "fts_enabled", False)
                    ),
                    "lexical_recall": cfg.memory_rag.lexical_recall,
                    "inject_items": cfg.memory_rag.inject_max_items,
                    "inject_chars": cfg.memory_rag.inject_max_chars,
                    "online_scan_limit": cfg.memory_rag.online_scan_limit,
                    "min_similarity": cfg.memory_rag.min_similarity,
                    "embedding_cache_ttl": cfg.memory_rag.embedding_cache_ttl,
                },
                "providers": self._provider_options(),
            }
        )

    # ------------------------------------------------------------------ #
    # 群配置
    # ------------------------------------------------------------------ #
    async def list_groups(self) -> dict[str, Any]:
        """获取已知群列表，并按查询参数决定是否强制刷新缓存。"""
        # 查询平台、运行时和本地配置中已知的群，并支持强制刷新缓存。
        force = bool(query_get("force", 0, int))
        try:
            groups = await self.plugin.collect_groups(force=force)
        except Exception as exc:
            logger.error(f"[ActiveReply] 获取群列表失败: {exc}", exc_info=True)
            return fail(f"获取群列表失败：{exc}")
        return ok({"groups": groups})

    @staticmethod
    def _schema_default(schema: dict[str, Any], path: str) -> Any:
        """按路径取出 schema 里该配置项的默认值。"""
        node: Any = schema
        parts = path.split(".")
        for part in parts[:-1]:
            node = (node.get(part) or {}).get("items") or {}
        leaf = node.get(parts[-1]) if isinstance(node, dict) else None
        return (leaf or {}).get("default")

    @staticmethod
    def _schema_hint(schema: dict[str, Any], path: str) -> str:
        """按 "a.b.c" 路径取出 schema 里对应配置项的 hint 文案。"""
        node: Any = schema
        parts = path.split(".")
        for part in parts[:-1]:
            node = (node.get(part) or {}).get("items") or {}
        leaf = node.get(parts[-1]) if isinstance(node, dict) else None
        return str((leaf or {}).get("hint") or "")

    async def get_group(self, group_id: str) -> dict[str, Any]:
        """返回指定群的有效配置、全局值、字段说明和展示名称。"""
        # 返回群级有效值、全局快照、字段元数据和展示名称。
        group_id = str(group_id or "").strip()
        if not group_id:
            return fail("group_id 不能为空")
        plugin = self.plugin
        cfg = plugin._global_cfg()
        global_values = build_global_snapshot(cfg)
        values, independent = plugin.group_store.effective_values(
            group_id, global_values
        )
        meta = plugin.group_store.get_meta(group_id)

        # 群名优先取平台返回的群列表，其次才是本地备注
        name = ""
        try:
            for item in await plugin.collect_groups():
                if item.get("group_id") == group_id:
                    name = str(item.get("name") or "")
                    break
        except Exception:
            name = ""
        if not name:
            name = str(meta.get("note") or "")

        schema = self._load_schema()
        return ok(
            {
                "group_id": group_id,
                "name": name,
                "independent": independent,
                "global": global_values,
                "values": values,
                "meta": meta,
                "fields": [
                    {
                        "path": spec.path,
                        "label": spec.label,
                        "kind": spec.kind,
                        "options": list(spec.options),
                        "labels": list(spec.labels),
                        "min": spec.minimum,
                        "max": spec.maximum,
                        "step": spec.step,
                        "hint": self._schema_hint(schema, spec.path),
                        "default": self._schema_default(schema, spec.path),
                    }
                    for spec in GROUP_OVERRIDE_FIELDS
                ],
                "providers": self._provider_options(),
            }
        )

    async def save_group(self, group_id: str) -> dict[str, Any]:
        """校验并保存群级覆盖，或切换为跟随全局配置。"""
        # 校验并保存群级允许覆盖项，或将该群设置为跟随全局配置。
        group_id = str(group_id or "").strip()
        if not group_id:
            return fail("group_id 不能为空")
        payload, error = await _read_payload()
        if error:
            return fail(error)
        independent = bool(payload.get("independent", True))
        note = str(payload.get("note") or "")
        values = (
            payload.get("values") if isinstance(payload.get("values"), dict) else {}
        )

        plugin = self.plugin
        try:
            if independent:
                if not values:
                    return fail("启用独立配置时至少需要提交一个字段")
                # 只保存“与全局值不同”的字段。页面会提交完整表单，
                # 若把未修改的默认值（尤其是记忆 RAG 的默认 False）也写入，
                # 会导致该群记忆在没有提示的情况下被关闭。
                global_values = build_global_snapshot(plugin._global_cfg())
                existing = plugin.group_store.get_overrides(group_id)
                overrides: dict[str, Any] = {}
                for spec in GROUP_OVERRIDE_FIELDS:
                    current = get_by_path(existing, spec.path)
                    if current is not None:
                        overrides[spec.path] = current
                for path, value in values.items():
                    spec = GROUP_FIELD_BY_PATH.get(str(path))
                    if spec is None:
                        continue
                    coerced = coerce_group_value(spec, value)
                    if coerced == global_values.get(path):
                        overrides.pop(path, None)
                    else:
                        overrides[path] = coerced
                saved = plugin.group_store.save_group_config(
                    group_id,
                    overrides,
                    independent=bool(overrides),
                    note=note,
                )
            else:
                saved = plugin.group_store.save_group_config(
                    group_id, {}, independent=False, note=note
                )
        except Exception as exc:
            logger.error(f"[ActiveReply] 保存群配置失败: {exc}", exc_info=True)
            return fail(f"保存失败：{exc}")

        plugin.invalidate_config_cache()
        # 提交值全部等于全局值时会退化为「跟随全局」，返回值必须以实际存储为准，
        # 否则前端会显示「已独立配置」而后台其实已跟随全局。
        actual_independent = bool(
            plugin.group_store.get_meta(group_id).get("independent")
        )
        logger.info(
            "[ActiveReply] 群配置已保存 | group=%s independent=%s",
            group_id,
            actual_independent,
        )
        return ok(
            {
                "group_id": group_id,
                "independent": actual_independent,
                "saved": saved,
            }
        )

    async def reset_group(self, group_id: str) -> dict[str, Any]:
        """删除指定群的本地覆盖配置，使其恢复继承全局设置。"""
        # 删除该群持久化覆盖项，使其恢复跟随全局设置。
        group_id = str(group_id or "").strip()
        if not group_id:
            return fail("group_id 不能为空")
        removed = self.plugin.group_store.remove(group_id)
        self.plugin.invalidate_config_cache()
        return ok({"group_id": group_id, "removed": removed})

    # ------------------------------------------------------------------ #
    # 开发者模式
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hash_dev_key(key: str) -> str:
        """计算开发者密钥的加盐 SHA-256，用于与固定哈希做常量时间比对。"""
        # 只比较哈希，源码、日志和响应里都不出现明文密钥。
        return hashlib.sha256((_DEV_KEY_SALT + str(key)).encode("utf-8")).hexdigest()

    def _check_dev_token(self, token: str) -> bool:
        """校验开发者令牌是否仍在有效期内；随机令牌只做字典查找。"""
        token = str(token or "")
        if not token:
            return False
        now = time.time()
        # 顺手清理过期令牌，避免长时间运行时内存条目越积越多。
        expired = [
            key for key, expiry in self._dev_tokens.items() if float(expiry) <= now
        ]
        for key in expired:
            self._dev_tokens.pop(key, None)
        expiry = self._dev_tokens.get(token)
        return bool(expiry and float(expiry) > now)

    def _dev_token_from_request(self, payload: dict[str, Any] | None = None) -> str:
        """从请求体或 X-Dev-Token 请求头取开发者令牌（不再接受 query 参数）。

        令牌可能出现在 URL、代理日志与 Referer 中，query 回退会把它泄露给
        这些中间环节，因此只信 body 与专用请求头。
        """
        if isinstance(payload, dict):
            token = payload.get("dev_token")
            if token:
                return str(token)
        return header_get("X-Dev-Token", "")

    def _require_dev(self, payload: dict[str, Any] | None = None) -> str | None:
        """开发者接口门禁：令牌有效返回 ``None``，否则返回统一错误文案。"""
        if self._check_dev_token(self._dev_token_from_request(payload)):
            return None
        return _DEV_INVALID_MSG

    def _dev_login_blocked(self) -> bool:
        """失败次数达到窗口内上限时拒绝继续尝试（防暴力破解）。"""
        now = time.time()
        cutoff = now - _DEV_FAIL_WINDOW_SEC
        kept = [stamp for stamp in self._dev_fail_times if float(stamp) > cutoff]
        self._dev_fail_times[:] = kept
        return len(kept) >= _DEV_MAX_FAILS

    async def dev_login(self) -> dict[str, Any]:
        """校验开发者密钥，成功时签发短时内存令牌。"""
        payload, error = await _read_payload()
        if error:
            return fail(error)
        if self._dev_login_blocked():
            return fail("尝试过于频繁，请稍后再试")
        key = str(payload.get("key") or "")
        # 常量时间比较，避免通过响应耗时逐字节猜哈希。
        if not hmac.compare_digest(self._hash_dev_key(key), _DEV_KEY_EFFECTIVE_SHA256):
            self._dev_fail_times.append(time.time())
            logger.warning("[ActiveReply] 开发者密钥校验失败")
            return fail("开发者密钥错误")
        token = secrets.token_urlsafe(32)
        self._dev_tokens[token] = time.time() + _DEV_TOKEN_TTL_SEC
        self._dev_fail_times.clear()
        logger.info("[ActiveReply] 开发者模式已解锁")
        return ok({"token": token, "expires_in": _DEV_TOKEN_TTL_SEC})

    async def dev_logout(self) -> dict[str, Any]:
        """销毁当前开发者令牌，退出开发者模式。"""
        payload, error = await _read_payload()
        if error:
            return fail(error)
        self._dev_tokens.pop(self._dev_token_from_request(payload), None)
        return ok({})

    def _current_prompt_overrides(self) -> dict[str, str]:
        """读取当前已保存的群/私聊抽取提示词补充层（r3.3n）。"""
        raw = getattr(self.plugin, "config", {}) or {}
        mem = raw.get("memory_rag") if isinstance(raw, dict) else {}
        if not isinstance(mem, dict):
            mem = {}
        gen = mem.get("memory_generation")
        if not isinstance(gen, dict):
            gen = {}

        def pick(key: str) -> str:
            return str(gen.get(key) or mem.get(key) or "").strip()

        return {
            "group_prompt": pick("memory_group_prompt"),
            "private_prompt": pick("memory_private_prompt"),
        }

    async def dev_prompts(self) -> dict[str, Any]:
        """开发者接口：读取群/私聊抽取提示词（r3.3n）。

        与 dev_prompts_save 一致从 JSON body 读令牌，配合路由改为 POST，
        令牌不再经过 URL。
        """
        payload, error = await _read_payload()
        if error:
            return fail(error)
        denied = self._require_dev(payload if isinstance(payload, dict) else None)
        if denied:
            return fail(denied)
        data = self._current_prompt_overrides()
        data["default_group"] = DEFAULT_GROUP_SUMMARY_PROMPT
        data["default_private"] = DEFAULT_PRIVATE_SUMMARY_PROMPT
        return ok(data)

    async def dev_prompts_save(self) -> dict[str, Any]:
        """开发者接口：保存群/私聊抽取提示词补充层（r3.3n）。

        提示词已从「记忆 RAG」全局设置移到开发者模式，因此这里绕过 schema
        校验直接写配置；空字符串表示恢复默认补充层。
        """
        payload, error = await _read_payload()
        if error:
            return fail(error)
        denied = self._require_dev(payload if isinstance(payload, dict) else None)
        if denied:
            return fail(denied)
        group_prompt = str(payload.get("group_prompt") or "").strip()
        private_prompt = str(payload.get("private_prompt") or "").strip()
        max_chars = 8000
        if len(group_prompt) > max_chars or len(private_prompt) > max_chars:
            return fail(f"提示词过长（单条最多 {max_chars} 字）")
        patch = {
            "memory_rag": {
                "memory_generation": {
                    "memory_group_prompt": group_prompt,
                    "memory_private_prompt": private_prompt,
                }
            }
        }
        error, _ = self._apply_config_patch(
            patch, error_prefix="保存提示词失败"
        )
        if error:
            return fail(error)
        logger.info("[ActiveReply] 开发者保存提示词 | group=%s private=%s", len(group_prompt), len(private_prompt))
        return ok(self._current_prompt_overrides())

    # ------------------------------------------------------------------ #
    # 记忆 RAG
    # ------------------------------------------------------------------ #
    def _forgetting_rules(self) -> dict[str, Any]:
        """返回当前生效的遗忘规则摘要，供记忆页展示（r3.3l）。"""
        try:
            rag = self.plugin._global_cfg().memory_rag
        except Exception:
            return {}
        return {
            "ttl_adaptive": bool(rag.ttl_adaptive_enable),
            "ttl_base": {
                "plan": int(rag.ttl_base.base_days("plan")),
                "episode": int(rag.ttl_base.base_days("episode")),
                "preference": int(rag.ttl_base.base_days("preference")),
                "relation": int(rag.ttl_base.base_days("relation")),
                "fact": int(rag.ttl_base.base_days("fact")),
            },
            "decay_enable": bool(rag.importance_decay_enable),
            "decay_rate": float(rag.importance_decay_rate),
            "decay_floor": float(rag.importance_decay_floor),
            "archive_enable": bool(rag.archive_enable),
            "archive_days": int(rag.archive_days),
            "archive_importance": float(rag.archive_importance),
            "archive_retention_days": int(rag.archive_retention_days),
            "consolidation_enable": bool(rag.consolidation_enable),
            "consolidation_interval_days": int(rag.consolidation_interval_days),
        }

    async def rag_stats(self) -> dict[str, Any]:
        """返回记忆库统计信息及已存在的记忆作用域列表。"""
        # 获取记忆库统计信息及已存在作用域列表。
        store = self.plugin.memory_store
        if store is None:
            return fail("记忆库未初始化")
        try:
            stats = await asyncio.to_thread(self._memory_stats)
            scopes = await asyncio.to_thread(store.list_group_scopes)
        except Exception as exc:
            logger.error(f"[ActiveReply] 读取记忆统计失败: {exc}", exc_info=True)
            return fail(f"读取失败：{exc}")
        return ok(
            {
                "stats": stats,
                "group_scopes": scopes,
                "rules": self._forgetting_rules(),
            }
        )

    def _enrich_group_roles(self, item: dict[str, Any]) -> None:
        """用当前群角色缓存补全记忆里的参与者身份（旧记忆没有 role_identities 时）。

        角色缓存只保留群主/群管与最近查询的成员，TTL 10 分钟；这里只读缓存，
        不触发网络请求，避免打开列表时卡住。找不到缓存时保持原样（前端按群成员显示）。
        """
        if not isinstance(item, dict):
            return
        group_id = str(item.get("group_id") or "").strip()
        related = item.get("related_role_ids")
        if not group_id or not isinstance(related, list) or not related:
            return
        cache = getattr(self.plugin, "_group_role_cache", None)
        if not isinstance(cache, dict):
            return
        entry = cache.get(group_id)
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            return
        updated_at, roles = entry
        if not isinstance(roles, dict):
            return
        try:
            if time.time() - float(updated_at) > 600:
                return
        except (TypeError, ValueError):
            return
        meta = item.get("extra_metadata")
        if not isinstance(meta, dict):
            meta = {}
        identities = meta.get("role_identities")
        if not isinstance(identities, dict):
            identities = {}
        merged = dict(identities)
        changed = False
        for role_id in related:
            role = roles.get(str(role_id))
            if role and merged.get(str(role_id)) != role:
                merged[str(role_id)] = role
                changed = True
        sender_id = str(item.get("sender_id") or "")
        if sender_id:
            role = roles.get(sender_id)
            if role and item.get("sender_role") != role:
                item["sender_role"] = role
                merged[sender_id] = role
                changed = True
        if changed:
            meta = dict(meta)
            meta["role_identities"] = merged
            item["extra_metadata"] = meta

    async def _group_name_map(self) -> dict[str, str]:
        """尽力返回 group_id -> 群名；优先用插件缓存，避免每次列表都拉平台。"""
        groups = getattr(self.plugin, "_groups_cache", None)
        if not groups:
            try:
                groups = await self.plugin.collect_groups(force=False)
            except Exception:
                groups = []
        mapping: dict[str, str] = {}
        for group in groups or []:
            if not isinstance(group, dict):
                continue
            gid = str(group.get("group_id") or "").strip()
            name = str(group.get("name") or "").strip()
            if gid and name:
                mapping[gid] = name
        return mapping

    async def rag_list(self) -> dict[str, Any]:
        """按 URL 查询条件分页列出记忆，并补充记忆来源字段。"""
        # 从 URL 查询参数读取筛选和分页条件，补充来源标签后返回记忆行。
        store = self.plugin.memory_store
        if store is None:
            return fail("记忆库未初始化")
        page = query_get("page", 1, int)
        page_size = query_get("page_size", 20, int)
        keyword = query_get("keyword", "", str)
        group_scope = query_get("group_scope", "", str)
        role_id = query_get("role_id", "", str)
        memory_type = query_get("memory_type", "", str)
        identity = query_get("identity", "", str)
        lifecycle = str(query_get("lifecycle", "", str) or "").strip().lower()
        include_archived = bool(query_get("include_archived", 0, int))
        # r3.3：控制台「已归档」开关只看归档记录，优先于 include_archived。
        archived_only = bool(query_get("archived_only", 0, int))
        rag_cfg = self.plugin._global_cfg().memory_rag
        if archived_only:
            # r3.3m：「已归档」视图与生命周期筛选互斥，避免 SQL 恒空（P2 F-01）。
            lifecycle = ""
        try:
            data = await asyncio.to_thread(
                store.list_memories,
                page=page or 1,
                page_size=page_size or 20,
                keyword=keyword or "",
                group_scope=group_scope or "",
                role_id=role_id or "",
                memory_type=memory_type or "",
                identity=identity or "",
                include_archived=include_archived,
                archived_only=archived_only,
                lifecycle=lifecycle,
                lifecycle_importance=float(rag_cfg.archive_importance),
                lifecycle_days=float(rag_cfg.archive_days),
                lifecycle_decay_enable=bool(rag_cfg.importance_decay_enable),
                lifecycle_decay_rate=float(rag_cfg.importance_decay_rate),
                lifecycle_decay_floor=float(rag_cfg.importance_decay_floor),
                lifecycle_access_window_days=float(
                    rag_cfg.importance_decay_access_window_days
                ),
                lifecycle_access_max_count=int(
                    rag_cfg.importance_decay_access_max_count
                ),
                lifecycle_protected=float(
                    rag_cfg.importance_protected_threshold
                ),
            )
        except Exception as exc:
            logger.error(f"[ActiveReply] 读取记忆列表失败: {exc}", exc_info=True)
            return fail(f"读取失败：{exc}")
        decor_now = time.time()
        decor_rag = self.plugin._global_cfg().memory_rag
        group_names = await self._group_name_map()
        for item in data.get("items") or []:
            if not isinstance(item, dict):
                continue
            self._enrich_group_roles(item)
            meta = item.get("extra_metadata") or {}
            group_id = str(item.get("group_id") or "").strip()
            item["group_name"] = group_names.get(group_id, "")
            item["source"] = str(meta.get("source") or "manual")
            item["extra_metadata"] = self._compact_list_metadata(meta)
            self._decorate_forgetting(item, now=decor_now, rag=decor_rag)
        return ok(data)

    # ------------------------------------------------------------------ #
    # 自动记忆
    # ------------------------------------------------------------------ #
    def _auto_memory_state(self) -> dict[str, Any]:
        """返回自动抽取参数与开关状态（统计数字由记忆页统一提供）。"""
        # 自动抽取参数；统计数字由记忆页统一提供，这里不再重复查询。
        cfg = self.plugin._global_cfg()
        return {
            "enabled": bool(
                cfg.memory_rag.enable and cfg.memory_rag.auto_extract_enable
            ),
            "memory_enabled": bool(cfg.memory_rag.enable),
            "provider_id": cfg.memory_rag.auto_extract_provider_id,
            "batch_messages": cfg.memory_rag.batch_messages,
            "interval_minutes": cfg.memory_rag.auto_extract_interval_minutes,
            "private_every_n": cfg.memory_rag.private_extract_every_n,
            "min_confidence": cfg.memory_rag.auto_extract_min_confidence,
            "scope": cfg.memory_rag.auto_extract_scope,
            "providers": self._provider_options(),
        }

    async def memory_auto(self) -> dict[str, Any]:
        """返回自动记忆面板所需的只读状态。"""
        # 自动记忆面板的只读状态接口。
        return ok(self._auto_memory_state())

    async def memory_auto_toggle(self) -> dict[str, Any]:
        """只更新自动抽取开关，保存配置并使运行时解析缓存失效。

        写盘失败不再静默返回成功：恢复内存中的旧配置并返回统一错误响应，
        让前端能提示用户，避免重启后开关被悄悄回滚（F21）。
        """
        payload, error = await _read_payload()
        if error:
            return fail(error)
        if "enable" not in payload:
            return fail("缺少 enable 参数")
        enable = bool(payload.get("enable"))
        patch = {"memory_rag": {"memory_generation": {"auto_extract_enable": enable}}}
        error, _ = self._apply_config_patch(patch, error_prefix="保存配置失败")
        if error:
            return fail(error)
        return ok(self._auto_memory_state())

    def _restore_config(self, previous: dict[str, Any]) -> None:
        """保存失败时把内存配置回滚到改动前的快照（F21）。"""
        try:
            self.plugin.config.clear()
            self.plugin.config.update(previous)
        except Exception as exc:
            logger.warning(f"[ActiveReply] 回滚内存配置失败: {exc}")

    def _decorate_forgetting(
        self,
        item: dict[str, Any],
        now: float | None = None,
        rag: Any = None,
    ) -> None:
        """为记忆行补充「遗忘可见性」字段（r3.3k）。

        字段：``effective_importance``（当前衰减后的有效重要性）、
        ``valid_until_iso``、``days_left``（可为负）、``expired``；
        归档原因已在 ``archive_reason`` 中。
        """
        if not isinstance(item, dict):
            return
        current = time.time() if now is None else float(now)
        if rag is None:
            try:
                rag = self.plugin._global_cfg().memory_rag
            except Exception:
                rag = None
        try:
            raw_importance = item.get("importance")
            # importance 为 0-10，缺省或非法时按默认 5.0 展示（confidence 仍是 0-1）。
            importance = (
                IMPORTANCE_DEFAULT if raw_importance is None else float(raw_importance)
            )
        except (TypeError, ValueError):
            importance = IMPORTANCE_DEFAULT
        effective = importance
        if rag is not None and bool(rag.importance_decay_enable):
            try:
                effective = compute_effective_importance(
                    importance=importance,
                    confidence=item.get("confidence"),
                    created_at=float(item.get("created_at") or 0.0),
                    last_access=float(item.get("last_access") or 0.0),
                    access_count=int(item.get("access_count") or 0),
                    decay_reset_at=float(item.get("decay_reset_at") or 0.0),
                    now=current,
                    decay_rate=float(rag.importance_decay_rate),
                    decay_floor=float(rag.importance_decay_floor),
                    access_window_days=float(
                        rag.importance_decay_access_window_days
                    ),
                    access_max_count=int(rag.importance_decay_access_max_count),
                    protected=float(rag.importance_protected_threshold),
                )
            except Exception:
                effective = importance
        item["effective_importance"] = round(
            max(0.0, min(IMPORTANCE_MAX, effective)), 3
        )
        store = getattr(self.plugin, "memory_store", None)
        try:
            item["created_at_iso"] = (
                store.format_timestamp_iso(float(item.get("created_at") or 0.0))
                if store
                else ""
            )
        except Exception:
            item["created_at_iso"] = ""
        try:
            last_access_ts = float(item.get("last_access") or 0.0)
            item["last_access_iso"] = (
                store.format_timestamp_iso(last_access_ts)
                if store and last_access_ts > 0
                else ""
            )
        except Exception:
            item["last_access_iso"] = ""
        if item.get("archived"):
            # r3.3m：优先用独立 archived_at，旧归档行回落到 decay_reset_at。
            store = getattr(self.plugin, "memory_store", None)
            try:
                archived_at = float(
                    item.get("archived_at") or item.get("decay_reset_at") or 0.0
                )
            except (TypeError, ValueError):
                archived_at = 0.0
            item["archived_at"] = archived_at
            try:
                item["archived_at_iso"] = (
                    store.format_timestamp_iso(archived_at)
                    if store and archived_at > 0
                    else ""
                )
            except Exception:
                item["archived_at_iso"] = ""
        try:
            valid_ts = float(item.get("valid_until") or 0.0)
        except (TypeError, ValueError):
            valid_ts = 0.0
        if valid_ts > 0:
            store = getattr(self.plugin, "memory_store", None)
            try:
                item["valid_until_iso"] = (
                    store.format_timestamp_iso(valid_ts) if store else ""
                )
            except Exception:
                item["valid_until_iso"] = ""
            item["days_left"] = int(math.ceil((valid_ts - current) / 86400.0))
            item["expired"] = valid_ts < current
        else:
            item["valid_until_iso"] = ""
            item["days_left"] = None
            item["expired"] = False

    async def _episode_atoms(self, episode_id: int) -> list[dict[str, Any]]:
        """返回挂在某个 episode 下的原子记忆（最多扫描 200 条）。"""
        store = self.plugin.memory_store
        if store is None:
            return []
        data = await asyncio.to_thread(
            store.list_memories,
            page=1,
            page_size=200,
            include_archived=True,
        )
        atoms: list[dict[str, Any]] = []
        for item in data.get("items", []):
            meta = item.get("extra_metadata")
            if not isinstance(meta, dict):
                continue
            try:
                linked = int(meta.get("episode_id") or 0)
            except (TypeError, ValueError):
                linked = 0
            if linked == int(episode_id):
                atoms.append(item)
        atoms.sort(key=lambda entry: int(entry.get("memory_id") or 0))
        return atoms

    async def rag_detail(self, memory_id: str) -> dict[str, Any]:
        """按记忆 ID 返回完整记录；记录不存在时返回统一失败响应。"""
        # 按 ID 返回完整记忆对象供详情抽屉显示。
        store = self.plugin.memory_store
        if store is None:
            return fail("记忆库未初始化")
        item = await asyncio.to_thread(store.get_memory, memory_id)
        if not item:
            return fail("记忆不存在")
        self._enrich_group_roles(item)
        self._decorate_forgetting(
            item, now=time.time(), rag=self.plugin._global_cfg().memory_rag
        )
        meta = item.get("extra_metadata")
        meta = meta if isinstance(meta, dict) else {}
        try:
            parent_id = int(meta.get("episode_id") or 0)
        except (TypeError, ValueError):
            parent_id = 0
        if parent_id:
            parent = await asyncio.to_thread(store.get_memory, parent_id)
            if parent:
                item["episode"] = {
                    "memory_id": parent.get("memory_id"),
                    "kind": parent.get("kind"),
                    "content": parent.get("content"),
                }
        if str(item.get("kind") or "").lower() == "episode":
            atoms = await self._episode_atoms(int(item.get("memory_id") or 0))
            item["atoms"] = atoms
        return ok(item)

    def _invalidate_recall_cache_hook(self, action: str) -> None:
        """页面改动了记忆数据后，通知插件失效短期召回缓存。

        使用 ``getattr`` 探测插件钩子：旧版本插件还没有
        ``_invalidate_recall_cache`` 时静默跳过，避免页面接口因缓存钩子缺失
        而失败。缓存失效失败只记日志，不回滚已经完成的记忆操作。
        """
        invalidate = getattr(self.plugin, "_invalidate_recall_cache", None)
        if not callable(invalidate):
            return
        try:
            invalidate(action)
        except Exception as exc:
            logger.warning(f"[ActiveReply] 失效召回缓存失败: {exc}")

    async def rag_delete(self, memory_id: str) -> dict[str, Any]:
        """删除指定记忆，并记录页面操作日志。"""
        # 删除指定记忆并记录页面操作日志。
        store = self.plugin.memory_store
        if store is None:
            return fail("记忆库未初始化")
        removed = await asyncio.to_thread(store.delete_memory, memory_id)
        if not removed:
            return fail("记忆不存在或已删除")
        self._invalidate_recall_cache_hook("page_api")
        logger.info("[ActiveReply] 页面删除记忆 | id=%s", memory_id)
        return ok(
            {
                "memory_id": memory_id,
                "deleted": True,
                "trash_id": await asyncio.to_thread(store.latest_trash_id),
            }
        )

    async def rag_archive(self, memory_id: str) -> dict[str, Any]:
        """归档指定记忆：保留数据但不再参与召回。"""
        # r3.3：归档是遗忘链路的手动入口，只改 archived 标记。
        store = self.plugin.memory_store
        if store is None:
            return fail("记忆库未初始化")
        archived = await asyncio.to_thread(store.archive_memory, memory_id)
        if not archived:
            return fail("记忆不存在或已归档")
        self._invalidate_recall_cache_hook("page_api-archive")
        logger.info("[ActiveReply] 页面归档记忆 | id=%s", memory_id)
        return ok({"memory_id": int(memory_id), "archived": True})

    def _unarchive_ttl_days(self, item: dict[str, Any]) -> float | None:
        """按 TTL 自适应配置计算恢复后的有效期天数；``None`` 表示不续期。

        与写入侧保持一致：基础天数乘以 ``0.5 + clamp(importance)/10`` 的系数；
        关闭 TTL 自适应或该类型基础天数为 0（永不过期）时不写有效期。
        """
        kind = str(item.get("kind") or "fact")
        try:
            cfg = self.plugin._global_cfg().memory_rag
            adaptive = bool(cfg.ttl_adaptive_enable)
            base = int(cfg.ttl_base.base_days(kind))
        except Exception as exc:
            logger.warning(f"[ActiveReply] 计算恢复有效期失败: {exc}")
            return None
        if not adaptive:
            # r3.3m：关闭自适应 TTL 时沿用旧按类型硬过期，避免恢复后仍过期。
            try:
                legacy_days = cfg.kind_decay.expiry_days(kind)
            except Exception:
                legacy_days = None
            return float(legacy_days) if legacy_days else None
        if base <= 0:
            return None
        try:
            importance = float(item.get("importance", 5.0))
        except (TypeError, ValueError):
            importance = 5.0
        importance = min(IMPORTANCE_MAX, max(0.0, importance))
        return float(base) * (0.5 + importance / 10.0)

    async def rag_unarchive(self, memory_id: str) -> dict[str, Any]:
        """取消归档：重置衰减基准并按 TTL 自适应恢复有效期。"""
        store = self.plugin.memory_store
        if store is None:
            return fail("记忆库未初始化")
        item = await asyncio.to_thread(store.get_memory, memory_id)
        if not item:
            return fail("记忆不存在")
        ttl_days = self._unarchive_ttl_days(item)
        new_item = await asyncio.to_thread(
            store.unarchive_memory, memory_id, ttl_days=ttl_days
        )
        if new_item is None:
            return fail("记忆不存在或未归档")
        self._invalidate_recall_cache_hook("page_api-unarchive")
        logger.info("[ActiveReply] 页面取消归档记忆 | id=%s", memory_id)
        return ok(
            {
                "memory_id": int(new_item.get("memory_id") or 0),
                "archived": False,
                "memory": new_item,
            }
        )

    @staticmethod
    def _truthy(value: Any) -> bool:
        """把 ``1/true/"1"`` 等提交值归一化成布尔真值。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    async def rag_purge(self, memory_id: str) -> dict[str, Any]:
        """彻底删除指定记忆：不写回收站且不可撤销，必须显式确认。"""
        store = self.plugin.memory_store
        if store is None:
            return fail("记忆库未初始化")
        payload, error = await _read_payload()
        if error:
            return fail(error)
        if not self._truthy(payload.get("confirm")):
            return fail("彻底删除需要确认（confirm）")
        removed = await asyncio.to_thread(store.delete_memory_permanently, memory_id)
        if not removed:
            return fail("记忆不存在或已删除")
        self._invalidate_recall_cache_hook("page_api-purge")
        logger.info("[ActiveReply] 页面彻底删除记忆 | id=%s", memory_id)
        return ok({"memory_id": int(memory_id), "purged": True})

    async def rag_restore(self) -> dict[str, Any]:
        """撤销删除：从回收站恢复一条记忆。"""
        store = self.plugin.memory_store
        if store is None:
            return fail("记忆库未初始化")
        payload, error = await _read_payload()
        if error:
            return fail(error)
        try:
            trash_id = int(payload.get("trash_id") or 0)
        except (TypeError, ValueError):
            return fail("trash_id 无效")
        if trash_id <= 0:
            return fail("trash_id 无效")
        new_id = await asyncio.to_thread(store.restore_memory, trash_id)
        if new_id is None:
            return fail("回收站中没有该记录")
        self._invalidate_recall_cache_hook("page_api")
        logger.info("[ActiveReply] 页面撤销删除 | trash=%s new_id=%s", trash_id, new_id)
        return ok({"memory_id": new_id, "restored": True})

    async def rag_cleanup(self) -> dict[str, Any]:
        """在线程池执行旧格式记忆数据清理，避免阻塞事件循环。"""
        # 在线程池执行旧数据规范化，避免同步 SQLite 操作阻塞事件循环。
        store = self.plugin.memory_store
        if store is None:
            return fail("记忆库未初始化")
        try:
            result = await asyncio.to_thread(store.cleanup_legacy_records)
        except Exception as exc:
            logger.error(f"[ActiveReply] 清理记忆失败: {exc}", exc_info=True)
            return fail(f"清理失败：{exc}")
        self._invalidate_recall_cache_hook("page_api")
        return ok(result)

    async def prompt_preview(self) -> dict[str, Any]:
        """返回最终提示词（含系统追加规则）供面板展示。"""
        # r3.5a：与 dev_prompts 一致从 JSON body 读令牌，避免 token 进 URL。
        payload, error = await _read_payload()
        if error:
            return fail(error)
        denied = self._require_dev(payload if isinstance(payload, dict) else None)
        if denied:
            return fail(denied)
        kind = (
            "private"
            if str((payload or {}).get("kind") or "").strip().lower() == "private"
            else "group"
        )
        try:
            return ok(self.plugin.prompt_preview(kind))
        except Exception as exc:
            logger.error(f"[ActiveReply] 生成提示词预览失败: {exc}", exc_info=True)
            return fail(f"生成预览失败：{exc}")

    async def rag_selftest(self) -> dict[str, Any]:
        """记忆 RAG 自检：默认只读检查，deep 模式才写临时记忆并调用模型。

        默认（shallow）只检查记忆库可读、FTS 能力与 Embedding 提供商是否就绪，
        不向生产库写入任何数据，也不调用 embedding / 抽取模型；只有显式传入
        ``deep=true`` 时才执行「生成向量 → 写入/读回/语义检索/删除临时记忆」
        并运行一次提示词抽取自检（F23）。``steps`` 里不回传服务器绝对路径
        （F43）。
        """
        plugin = self.plugin
        steps: list[dict[str, Any]] = []

        def add(name: str, passed: bool, detail: str = "") -> None:
            """将单项自检结果追加到统一结构的步骤列表。"""
            # 使用同一结构累计每个自检步骤，便于前端逐项显示失败原因。
            steps.append({"name": name, "ok": bool(passed), "detail": str(detail)})

        def finish() -> dict[str, Any]:
            """按已收集的步骤计算总结果并返回统一响应。"""
            result["passed"] = all(step["ok"] for step in steps)
            return ok(result)

        try:
            payload = await read_json()
        except JsonBodyError:
            payload = {}
        # r3.3：自检（含 deep 模式会写临时记忆）只对开发者模式开放。
        denied = self._require_dev(payload if isinstance(payload, dict) else None)
        if denied:
            return fail(denied)
        deep = bool(payload.get("deep")) if isinstance(payload, dict) else False
        deep = deep or bool(query_get("deep", 0, int))
        result: dict[str, Any] = {
            "passed": False,
            "mode": "deep" if deep else "shallow",
            "steps": steps,
        }

        store = plugin.memory_store
        if store is None:
            add("记忆库初始化", False, "memory_store 为 None")
            return finish()
        # 只回传文件名：绝对路径属于服务器内部信息（F43）。
        add("记忆库初始化", True, "memory_rag.db")

        try:
            stats = await asyncio.to_thread(store.get_stats)
        except Exception as exc:
            add("记忆库可读", False, str(exc))
        else:
            try:
                total = int((stats or {}).get("total_memories") or 0)
            except (AttributeError, TypeError, ValueError):
                total = 0
            add("记忆库可读", True, f"共 {total} 条记忆")

        fts_enabled = bool(getattr(store, "fts_enabled", False))
        add(
            "FTS 索引",
            True,
            "已启用" if fts_enabled else "未启用（词法检索走兼容路径）",
        )

        provider = plugin._embedding_provider()
        if provider is None:
            add("Embedding 提供商", False, "未找到可用的 embedding provider")
        else:
            provider_id, model = plugin._provider_meta_label(provider)
            add("Embedding 提供商", True, model or provider_id or "unknown")

        if not deep or provider is None:
            return finish()

        try:
            vector = await provider.get_embedding("ActiveReply 记忆自检")
        except Exception as exc:
            add("生成向量", False, str(exc))
            return finish()
        if not vector:
            add("生成向量", False, "返回空向量")
            return finish()
        add("生成向量", True, f"dim={len(vector)}")

        memory_id: int | None = None
        try:
            memory_id = await asyncio.to_thread(
                store.add_memory,
                content="ActiveReply 记忆自检",
                embedding=vector,
                role_ids=["selftest"],
                memory_time=time.time(),
                group_scope="selftest",
                group_id="selftest",
                platform_id="selftest",
                extra_metadata={"selftest": True},
            )
            add("写入记忆", True, f"id={memory_id}")
        except Exception as exc:
            add("写入记忆", False, str(exc))
            return finish()

        try:
            got = await asyncio.to_thread(store.get_memory, memory_id or 0)
            add(
                "读回记忆",
                bool(got and got.get("content") == "ActiveReply 记忆自检"),
                "内容一致" if got else "读回为空",
            )

            hits = await asyncio.to_thread(
                store.search_memories,
                query_embedding=vector,
                recall_k=5,
                group_scope="selftest",
                max_results=5,
            )
            found = any(int(hit["memory_id"]) == int(memory_id or 0) for hit in hits)
            add("语义检索", found, f"返回 {len(hits)} 条，命中={found}")
        except Exception as exc:
            add("读回/检索", False, str(exc))
        finally:
            try:
                removed = await asyncio.to_thread(store.delete_memory, memory_id or 0)
                add("清理自检数据", removed, f"删除 id={memory_id}")
            except Exception as exc:
                add("清理自检数据", False, str(exc))

        # 提示词自检与存储自检合并成一次「自检」，结果单独成段，互不影响成败判定。
        try:
            prompt_result = await plugin.run_prompt_test("group")
        except Exception as exc:
            prompt_result = {"ok": False, "error": str(exc)}
        result["prompt"] = {
            "ok": bool(prompt_result.get("ok")),
            "error": str(prompt_result.get("error") or ""),
            "version": prompt_result.get("prompt_version"),
            "model": prompt_result.get("model"),
            "parser": prompt_result.get("parser"),
            "elapsed_ms": prompt_result.get("elapsed_ms"),
            "items": prompt_result.get("items") or [],
            "evidence_ok": prompt_result.get("evidence_ok"),
            "raw": str(prompt_result.get("raw") or "")[:2000],
        }
        return finish()

    # ------------------------------------------------------------------ #
    # 用户屏蔽
    # ------------------------------------------------------------------ #
    @staticmethod
    def _block_view(entry: dict[str, Any]) -> dict[str, Any]:
        item = dict(entry)
        item["until_text"] = format_remaining(PageApi._safe_until(entry.get("until")))
        item["scope_text"] = (
            "全部会话（含私聊）" if entry.get("scope") == "all" else "仅本群"
        )
        return item

    async def blocks(self) -> dict[str, Any]:
        plugin = self.plugin
        cfg = plugin._global_cfg()
        group_id = str(query_get("group_id", "", str)).strip()
        # 屏蔽记录保存在 JSON 文件中，读取/同步同样走线程池，避免文件 IO 与
        # 锁等待阻塞事件循环（与记忆接口的处理方式保持一致）。
        try:
            await asyncio.to_thread(
                plugin.block_store.sync_from_config, cfg.user_block.blacklist
            )
        except Exception as exc:
            logger.warning(f"[ActiveReply] 同步屏蔽配置失败: {exc}")
        global_entries = await asyncio.to_thread(plugin.block_store.global_entries)
        group_entries = (
            await asyncio.to_thread(plugin.block_store.group_entries, group_id)
            if group_id
            else []
        )
        return ok(
            {
                "enabled": bool(cfg.user_block.enable),
                "hint": bool(cfg.user_block.block_hint),
                "global": [self._block_view(item) for item in global_entries],
                "group": [self._block_view(item) for item in group_entries],
                "group_id": group_id,
                "defaults": [
                    {
                        "user_id": uid,
                        "reason": default_reason(uid),
                    }
                    for uid in cfg.user_block.blacklist
                ],
            }
        )

    async def blocks_add(self) -> dict[str, Any]:
        payload, error = await _read_payload()
        if error:
            return fail(error)
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id or not user_id.isalnum() or not user_id.isascii():
            return fail("用户 ID 只能包含英文字母和数字")
        reason = str(payload.get("reason") or "").strip()
        scope = "group" if str(payload.get("scope") or "").lower() == "group" else "all"
        group_id = self._valid_group_id(payload.get("group_id"))
        if scope == "group" and not group_id:
            return fail("群号无效：只允许纯数字群号或「平台:群号」形式")
        until = parse_until(str(payload.get("duration") or "0"))
        if until is None:
            return fail("时间格式不正确：0 表示永久，或使用 1d2h / 30m / 60（分钟）")
        try:
            entry = self.plugin.block_store.add(
                user_id,
                reason=reason,
                scope=scope,
                until=until,
                group_id=group_id if scope == "group" else "",
                source="page",
            )
        except Exception as exc:
            return fail(f"添加屏蔽失败：{exc}")
        if scope == "all":
            self.plugin._sync_blacklist_config()
        return ok({"entry": self._block_view(entry)})

    async def blocks_remove(self) -> dict[str, Any]:
        payload, error = await _read_payload()
        if error:
            return fail(error)
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id or not user_id.isalnum() or not user_id.isascii():
            return fail("用户 ID 只能包含英文字母和数字")
        scope = "group" if str(payload.get("scope") or "").lower() == "group" else "all"
        group_id = self._valid_group_id(payload.get("group_id"))
        if scope == "group" and not group_id:
            return fail("群号无效：只允许纯数字群号或「平台:群号」形式")
        if scope == "all":
            removed = self.plugin.block_store.remove_all_for_user(user_id)
            self.plugin._sync_blacklist_config()
        else:
            removed = self.plugin.block_store.remove(
                user_id, scope="group", group_id=group_id
            )
        if not removed:
            return fail("未找到对应的屏蔽记录")
        return ok({"removed": True, "user_id": user_id})

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def _provider_options(self) -> list[dict[str, str]]:
        """生成按提供商 ID 排序的文本模型选择项。"""
        # 生成配置页面可用的文本模型提供商下拉选项并按 ID 排序。
        options: list[dict[str, str]] = []
        try:
            providers = self.plugin.context.get_all_providers() or []
        except Exception:
            return options
        for provider in providers:
            provider_id = ""
            model = ""
            try:
                meta = provider.meta()
                provider_id = str(getattr(meta, "id", "") or "")
                model = str(getattr(meta, "model", "") or "")
            except Exception:
                provider_id = str(getattr(provider, "provider_id", "") or "")
            if not provider_id:
                continue
            options.append(
                {
                    "id": provider_id,
                    "label": f"{provider_id}（{model}）" if model else provider_id,
                }
            )
        options.sort(key=lambda item: item["id"])
        return options
