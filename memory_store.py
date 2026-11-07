"""记忆 RAG 的 SQLite 存储层。

设计取舍：
* 向量以 JSON 文本落库，检索时在进程内计算余弦相似度；
* 单次检索最多扫描 ``default_scan_limit`` 条最新记录，避免全表读取；
* 角色与记忆是多对多关系，单独建表并开启外键级联删除；
* 所有查询参数化，不拼接外部输入。

数据文件位于 ``data/plugin_data/astrbot_plugin_active_reply/memory_rag.db``。
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from array import array
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .constants import IMPORTANCE_DEFAULT, IMPORTANCE_MAX
from .memory_scope import ScopeFilter, is_private_scope
from .vector_math import cosine_scores

DEFAULT_SCAN_LIMIT = 5000
# 相似记忆扫描窗口：沿用历史默认值，避免多候选把每次写入扫描量放大，
# 避免为了「多候选」而把每次写入的扫描量悄悄放大。
_SIMILAR_SCAN_LIMIT = 2000
# 数据版本号：启动时的全表 backfill（作用域归一 + 角色回填）只跑一次，
# 版本一致时直接跳过，避免每次启动都全表扫描。新增迁移时递增。
# v2：清空历史 ``superseded_by``（M-01 取消取代语义后，旧数据不再隐藏）；
# 迁移前会备份 DB，失败时不推进版本、下次启动重试。
_DATA_VERSION = 2
_DATA_VERSION_KEY = "data_version"
# F04：v2 迁移曾把 ``qq:private:u1`` 归一成 ``qq:u1``，把私聊标记永久丢掉。
# 修复后的作用域归一由独立版本号控制：已经跑过旧迁移的 r3.1 库也要再扫一次，
# 但不需要因此递增 ``_DATA_VERSION``（否则会重复执行 superseded 迁移）。
_SCOPE_PRIVATE_VERSION = 1
_SCOPE_PRIVATE_VERSION_KEY = "scope_private_version"

# 存储版本：数据库结构性兼容版本，与 _DATA_VERSION（数据回填版本）分开。
# 每次改变表结构/语义时递增；高于当前插件的数据库只告警并跳过迁移，避免降级写坏。
STORE_VERSION = 3
_STORE_VERSION_KEY = "store_version"
_DAILY_BACKUP_KEEP = 7

# F02：角色 ID 白名单（与计划 §4 的跨文件契约一致），最多保留 32 个。
_ROLE_ID_RE = re.compile(r"^[A-Za-z0-9_.:@\-]{1,64}$")
_MAX_ROLE_IDS = 32

# F05：merge 覆盖旧正文前，把旧正文快照写进 extra_metadata，最多保留最近 3 条。
_PREVIOUS_CONTENT_LIMIT = 3
_PREVIOUS_CONTENT_MAX_LEN = 500
_PREVIOUS_EVIDENCE_MAX_LEN = 200

logger = logging.getLogger(__name__)


def compute_effective_importance(
    *,
    importance: float,
    confidence: float | None = None,
    created_at: float,
    last_access: float = 0.0,
    access_count: int = 0,
    decay_reset_at: float = 0.0,
    now: float | None = None,
    decay_rate: float = 0.01,
    decay_floor: float = 0.5,
    access_window_days: float = 30.0,
    access_max_count: int = 10,
    protected: float = IMPORTANCE_MAX,
) -> float:
    """计算记忆在查询时刻的「有效重要性」（r3.3，不写库）。

    以 LivingMemory 的近似公式做时间演化：从 ``decay_reset_at``（恢复归档时
    重置）或 ``created_at`` 起按指数衰减；近期被访问过、访问次数多的记忆衰减
    更慢（访问越多越「抗遗忘」）。结果夹在 ``[decay_floor, 10.0]`` 之间；
    ``importance >= protected`` 的记忆视为受保护，直接返回原值。

    Args:
        importance: 写入时记录的原始重要性。
        confidence: 兼容参数，当前公式不参与计算（保留给调用方对齐口径）。
        created_at: 记忆创建时间（Unix 秒）。
        last_access: 最近一次被召回的时间；0 表示从未访问。
        access_count: 累计访问次数。
        decay_reset_at: 衰减基准重置时间；非 0 时优先于 ``created_at``。
        now: 计算时刻，默认当前时间。
        decay_rate: 每天的重要性衰减率。
        decay_floor: 有效重要性下限。
        access_window_days: 判定「近期访问」的窗口天数。
        access_max_count: 访问次数的饱和上限。
        protected: 受保护阈值，达到或超过时不再衰减。
    """
    del confidence  # 保留该参数只为与调用方口径对齐，当前公式不使用。
    try:
        base = float(importance)
    except (TypeError, ValueError):
        base = IMPORTANCE_DEFAULT
    base = max(0.0, min(IMPORTANCE_MAX, base))
    if base >= max(0.0, min(IMPORTANCE_MAX, float(protected))):
        return base
    current = time.time() if now is None else float(now)
    try:
        anchor = float(decay_reset_at or 0.0) or float(created_at or 0.0)
    except (TypeError, ValueError):
        anchor = 0.0
    if anchor <= 0:
        anchor = current
    age_days = max(0.0, (current - anchor) / 86400.0)
    window_sec = max(1.0, float(access_window_days)) * 86400.0
    recent = 1.0 if current - float(last_access or 0.0) <= window_sec else 0.5
    try:
        hits = max(0.0, float(access_count))
        max_hits = max(1.0, float(access_max_count))
        access_factor = min(1.0, hits / max_hits)
    except (TypeError, ValueError):
        access_factor = 0.0
    rate = max(0.0, min(0.999, float(decay_rate)))
    rate *= 1.0 - 0.5 * access_factor * recent
    value = base * (1.0 - rate) ** age_days
    floor = max(0.0, min(IMPORTANCE_MAX, float(decay_floor)))
    return max(floor, min(IMPORTANCE_MAX, value))


class MemoryStore:
    """记忆记录与角色标签的 SQLite 持久化存储。

    记忆正文、时间、作用域、来源元数据和 embedding 存在 ``memories`` 主表，
    每条记忆关联的角色 ID 单独放入 ``memory_roles``。向量按 JSON 保存，
    检索时只扫描满足结构化条件的一段最新记录，再在 Python 中计算余弦相似度；
    该实现不依赖额外向量数据库，但扫描上限意味着语义检索并非全库精确索引。
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        default_scan_limit: int = DEFAULT_SCAN_LIMIT,
        display_timezone: str = "Asia/Shanghai",
    ) -> None:
        """初始化 SQLite 记忆存储、展示时区和兼容性迁移。

        Args:
            db_path: SQLite 数据库文件路径。
            default_scan_limit: 语义检索默认扫描的最新记录数上限。
            display_timezone: ISO 时间字段使用的时区名称。
        """
        # 建立数据库目录与 schema，随后将历史群范围格式迁移到规范形式。
        self.db_path = Path(db_path)
        self.default_scan_limit = max(1, int(default_scan_limit))
        self._lock = threading.RLock()
        self._timezone_name = "Asia/Shanghai"
        self._tzinfo: Any = timezone.utc
        # FTS5 可用性在 _init_schema 里探测；不可用时词法通道自动降级为空结果。
        self.fts_enabled = False
        # 词法查询失败只告警一次，避免坏索引/超时把日志刷满（F28）。
        self._lexical_error_logged = False
        # 数据库版本高于插件时置 False：只告警并跳过迁移，避免降级写坏新库。
        self.store_compatible = True
        self.store_version = 0
        self.set_display_timezone(display_timezone)
        self._init_schema()
        # 先做每日快照，再执行任何版本检查/迁移；备份失败不阻塞启动。
        self._daily_backup()
        self._ensure_store_version()
        # L-09：只有数据版本落后时才做全表 backfill；正常启动不再扫描整表。
        self.ensure_backfills()

    # ------------------------------------------------------------------ #
    # 基础设施
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        """创建配置好行映射、外键与忙等待的数据库连接。

        每次操作独立连接，并开启外键以保证删除记忆时同步删除角色关联。
        显式设置 ``busy_timeout``：并发写入短暂排队 3 秒再报错，而不是立刻抛
        ``database is locked``；``synchronous=NORMAL`` 与 WAL 配合降低提交延迟
        （断电时可能丢最后若干事务，但不会损坏数据库）。
        """
        conn = sqlite3.connect(str(self.db_path), timeout=3.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 3000")
        conn.execute("PRAGMA foreign_keys = ON")
        # 数据库版本高于插件时只允许读，防止降级写入破坏新结构。
        if not self.store_compatible:
            conn.execute("PRAGMA query_only = ON")
        try:
            conn.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.DatabaseError:  # pragma: no cover - 只读库等极端环境
            # 同步级别设置失败不影响读写，保持默认级别即可。
            logger.debug("[ActiveReply] 无法设置 SQLite synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        """确保记忆表、角色关联表及检索所需索引存在。"""
        # 创建记忆主表、角色关联表及常用筛选索引；已有数据库保持兼容。
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, closing(self._connect()) as conn, conn:
            # PRAGMA 必须在任何 DDL 开启事务之前执行，否则切 WAL 会失败。
            self._apply_schema_pragmas(conn)
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    memory_time REAL NOT NULL,
                    group_scope TEXT NOT NULL DEFAULT '',
                    group_id TEXT NOT NULL DEFAULT '',
                    platform_id TEXT NOT NULL DEFAULT '',
                    extra_metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_roles (
                    memory_id INTEGER NOT NULL,
                    role_id TEXT NOT NULL,
                    PRIMARY KEY (memory_id, role_id),
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_memories_time ON memories(memory_time);
                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(group_scope);
                CREATE INDEX IF NOT EXISTS idx_memories_group ON memories(group_id);
                CREATE INDEX IF NOT EXISTS idx_memories_platform ON memories(platform_id);
                CREATE INDEX IF NOT EXISTS idx_memories_content ON memories(content);
                CREATE INDEX IF NOT EXISTS idx_roles_role ON memory_roles(role_id);
                CREATE INDEX IF NOT EXISTS idx_roles_memory ON memory_roles(memory_id);

                CREATE TABLE IF NOT EXISTS memory_trash (
                    trash_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    roles TEXT NOT NULL DEFAULT '[]',
                    deleted_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plugin_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT ''
                );
                """
            )
            self._ensure_columns(conn)
            # 列补齐后再建召回复合索引；旧库升级时同样适用。
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_recall "
                "ON memories(archived, superseded_by, memory_time DESC)"
            )
            self._ensure_fts(conn)

    def _apply_schema_pragmas(self, conn: sqlite3.Connection) -> None:
        """尝试把数据库切到 WAL 并放宽同步级别。

        网络盘、只读目录或受限文件系统上 PRAGMA 可能失败；此时只记 warning
        并继续，插件仍以回滚日志模式启动，不能让启动路径被阻塞。
        """
        for pragma, value in (("journal_mode", "WAL"), ("synchronous", "NORMAL")):
            try:
                conn.execute(f"PRAGMA {pragma} = {value}")
            except sqlite3.DatabaseError as exc:
                logger.warning(
                    "[ActiveReply] SQLite PRAGMA %s=%s 设置失败，继续使用默认模式: %s",
                    pragma,
                    value,
                    exc,
                )

    _EXTRA_COLUMNS = {
        "memory_type": "TEXT NOT NULL DEFAULT 'group'",
        "sender_id": "TEXT NOT NULL DEFAULT ''",
        "sender_role": "TEXT NOT NULL DEFAULT 'member'",
        "source_msg_id": "TEXT NOT NULL DEFAULT ''",
        "source_time": "REAL NOT NULL DEFAULT 0",
        "importance": "REAL NOT NULL DEFAULT 5.0",
        "last_access": "REAL NOT NULL DEFAULT 0",
        "access_count": "INTEGER NOT NULL DEFAULT 0",
        "superseded_by": "INTEGER DEFAULT NULL",
        "archived": "INTEGER NOT NULL DEFAULT 0",
        # 向量改用二进制存储，避免每次召回都 json.loads 数万字符；JSON 列保留给
        # 旧库与回收站兼容，读取时优先用 BLOB。
        "embedding_blob": "BLOB",
        "embedding_norm": "REAL",
        # 中文 bigram 文本，供 FTS5 词法通道使用。
        "grams": "TEXT",
        # 原子记忆维度：类型、实体、有效期、置信度、抽取批次。
        "kind": "TEXT NOT NULL DEFAULT 'fact'",
        "entities": "TEXT NOT NULL DEFAULT '[]'",
        "valid_until": "REAL DEFAULT NULL",
        "confidence": "REAL NOT NULL DEFAULT 0.5",
        "source_batch": "TEXT NOT NULL DEFAULT ''",
        # r3.3：取消归档/回收站恢复时重置的重要性衰减基准时间。
        "decay_reset_at": "REAL DEFAULT NULL",
        # r3.3k：归档原因（manual/expired/stale/consolidation），供控制台展示遗忘原因。
        "archive_reason": "TEXT NOT NULL DEFAULT ''",
        # r3.3m：独立记录归档时间，避免与重要性衰减锚点 decay_reset_at 语义耦合。
        "archived_at": "REAL DEFAULT NULL",
    }

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        """为旧版数据库补齐新增字段（幂等）。"""
        existing = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        for name, ddl in self._EXTRA_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {ddl}")

    # ------------------------------------------------------------------ #
    # 词法索引（FTS5）与二进制向量
    # ------------------------------------------------------------------ #
    _GRAM_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")

    @classmethod
    def _build_grams(cls, text: str) -> str:
        """把正文切成「英文词 + 中文 bigram」的检索词串。

        FTS5 的 trigram 分词对两个汉字的中文查询不生效（实测 ``火锅`` 匹配不到），
        因此这里自己生成 bigram，用 unicode61 分词器索引，2 字查询也能命中。
        """
        tokens: list[str] = []
        for match in cls._GRAM_TOKEN_RE.finditer(str(text or "").lower()):
            segment = match.group(0)
            if segment[0].isascii():
                if len(segment) >= 2:
                    tokens.append(segment)
                continue
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        return " ".join(dict.fromkeys(tokens))

    @staticmethod
    def _encode_vector(vector: list[float]) -> bytes:
        """把浮点向量编码为小端 float32 BLOB。"""
        return array("f", [float(value) for value in vector]).tobytes()

    @staticmethod
    def _decode_vector(raw: Any) -> list[float]:
        """把 BLOB 还原成浮点列表；格式不符时返回空列表。"""
        if not raw:
            return []
        try:
            values = array("f")
            values.frombytes(bytes(raw))
        except (ValueError, TypeError):
            return []
        return [float(value) for value in values]

    def _row_embedding(self, row: sqlite3.Row) -> list[float]:
        """读取一行记录的记忆向量：优先 BLOB，回退旧版 JSON。"""
        try:
            keys = row.keys()
        except AttributeError:
            keys = []
        if "embedding_blob" in keys:
            vector = self._decode_vector(row["embedding_blob"])
            if vector:
                return vector
        return self._parse_embedding(row["embedding"])

    def _ensure_fts(self, conn: sqlite3.Connection) -> None:
        """建立记忆全文索引与同步触发器；FTS5 不可用时安静降级。

        外部内容表（content='memories'）只存索引，正文仍以主表为准；触发器
        保证插入 / 更新 / 删除时索引与主表一致，``rebuild`` 用于整体重建。
        """
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
                "content, grams, content='memories', content_rowid='id', "
                "tokenize='unicode61')"
            )
        except sqlite3.DatabaseError:  # pragma: no cover - 依赖 SQLite 构建
            self.fts_enabled = False
            return
        self.fts_enabled = True
        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS memory_fts_ai AFTER INSERT ON memories BEGIN
              INSERT INTO memory_fts(rowid, content, grams)
              VALUES (new.id, new.content, coalesce(new.grams, ''));
            END;
            CREATE TRIGGER IF NOT EXISTS memory_fts_ad AFTER DELETE ON memories BEGIN
              INSERT INTO memory_fts(memory_fts, rowid, content, grams)
              VALUES ('delete', old.id, old.content, coalesce(old.grams, ''));
            END;
            CREATE TRIGGER IF NOT EXISTS memory_fts_au
            AFTER UPDATE OF content, grams ON memories BEGIN
              INSERT INTO memory_fts(memory_fts, rowid, content, grams)
              VALUES ('delete', old.id, old.content, coalesce(old.grams, ''));
              INSERT INTO memory_fts(rowid, content, grams)
              VALUES (new.id, new.content, coalesce(new.grams, ''));
            END;
            """
        )
        self._backfill_grams(conn)
        # F28：索引被清空 / 触发器缺失 / 文件损坏时自动重建，避免词法通道永久为空。
        try:
            self._verify_fts_index(conn)
        except Exception as exc:  # 自检失败只告警，绝不能阻塞启动
            logger.warning("[ActiveReply] FTS 索引自检失败，本次启动跳过: %s", exc)

    def _verify_fts_index(self, conn: sqlite3.Connection) -> None:
        """自检 FTS 索引与主表是否一致，不一致时重建并告警（F28）。

        外部内容表（``content='memories'``）的 ``SELECT COUNT(*) FROM memory_fts``
        实际会去读内容表，行数永远等于 ``memories``，不能作为索引自检依据；
        这里改用 ``memory_fts_docsize`` 影子表的行数（每个被索引的行一条），
        清空索引后会变 0。行数一致时再用 FTS5 的 ``integrity-check`` 兜底。
        """
        memories_count = int(
            conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        )
        fts_count = self._fts_index_row_count(conn)
        if fts_count is None or fts_count != memories_count:
            logger.warning(
                "[ActiveReply] FTS 索引与记忆表不一致（索引 %s 行 / 记忆 %s 行），"
                "正在重建",
                fts_count,
                memories_count,
            )
            self._rebuild_fts(conn, memories_count)
            return
        try:
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('integrity-check')")
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "[ActiveReply] FTS 索引完整性校验失败（可能已损坏），正在重建: %s", exc
            )
            self._rebuild_fts(conn, memories_count)

    @staticmethod
    def _fts_index_row_count(conn: sqlite3.Connection) -> int | None:
        """返回 FTS5 索引实际行数；无法读取时返回 None（交给 rebuild 兜底）。"""
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM memory_fts_docsize"
            ).fetchone()
            if row is not None:
                return int(row["cnt"])
        except (sqlite3.DatabaseError, TypeError, ValueError):
            pass
        try:
            # 极少数没有 docsize 影子表的构建：退回文档表 count（仅作参考）。
            row = conn.execute("SELECT COUNT(*) AS cnt FROM memory_fts").fetchone()
            if row is not None:
                return int(row["cnt"])
        except (sqlite3.DatabaseError, TypeError, ValueError):
            return None
        return None

    def _rebuild_fts(self, conn: sqlite3.Connection, expected: int) -> None:
        """重建 FTS 索引；失败时降级为不可用并告警，不向调用方抛错。"""
        try:
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
            repaired = self._fts_index_row_count(conn)
        except sqlite3.DatabaseError as exc:
            self.fts_enabled = False
            logger.error(
                "[ActiveReply] FTS 索引重建失败，词法检索降级为空: %s", exc
            )
            return
        if repaired != expected:
            logger.warning(
                "[ActiveReply] FTS 索引重建后仍有差异（索引 %s 行 / 记忆 %s 行）",
                repaired,
                expected,
            )
        else:
            logger.info("[ActiveReply] FTS 索引已重建 | rows=%s", repaired)

    def _backfill_grams(self, conn: sqlite3.Connection, batch: int = 500) -> int:
        """为旧数据补算 bigram 文本；只有真的补了数据才重建索引。

        旧实现单次只处理 5000 行，库更大时要重启多次才能补齐；这里循环分批，
        一次启动就能完成，同时避免每次启动都做无谓的全量 rebuild。
        """
        size = max(1, int(batch))
        filled = 0
        while True:
            rows = conn.execute(
                "SELECT id, content FROM memories WHERE grams IS NULL LIMIT ?",
                (size,),
            ).fetchall()
            if not rows:
                break
            conn.executemany(
                "UPDATE memories SET grams = ? WHERE id = ?",
                [(self._build_grams(row["content"]), int(row["id"])) for row in rows],
            )
            filled += len(rows)
            if len(rows) < size:
                break
        if filled:
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
        return filled

    _MESSAGE_TYPE_SEGMENTS = {
        "group",
        "groupmessage",
        "friend",
        "friendmessage",
        "private",
    }
    # F04：私聊消息段必须保留 ``:private:`` 标记，不能再归一成 ``平台:用户ID``。
    _PRIVATE_MESSAGE_SEGMENTS = frozenset({"friend", "friendmessage", "private"})

    def normalize_group_scopes(self) -> int:
        """归一遍历史作用域格式，并修复丢失的私聊标记（F04）。

        * ``platform:GroupMessage:123`` → ``platform:123``（群作用域）；
        * ``platform:FriendMessage:u1`` / ``platform:private:u1`` → ``platform:private:u1``
          （私聊作用域，保留 ``:private:`` 供 ``memory_scope.is_private_scope`` 识别）；
        * ``memory_type='private'`` 但作用域缺少 ``:private:`` 的旧行，用发送者 ID
          兜底补成私聊作用域（v2 迁移丢标记后的补救）。

        群作用域与既有群记忆保持原语义；只更新真正需要变化的行。
        """
        changed = 0
        with self._lock, closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT id, group_scope, group_id, platform_id, memory_type, "
                "sender_id FROM memories"
            ).fetchall()
            for row in rows:
                memory_id = int(row["id"])
                scope = str(row["group_scope"] or "").strip()
                current_group = str(row["group_id"] or "")
                current_platform = str(row["platform_id"] or "")
                memory_type = str(row["memory_type"] or "group").strip().lower()
                new_scope: str | None = None
                new_group = current_group
                new_platform = current_platform
                new_type = memory_type

                if not scope:
                    # 私聊类型却没有作用域：能拿到发送者时补全，其它保持原样。
                    sender = str(row["sender_id"] or "").strip()
                    if memory_type == "private" and sender and current_platform:
                        new_scope = f"{current_platform}:private:{sender}"
                else:
                    parts = scope.split(":")
                    segment = parts[1].strip().lower() if len(parts) >= 3 else ""
                    platform = parts[0].strip()
                    tail = parts[-1].strip()
                    if segment in self._PRIVATE_MESSAGE_SEGMENTS:
                        # F04：friend/private 段统一规范成 ``平台:private:用户ID``。
                        if platform and tail:
                            new_scope = f"{platform}:private:{tail}"
                            new_platform = platform
                            # 私聊记忆不能挂在群号下，避免按群号过滤时串味。
                            new_group = ""
                            if new_type not in {"personal", "private"}:
                                new_type = "private"
                    elif segment in self._MESSAGE_TYPE_SEGMENTS:
                        # 群作用域：``平台:GroupMessage:群号`` → ``平台:群号``。
                        if platform and tail:
                            new_scope = f"{platform}:{tail}"
                        elif tail:
                            new_scope = tail
                        if new_scope is not None:
                            new_group = current_group or tail
                            new_platform = current_platform or platform
                    elif memory_type == "private" and not is_private_scope(scope):
                        # F04 兜底：v2 迁移把私聊行写成了 ``平台:用户ID``。
                        if len(parts) >= 2 and tail and platform:
                            new_scope = f"{platform}:private:{tail}"
                            new_platform = platform
                            new_group = ""

                if new_scope is None:
                    continue
                if (
                    new_scope == scope
                    and new_group == current_group
                    and new_platform == current_platform
                    and new_type == memory_type
                ):
                    continue
                conn.execute(
                    "UPDATE memories SET group_scope = ?, group_id = ?, "
                    "platform_id = ?, memory_type = ? WHERE id = ?",
                    (new_scope, new_group, new_platform, new_type, memory_id),
                )
                changed += 1
        return changed

    def _ensure_private_scope_repair(self) -> None:
        """对已经跑过 v2 迁移的库补做一次私聊作用域修复（F04，幂等）。

        r3.1 的 ``data_version`` 已经是 2，``ensure_backfills`` 不会再跑全表
        归一；这里用独立版本号，只补扫一次，不改动 ``data_version`` 契约。
        """
        try:
            raw = self.get_meta(_SCOPE_PRIVATE_VERSION_KEY)
            if int(raw or 0) >= _SCOPE_PRIVATE_VERSION:
                return
        except (TypeError, ValueError):
            pass
        try:
            changed = self.normalize_group_scopes()
            self.set_meta(_SCOPE_PRIVATE_VERSION_KEY, str(_SCOPE_PRIVATE_VERSION))
        except Exception as exc:  # 修复失败不阻塞启动，下次再试
            logger.warning("[ActiveReply] 私聊作用域修复失败（下次启动重试）: %s", exc)
            return
        if changed:
            logger.info("[ActiveReply] 私聊作用域修复完成 | updated=%s", changed)

    def backfill_roles_from_sender(self) -> int:
        """给没有角色关联的旧记忆补上写入时的发送者（幂等）。

        可见性规则依赖 ``memory_roles`` 判断“本人参与”；v1.1 之前写入的全局
        记忆可能只有 ``sender_id`` 没有角色行，补不上的仍按不可跨群处理，
        宁可少召回，也不把别人的记忆带到别的会话。
        """
        with self._lock, closing(self._connect()) as conn, conn:
            before = conn.total_changes
            conn.execute(
                "INSERT INTO memory_roles(memory_id, role_id) "
                "SELECT m.id, m.sender_id FROM memories AS m "
                "WHERE m.sender_id != '' AND NOT EXISTS ("
                "SELECT 1 FROM memory_roles AS r WHERE r.memory_id = m.id)"
            )
            return int(conn.total_changes - before)

    # ------------------------------------------------------------------ #
    # 启动期迁移标记（L-09）
    # ------------------------------------------------------------------ #
    def get_meta(self, key: str) -> str | None:
        """读取 ``plugin_meta`` 中的一项；键为空或不存在时返回 None。"""
        name = str(key or "").strip()
        if not name:
            return None
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT value FROM plugin_meta WHERE key = ?", (name,)
            ).fetchone()
        return str(row["value"]) if row is not None else None

    def set_meta(self, key: str, value: str) -> bool:
        """写入或覆盖一项 ``plugin_meta``；键为空时返回 False。"""
        name = str(key or "").strip()
        if not name:
            return False
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO plugin_meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (name, str(value)),
            )
        return True

    def needs_backfill(self) -> bool:
        """数据版本是否落后于 ``_DATA_VERSION``（需要跑一次全表 backfill）。"""
        raw = self.get_meta(_DATA_VERSION_KEY)
        try:
            return int(raw or 0) < _DATA_VERSION
        except (TypeError, ValueError):
            return True

    def mark_backfilled(self) -> None:
        """记录当前数据版本，后续启动直接跳过全表 backfill。"""
        self.set_meta(_DATA_VERSION_KEY, str(_DATA_VERSION))

    def _data_version(self) -> int:
        """读取当前数据版本；缺失或非法值按 0（最旧版本）处理。"""
        raw = self.get_meta(_DATA_VERSION_KEY)
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    def _ensure_store_version(self) -> None:
        """记录/检查数据库存储版本；高于插件版本时跳过迁移。"""
        try:
            stored = int(self.get_meta(_STORE_VERSION_KEY) or 0)
        except (TypeError, ValueError):
            stored = 0
        if stored > STORE_VERSION:
            self.store_compatible = False
            self.store_version = stored
            logger.error(
                "[ActiveReply] 记忆库版本高于插件 | db=%s plugin=%s；已跳过迁移，"
                "请升级插件，避免降级写入",
                stored,
                STORE_VERSION,
            )
            return
        self.store_version = STORE_VERSION
        if stored < STORE_VERSION:
            self.set_meta(_STORE_VERSION_KEY, str(STORE_VERSION))
            logger.info(
                "[ActiveReply] 记忆库存储版本 %s → %s", stored or "未记录", STORE_VERSION
            )

    def _daily_backup(self) -> Path | None:
        """每次插件初始化时，若当天快照不存在则做一份完整 SQLite 快照，保留最近 7 份。"""
        if not self.db_path.exists():
            return None
        backup_dir = self.db_path.parent / "backups"
        target = backup_dir / f"{self.db_path.stem}-{datetime.now().strftime('%Y%m%d')}.db"
        temp = target.with_suffix(target.suffix + ".tmp")
        if target.exists():
            return target
        try:
            with closing(self._connect()) as source:
                count = int(
                    source.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                )
            if count <= 0:
                return None
            backup_dir.mkdir(parents=True, exist_ok=True)
            with (
                closing(self._connect()) as source,
                closing(sqlite3.connect(str(temp))) as backup,
            ):
                source.backup(backup)
                check = backup.execute("PRAGMA quick_check").fetchone()
                if not check or str(check[0]).lower() != "ok":
                    raise sqlite3.DatabaseError(f"quick_check={check!r}")
            os.replace(temp, target)
            backups = sorted(backup_dir.glob(f"{self.db_path.stem}-*.db"))
            for old in backups[:-_DAILY_BACKUP_KEEP]:
                try:
                    old.unlink()
                except OSError:
                    pass
            logger.info(
                "[ActiveReply] 记忆库每日备份完成 | %s（%s 条）",
                target.name,
                count,
            )
            return target
        except (sqlite3.DatabaseError, OSError) as exc:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            logger.warning("[ActiveReply] 记忆库每日备份失败，继续启动: %s", exc)
            return None

    def _backup_database(self, suffix: str) -> Path | None:
        """迁移前把数据库完整备份到同目录；失败返回 None（由调用方中止迁移）。"""
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        target = self.db_path.with_name(f"{self.db_path.name}.{suffix}.{stamp}.bak")
        try:
            with (
                closing(self._connect()) as source,
                closing(sqlite3.connect(str(target))) as backup,
            ):
                source.backup(backup)
        except (sqlite3.DatabaseError, OSError) as exc:
            logger.error("[ActiveReply] 记忆库迁移前备份失败: %s", exc)
            return None
        logger.info("[ActiveReply] 记忆库迁移前已备份 | %s", target.name)
        return target

    def _scale_original_importance(self, raw: Any) -> str | None:
        """把旧元数据中的原始重要性放大到 0–10；无需变更时返回 None。"""
        try:
            metadata = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(metadata, dict) or "original_importance" not in metadata:
            return None
        original = self._safe_float(metadata["original_importance"], IMPORTANCE_MAX)
        if original > 1.0:
            return None
        metadata["original_importance"] = min(
            IMPORTANCE_MAX, original * IMPORTANCE_MAX
        )
        return json.dumps(metadata, ensure_ascii=False)

    def _migrate_importance_scale(self) -> None:
        """把旧库的 0–1 重要性迁移为 0–10；失败时不写完成标记。"""
        if self.get_meta("importance_scale") == "10":
            return
        if self._backup_database("importance-scale") is None:
            logger.error("[ActiveReply] 记忆库备份失败，跳过重要性量表迁移")
            return
        try:
            with self._lock, closing(self._connect()) as conn, conn:
                conn.execute(
                    "UPDATE memories SET importance = MIN(?, importance * ?) "
                    "WHERE importance <= 1.0",
                    (IMPORTANCE_MAX, IMPORTANCE_MAX),
                )
                rows = conn.execute(
                    "SELECT id, extra_metadata FROM memories"
                ).fetchall()
                for row in rows:
                    metadata = self._scale_original_importance(row["extra_metadata"])
                    if metadata is not None:
                        conn.execute(
                            "UPDATE memories SET extra_metadata = ? WHERE id = ?",
                            (metadata, int(row["id"])),
                        )
                rows = conn.execute(
                    "SELECT trash_id, payload FROM memory_trash"
                ).fetchall()
                for row in rows:
                    try:
                        payload = json.loads(row["payload"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    changed = False
                    importance = self._safe_float(
                        payload.get("importance"), IMPORTANCE_MAX
                    )
                    if "importance" in payload and importance <= 1.0:
                        payload["importance"] = min(
                            IMPORTANCE_MAX, importance * IMPORTANCE_MAX
                        )
                        changed = True
                    metadata = self._scale_original_importance(
                        payload.get("extra_metadata")
                    )
                    if metadata is not None:
                        payload["extra_metadata"] = metadata
                        changed = True
                    if changed:
                        conn.execute(
                            "UPDATE memory_trash SET payload = ? WHERE trash_id = ?",
                            (
                                json.dumps(payload, ensure_ascii=False),
                                int(row["trash_id"]),
                            ),
                        )
            self.set_meta("importance_scale", "10")
        except Exception as exc:
            logger.error("[ActiveReply] 重要性量表迁移失败，下次启动重试: %s", exc)

    def clear_superseded(self) -> int:
        """清空历史 ``superseded_by`` 标记，返回更新的行数（v2 迁移）。"""
        with self._lock, closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE memories SET superseded_by = NULL "
                "WHERE superseded_by IS NOT NULL"
            )
            return int(cursor.rowcount or 0)

    def ensure_backfills(self) -> None:
        """数据版本落后时执行一次性迁移，并把版本标记为最新。

        幂等：重复调用在标记完成后直接返回。旧库（无 ``plugin_meta`` 记录）
        首次仍会执行，行为与升级前一致；之后每次启动不再扫描全表。
        v2 迁移失败时不推进版本号（下次启动重试），也不影响本次其它读写。
        数据库版本高于插件时跳过迁移，避免降级写坏新库。
        """
        if not self.store_compatible:
            logger.warning("[ActiveReply] 数据库版本不兼容，跳过记忆库迁移")
            return
        self._migrate_importance_scale()
        if not self.needs_backfill():
            # F04：data_version 已是 2 的 r3.1 库仍需要补做一次私聊作用域修复。
            self._ensure_private_scope_repair()
            return
        previous = self._data_version()
        try:
            if previous < 2:
                legacy_rows = 0
                if previous == 0:
                    # previous=0 既可能是全新库，也可能是没有 plugin_meta 的旧库：
                    # 有数据就要备份，只有真正空库才跳过（M1）。
                    with closing(self._connect()) as conn:
                        legacy_rows = int(
                            conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                        )
                # M-01 收尾：v1 写入的 superseded_by 已无对应语义，清空后旧记忆
                # 重新参与召回与去重；列与查询过滤保留作防御（半迁移库兜底）。
                # 升级库必须先有备份；备份失败中止迁移（不 UPDATE、不推进版本）。
                needs_backup = previous > 0 or legacy_rows > 0
                suffix = f"v{previous}" if previous > 0 else "legacy"
                if needs_backup and self._backup_database(suffix) is None:
                    logger.error(
                        "[ActiveReply] 记忆库备份失败，中止 v2 迁移（下次启动重试）"
                    )
                    return
                cleared = self.clear_superseded()
                logger.info(
                    "[ActiveReply] 记忆库 v2 迁移完成 | cleared_superseded=%s",
                    cleared,
                )
            # 后续步骤也必须受控：任一步抛错都不能让 MemoryStore 初始化失败。
            changed_scopes = self.normalize_group_scopes()
            added_roles = self.backfill_roles_from_sender()
            self.mark_backfilled()
            # 归一逻辑已包含 F04 的私聊修复，标记独立版本号避免重复全表扫描。
            self.set_meta(_SCOPE_PRIVATE_VERSION_KEY, str(_SCOPE_PRIVATE_VERSION))
        except Exception as exc:  # 迁移失败不阻塞启动，也不推进版本
            logger.error("[ActiveReply] 记忆库迁移失败，下次启动重试: %s", exc)
            return
        logger.info(
            "[ActiveReply] 记忆库一次性迁移完成 | data_version=%s 作用域归一=%s "
            "角色回填=%s",
            _DATA_VERSION,
            changed_scopes,
            added_roles,
        )

    def set_display_timezone(self, timezone_name: str) -> str:
        """设置时间展示使用的时区（与 AstrBot 全局 timezone 保持一致）。"""
        name = str(timezone_name or "").strip() or "Asia/Shanghai"
        try:
            tzinfo = ZoneInfo(name)
        except Exception:
            name = "Asia/Shanghai"
            try:
                tzinfo = ZoneInfo(name)
            except Exception:  # pragma: no cover - 极端环境兜底
                name = "UTC"
                tzinfo = timezone.utc
        self._timezone_name = name
        self._tzinfo = tzinfo
        return name

    def format_timestamp_iso(self, timestamp: float | int) -> str:
        """将 Unix 时间秒数转换为当前展示时区下的 ISO 8601 时间。

        F27：毫秒时间戳先归一成秒；非法/超出 ``datetime`` 范围的值回退到
        Unix 纪元，保证坏数据不会让列表/召回整条链抛 ``ValueError``。
        """
        # 将 Unix 秒时间戳按当前展示时区格式化为 ISO 8601 字符串。
        return self._format_iso_seconds(self._safe_timestamp(timestamp))

    def _format_iso_seconds(self, seconds: float) -> str:
        """把已经归一成秒的时间戳格式化为 ISO；越界值回退 Unix 纪元。"""
        try:
            dt = datetime.fromtimestamp(seconds, tz=self._tzinfo)
        except (OverflowError, OSError, ValueError):
            dt = datetime.fromtimestamp(0.0, tz=self._tzinfo)
        return dt.isoformat()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_role_ids(role_ids: list[str] | tuple[str, ...]) -> list[str]:
        """清理角色 ID，按输入顺序去重并做白名单过滤（F02）。

        控制台会把角色 ID 直接渲染进页面，写库前必须限制字符集与数量：
        只接受 ``^[A-Za-z0-9_.:@\\-]{1,64}$``，最多 32 个；非法值丢弃并告警
        （只记数量，不把恶意内容写进日志）。
        """
        # 清理空 ID 并按首次出现顺序去重，供关系表写入使用。
        result: list[str] = []
        seen: set[str] = set()
        rejected = 0
        for raw in role_ids or ():
            role_id = str(raw or "").strip()
            if not role_id or role_id in seen:
                continue
            if not _ROLE_ID_RE.fullmatch(role_id):
                rejected += 1
                continue
            seen.add(role_id)
            if len(result) >= _MAX_ROLE_IDS:
                rejected += 1
                continue
            result.append(role_id)
        if rejected:
            logger.warning(
                "[ActiveReply] 过滤非法/超限角色 ID %s 个（上限 %s 个，字符集白名单）",
                rejected,
                _MAX_ROLE_IDS,
            )
        return result

    @staticmethod
    def _parse_embedding(raw: str | None) -> list[float]:
        """将存储的 JSON 向量解析为浮点列表，解析失败时返回空列表。"""
        # 将数据库中的 JSON 向量还原；损坏或不符合数组格式时返回空向量。
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(parsed, list):
            return []
        vector: list[float] = []
        for item in parsed:
            try:
                vector.append(float(item))
            except (TypeError, ValueError):
                return []
        return vector

    @staticmethod
    def _parse_entities(raw: str | None) -> list[str]:
        """把实体 JSON 还原成字符串列表；损坏时返回空列表。"""
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()][:8]

    @staticmethod
    def _parse_metadata(raw: str | None) -> dict[str, Any]:
        """将扩展元数据 JSON 还原为字典，其他内容返回空字典。"""
        # 将扩展元数据安全还原为字典，未知或损坏内容按空元数据处理。
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """把任意列值转成有限浮点数；非法值回退默认值（F27）。"""
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return number if math.isfinite(number) else default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """把任意列值转成整数；非法值回退默认值（F27）。"""
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default

    @classmethod
    def _safe_timestamp(cls, value: Any, default: float = 0.0) -> float:
        """归一 Unix 时间戳：毫秒换算为秒、非法/非有限值回退默认值（F27）。"""
        number = cls._safe_float(value, default)
        if number > 1e12:  # 毫秒时间戳
            number /= 1000.0
        return number

    @staticmethod
    def _normalize_timestamp(raw: float | int) -> tuple[float, bool]:
        """把毫秒时间戳换算为秒，并返回换算后的数值和是否转换标记。"""
        # 兼容秒/毫秒两种 Unix 时间戳，第二个返回值表示是否执行了换算。
        value = float(raw)
        if value > 1e12:  # 毫秒时间戳
            return value / 1000.0, True
        return value, False

    def _time_fields(self, timestamp: Any) -> dict[str, Any]:
        """为记忆 API 生成时间戳、ISO 字符串及展示时区字段（F27 容错）。

        坏数据或毫秒时间戳不能让整表读取抛错：非有限 / 非法值回退 0，
        毫秒值先归一成秒，再从时间戳生成 ISO 字符串。
        """
        # 生成 API/页面共用的时间字段，明确携带展示时区名称。
        value = self._safe_timestamp(timestamp)
        return {
            "memory_time": value,
            "memory_time_iso": self._format_iso_seconds(value),
            "memory_time_timezone": self._timezone_name,
        }

    def _row_to_item(self, row: sqlite3.Row, role_ids: list[str]) -> dict[str, Any]:
        """将数据库行转换为包含角色和时区信息的普通字典。"""
        # 把 SQLite 行转换为不依赖 sqlite3.Row 的普通 API 字典。
        keys = row.keys()
        item: dict[str, Any] = {
            "memory_id": int(row["id"]),
            "content": str(row["content"]),
            "embedding_dim": self._safe_int(row["embedding_dim"], 0),
            "group_scope": str(row["group_scope"] or ""),
            "group_id": str(row["group_id"] or ""),
            "platform_id": str(row["platform_id"] or ""),
            "related_role_ids": role_ids,
            "extra_metadata": self._parse_metadata(row["extra_metadata"]),
            "created_at": self._safe_float(row["created_at"], 0.0),
            "updated_at": self._safe_float(row["updated_at"], 0.0),
            "memory_type": str(row["memory_type"] or "group")
            if "memory_type" in keys
            else "group",
            "sender_id": str(row["sender_id"] or "") if "sender_id" in keys else "",
            "sender_role": str(row["sender_role"] or "member")
            if "sender_role" in keys
            else "member",
            "source_msg_id": str(row["source_msg_id"] or "")
            if "source_msg_id" in keys
            else "",
            "source_time": self._safe_float(row["source_time"], 0.0)
            if "source_time" in keys
            else 0.0,
            # F54：0.0 是合法置信度/重要性，不能被 ``or`` 默认值吞掉。
            "importance": (
                IMPORTANCE_DEFAULT
                if "importance" not in keys or row["importance"] is None
                else self._safe_float(row["importance"], IMPORTANCE_DEFAULT)
            ),
            "last_access": self._safe_float(row["last_access"], 0.0)
            if "last_access" in keys
            else 0.0,
            "access_count": self._safe_int(row["access_count"], 0)
            if "access_count" in keys
            else 0,
            # r3.3：重要性衰减基准重置时间；旧库缺列时安全回落到 0。
            "decay_reset_at": (
                self._safe_float(row["decay_reset_at"], 0.0)
                if "decay_reset_at" in keys and row["decay_reset_at"] is not None
                else 0.0
            ),
            "superseded_by": (
                self._safe_int(row["superseded_by"], 0) or None
                if row["superseded_by"] is not None
                else None
            )
            if "superseded_by" in keys
            else None,
            "archived": self._safe_int(row["archived"], 0)
            if "archived" in keys
            else 0,
            # r3.3k：归档原因，旧库缺列时为空串（前端按“已归档”显示）。
            "archive_reason": str(row["archive_reason"] or "")
            if "archive_reason" in keys
            else "",
            "archived_at": (
                self._safe_float(row["archived_at"], 0.0)
                if "archived_at" in keys and row["archived_at"] is not None
                else 0.0
            ),
            "kind": str(row["kind"] or "fact") if "kind" in keys else "fact",
            "entities": self._parse_entities(row["entities"])
            if "entities" in keys
            else [],
            "valid_until": (
                self._safe_float(row["valid_until"], 0.0)
                if "valid_until" in keys and row["valid_until"] is not None
                else None
            ),
            "confidence": (
                0.5
                if "confidence" not in keys or row["confidence"] is None
                else self._safe_float(row["confidence"], 0.5)
            ),
            "source_batch": str(row["source_batch"] or "")
            if "source_batch" in keys
            else "",
        }
        item["type_label"] = {
            "personal": "个人",
            "group": "群聊",
            "private": "私聊",
        }.get(item["memory_type"], "未知")
        item["identity_label"] = {
            "owner": "所有者",
            "group_owner": "群主",
            "group_admin": "群管",
            "member": "群成员",
        }.get(item["sender_role"], "群成员")
        if item["source_time"]:
            item["source_time_iso"] = self.format_timestamp_iso(item["source_time"])
        item.update(self._time_fields(row["memory_time"]))
        return item

    def _roles_map(self, ids: list[int]) -> dict[int, list[str]]:
        """批量取回记忆关联的角色 ID，按 ``memory_id`` 分组。"""
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT memory_id, role_id FROM memory_roles "
                f"WHERE memory_id IN ({placeholders}) ORDER BY memory_id, role_id",
                ids,
            ).fetchall()
        grouped: dict[int, list[str]] = {}
        for row in rows:
            grouped.setdefault(int(row["memory_id"]), []).append(str(row["role_id"]))
        return grouped

    # ------------------------------------------------------------------ #
    # 写入 / 读取
    # ------------------------------------------------------------------ #
    def find_exact_memory(self, content: str, group_scope: str = "") -> int | None:
        """查找同作用域下正文完全相同的未归档、未过期记忆，用于快速去重。

        作用域必须精确匹配：否则两个群说同一句话时，后写入的那条会被当成
        重复丢弃，还会返回别的群的记忆 ID。过期记忆不参与去重，避免新记忆
        被一个已经不再召回的旧 ID 吞掉（P1-2）。
        """
        text = str(content or "").strip()
        if not text:
            return None
        scope = str(group_scope or "").strip()
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id FROM memories WHERE content = ? AND group_scope = ? "
                "AND archived = 0 AND superseded_by IS NULL "
                "AND (valid_until IS NULL OR valid_until >= ?) "
                "ORDER BY id DESC LIMIT 1",
                (text, scope, time.time()),
            ).fetchone()
        return int(row["id"]) if row else None

    def add_memory(
        self,
        *,
        content: str,
        embedding: list[float],
        role_ids: list[str],
        memory_time: float | None = None,
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        extra_metadata: dict[str, Any] | None = None,
        memory_type: str = "group",
        sender_id: str = "",
        sender_role: str = "member",
        source_msg_id: str = "",
        source_time: float | None = None,
        importance: float = IMPORTANCE_DEFAULT,
        kind: str = "fact",
        entities: list[str] | None = None,
        valid_until: float | None = None,
        confidence: float = 0.5,
        source_batch: str = "",
    ) -> int:
        """持久化一条记忆及其角色关联，并返回新记录 ID。

        正文、角色列表和 embedding 必须非空；记忆时间支持秒或毫秒。主表行
        与角色关联写入同一个 SQLite 事务，任一 SQL 操作失败时事务会回滚，
        避免留下没有关联角色或关联指向无效记忆的半成品。

        Args:
            content: 记忆正文；保存前去除首尾空白。
            embedding: 正文对应的数值向量；其维度随记录一起存储。
            role_ids: 与这条记忆相关的用户/角色 ID；空值会在写库前拒绝。
            memory_time: 记忆实际发生时间；省略时使用当前 Unix 秒时间。
            group_scope: 规范化群范围；空字符串表示全局记忆。
            group_id: 群号，用于结构化过滤及后台列表展示。
            platform_id: 平台标识，用于区分跨平台的同号群。
            extra_metadata: 来源、置信度等扩展信息，非字典输入不会被保存。

        Returns:
            SQLite 自动分配的正整数主键。

        Raises:
            ValueError: 正文、角色集合或 embedding 为空。
        """
        # 校验正文、角色与 embedding 后，在同一事务写入主记录和角色关联。
        clean_content = str(content or "").strip()
        if not clean_content:
            raise ValueError("content 不能为空")
        clean_roles = self._normalize_role_ids(role_ids)
        if not clean_roles:
            raise ValueError("至少需要一个 role_id")
        if not embedding:
            raise ValueError("embedding 不能为空")

        vector = [float(value) for value in embedding]
        raw_time = float(memory_time) if memory_time is not None else time.time()
        timestamp, _ = self._normalize_timestamp(raw_time)
        now = time.time()
        metadata = extra_metadata if isinstance(extra_metadata, dict) else {}
        clean_entities = [
            str(item).strip()
            for item in (entities or [])
            if str(item).strip() and len(str(item).strip()) >= 2
        ][:8]
        clean_kind = str(kind or "fact").strip().lower()
        if clean_kind not in {"fact", "preference", "plan", "episode", "relation"}:
            clean_kind = "fact"

        with self._lock, closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    content, embedding, embedding_dim, memory_time,
                    group_scope, group_id, platform_id, extra_metadata,
                    created_at, updated_at, memory_type, sender_id, sender_role,
                    source_msg_id, source_time, importance,
                    embedding_blob, embedding_norm, grams,
                    kind, entities, valid_until, confidence, source_batch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_content,
                    # 向量只存 BLOB；JSON 列留空串，仅供旧库/第三方读取回退。
                    "",
                    len(vector),
                    timestamp,
                    str(group_scope or "").strip(),
                    str(group_id or "").strip(),
                    str(platform_id or "").strip(),
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    now,
                    str(memory_type or "group"),
                    str(sender_id or "").strip(),
                    str(sender_role or "member"),
                    str(source_msg_id or "").strip(),
                    float(source_time or timestamp),
                    max(0.0, min(IMPORTANCE_MAX, float(importance))),
                    self._encode_vector(vector),
                    math.sqrt(sum(value * value for value in vector)),
                    self._build_grams(" ".join([clean_content, *clean_entities])),
                    clean_kind,
                    json.dumps(clean_entities, ensure_ascii=False),
                    float(valid_until) if valid_until else None,
                    max(0.0, min(1.0, float(confidence))),
                    str(source_batch or ""),
                ),
            )
            memory_id = int(cursor.lastrowid or 0)
            conn.executemany(
                "INSERT OR IGNORE INTO memory_roles(memory_id, role_id) VALUES (?, ?)",
                [(memory_id, role_id) for role_id in clean_roles],
            )
            return memory_id

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        """按主键读取完整记忆；无效 ID 或记录不存在时返回 None。"""
        # 按主键取一条完整记忆；无效 ID 或不存在时返回 None。
        try:
            target = int(memory_id)
        except (TypeError, ValueError):
            return None
        if target <= 0:
            return None
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (target,)
            ).fetchone()
            if not row:
                return None
            role_rows = conn.execute(
                "SELECT role_id FROM memory_roles WHERE memory_id = ? ORDER BY role_id",
                (target,),
            ).fetchall()
        return self._row_to_item(row, [str(r["role_id"]) for r in role_rows])

    def _stale_archive_candidates(
        self,
        *,
        days: float = 30.0,
        max_importance: float = 3.0,
        decay_enable: bool = True,
        decay_rate: float = 0.01,
        decay_floor: float = 0.5,
        access_window_days: float = 30.0,
        access_max_count: int = 10,
        protected: float = IMPORTANCE_MAX,
    ) -> list[int]:
        """返回「低价值将归档」筛选命中的记忆 ID（r3.3m）。

        与 ``archive_stale`` 完全同口径：先 SQL 粗筛原始 importance，再用
        ``compute_effective_importance`` 精筛有效重要性 <= 阈值，最后校验
        最后活跃时间超过 ``days``。最多扫描 2000 条，避免大库全表读。
        """
        now = time.time()
        threshold = self._safe_float(max_importance, 3.0)
        cutoff = now - max(1.0, self._safe_float(days, 30.0)) * 86400
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, importance, confidence, created_at, last_access, "
                "access_count, decay_reset_at FROM memories "
                "WHERE archived = 0 AND importance <= ? ORDER BY id LIMIT 2000",
                (min(IMPORTANCE_MAX, threshold + 2.0),),
            ).fetchall()
        ids: list[int] = []
        for row in rows:
            importance = self._safe_float(row["importance"], IMPORTANCE_DEFAULT)
            # 与 archive_stale 原行为一致：异常/越界值先夹到 0–10 再判断。
            effective = max(0.0, min(IMPORTANCE_MAX, importance))
            if decay_enable:
                effective = compute_effective_importance(
                    importance=importance,
                    confidence=self._safe_float(row["confidence"], 0.5),
                    created_at=self._safe_float(row["created_at"], 0.0),
                    last_access=self._safe_float(row["last_access"], 0.0),
                    access_count=self._safe_int(row["access_count"], 0),
                    decay_reset_at=self._safe_float(row["decay_reset_at"], 0.0),
                    now=now,
                    decay_rate=self._safe_float(decay_rate, 0.01),
                    decay_floor=self._safe_float(decay_floor, 0.5),
                    access_window_days=self._safe_float(access_window_days, 30.0),
                    access_max_count=self._safe_int(access_max_count, 10),
                    protected=self._safe_float(protected, IMPORTANCE_MAX),
                )
            if effective > threshold:
                continue
            last_touch = (
                self._safe_float(row["last_access"], 0.0)
                or self._safe_float(row["decay_reset_at"], 0.0)
                or self._safe_float(row["created_at"], 0.0)
            )
            if last_touch >= cutoff:
                continue
            ids.append(int(row["id"]))
        return ids

    def list_memories(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        keyword: str = "",
        group_scope: str = "",
        role_id: str = "",
        memory_type: str = "",
        identity: str = "",
        include_archived: bool = False,
        archived_only: bool = False,
        lifecycle: str = "",
        lifecycle_importance: float = 3.0,
        lifecycle_days: float = 30.0,
        lifecycle_decay_enable: bool = True,
        lifecycle_decay_rate: float = 0.01,
        lifecycle_decay_floor: float = 0.5,
        lifecycle_access_window_days: float = 30.0,
        lifecycle_access_max_count: int = 10,
        lifecycle_protected: float = IMPORTANCE_MAX,
    ) -> dict[str, Any]:
        """按关键词、作用域和角色过滤记忆，并返回一页结果及分页信息。

        关键词同时匹配正文和字符串形式的记忆 ID，方便控制台按内容或编号
        查找。总数查询和当前页查询共用同一组过滤条件；角色关联在页结果取出后
        批量读取，避免对每条记忆各发起一次查询。

        Args:
            page: 从 1 开始的页码，低于 1 时校正为 1。
            page_size: 每页记录数，限定在 1 到 200 之间。
            keyword: 可选正文/记录 ID 子串。
            group_scope: 可选精确作用域；空字符串表示不过滤作用域。
            role_id: 可选精确角色 ID。
            include_archived: 是否把已归档记忆也纳入列表。
            archived_only: 只返回已归档记忆（优先于 ``include_archived``）。

        Returns:
            包含 ``items``、``total``、``page``、``page_size`` 与 ``has_more`` 的字典。
        """
        # 组合关键词、群范围和角色条件分页查询，并批量装配角色列表。
        page = max(1, int(page))
        page_size = max(1, min(200, int(page_size)))
        offset = (page - 1) * page_size

        where: list[str] = []
        params: list[Any] = []
        keyword = str(keyword or "").strip()
        if keyword:
            # F54：转义 LIKE 通配符，避免用户搜 ``%``/``_`` 时全表命中。
            escaped = (
                keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            where.append(
                "(m.content LIKE ? ESCAPE '\\' OR CAST(m.id AS TEXT) LIKE ? ESCAPE '\\')"
            )
            params.extend([f"%{escaped}%", f"%{escaped}%"])
        group_scope = str(group_scope or "").strip()
        if group_scope == "__global__":
            # r3.3j：控制台「群范围=全局」筛选只看全局层（group_scope 为空）。
            where.append("m.group_scope = ''")
        elif group_scope:
            where.append("m.group_scope = ?")
            params.append(group_scope)
        role_id = str(role_id or "").strip()
        if role_id:
            where.append(
                "EXISTS (SELECT 1 FROM memory_roles r "
                "WHERE r.memory_id = m.id AND r.role_id = ?)"
            )
            params.append(role_id)
        memory_type = str(memory_type or "").strip()
        if memory_type == "unknown":
            # r3.3m：控制台「未知来源」筛选匹配非 personal/group/private 的旧数据。
            where.append("m.memory_type NOT IN ('personal', 'group', 'private')")
        elif memory_type:
            where.append("m.memory_type = ?")
            params.append(memory_type)
        identity = str(identity or "").strip()
        if identity:
            # r3.3f：身份筛选改为「发送者身份 或 任一参与者角色徽标」。
            # 旧实现只比 m.sender_role，会出现卡片上明明有「群主」徽标、
            # 但按「群主」筛选返回 0 条的情况（发送者本身是群成员）。
            where.append(
                "(m.sender_role = ? OR CASE WHEN json_valid(m.extra_metadata) "
                "THEN EXISTS ("
                "SELECT 1 FROM json_each(m.extra_metadata, '$.role_identities') "
                "WHERE json_each.value = ?) ELSE 0 END)"
            )
            params.extend([identity, identity])
        lifecycle = str(lifecycle or "").strip().lower()
        if lifecycle == "stale":
            # r3.3m：低价值将归档必须与 archive_stale 用同一算法（有效重要性），
            # 否则衰减后会被静默归档、筛选却看不到。
            stale_ids = self._stale_archive_candidates(
                days=lifecycle_days,
                max_importance=lifecycle_importance,
                decay_enable=lifecycle_decay_enable,
                decay_rate=lifecycle_decay_rate,
                decay_floor=lifecycle_decay_floor,
                access_window_days=lifecycle_access_window_days,
                access_max_count=lifecycle_access_max_count,
                protected=lifecycle_protected,
            )
            if not stale_ids:
                return {
                    "items": [],
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "has_more": False,
                }
            placeholders = ",".join("?" for _ in stale_ids)
            where.append(f"m.id IN ({placeholders})")
            params.extend(stale_ids)
        elif lifecycle in {"expiring", "expired", "cold"}:
            # r3.3l：生命周期筛选（即将过期 / 已过期 / 长期未访问）。
            now = time.time()
            if lifecycle == "expiring":
                where.append(
                    "m.archived = 0 AND m.valid_until IS NOT NULL "
                    "AND m.valid_until >= ? AND m.valid_until < ?"
                )
                params.extend([now, now + 7 * 86400.0])
            elif lifecycle == "expired":
                where.append(
                    "m.archived = 0 AND m.valid_until IS NOT NULL "
                    "AND m.valid_until < ?"
                )
                params.append(now)
            else:  # cold：长期未访问（按最后活跃时间）
                cutoff = now - max(1.0, self._safe_float(lifecycle_days, 30.0)) * 86400
                where.append(
                    "m.archived = 0 AND COALESCE(NULLIF(m.last_access, 0), "
                    "m.decay_reset_at, m.created_at) < ?"
                )
                params.append(cutoff)
        if archived_only:
            # r3.3：控制台「已归档」视图只列归档记录。
            where.append("m.archived = 1")
        elif not include_archived:
            where.append("m.archived = 0")
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        with self._lock, closing(self._connect()) as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM memories m {where_sql}", params
            ).fetchone()
            total = int(total_row["total"]) if total_row else 0
            rows = conn.execute(
                f"""
                SELECT m.*
                FROM memories m {where_sql}
                ORDER BY m.memory_time DESC, m.id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            ).fetchall()

            ids = [int(row["id"]) for row in rows]
            roles_map: dict[int, list[str]] = {}
            if ids:
                placeholders = ",".join("?" for _ in ids)
                role_rows = conn.execute(
                    f"""
                    SELECT memory_id, role_id FROM memory_roles
                    WHERE memory_id IN ({placeholders})
                    ORDER BY memory_id, role_id
                    """,
                    ids,
                ).fetchall()
                for role_row in role_rows:
                    roles_map.setdefault(int(role_row["memory_id"]), []).append(
                        str(role_row["role_id"])
                    )

        items = [
            self._row_to_item(row, roles_map.get(int(row["id"]), [])) for row in rows
        ]
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": offset + page_size < total,
        }

    def delete_memory(self, memory_id: int) -> bool:
        """删除记忆；ID 保持稳定不重排，被删内容暂存回收站以便撤销。"""
        return self.delete_memory_with_trash_id(memory_id) is not None

    def delete_memory_with_trash_id(self, memory_id: int) -> int | None:
        """删除记忆并返回回收站 ID（同一条连接内完成，F54）。

        ``delete_memory`` 只能返回 bool，调用方若要立刻撤销删除还需要再查一次
        ``latest_trash_id``，两次连接之间可能插入别的删除。本方法在同一个事务
        里返回 ``trash_id``；记录不存在或 ID 非法时返回 ``None``。
        """
        try:
            target = int(memory_id)
        except (TypeError, ValueError):
            return None
        if target <= 0:
            return None
        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (target,)
            ).fetchone()
            if row is None:
                return None
            payload = self._payload_for_trash(row)
            roles = [
                str(item["role_id"])
                for item in conn.execute(
                    "SELECT role_id FROM memory_roles WHERE memory_id = ?", (target,)
                ).fetchall()
            ]
            cursor = conn.execute(
                "INSERT INTO memory_trash(original_id, payload, roles, deleted_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    target,
                    json.dumps(payload, ensure_ascii=False),
                    json.dumps(roles, ensure_ascii=False),
                    time.time(),
                ),
            )
            conn.execute("DELETE FROM memories WHERE id = ?", (target,))
            trash_id = int(cursor.lastrowid or 0)
        return trash_id or None

    @staticmethod
    def _encode_trash_vector(vector: list[float]) -> dict[str, Any]:
        """把回收站向量压成 base64(float32) 编码（F54：不存完整 JSON 向量）。"""
        raw = array("f", [float(value) for value in vector]).tobytes()
        return {
            "codec": "float32-base64",
            "dim": len(vector),
            "data": base64.b64encode(raw).decode("ascii"),
        }

    @classmethod
    def _decode_trash_vector(cls, raw: Any) -> list[float]:
        """还原回收站向量；兼容旧版 JSON 文本/列表载荷。"""
        if isinstance(raw, dict):
            if str(raw.get("codec") or "") != "float32-base64":
                return []
            try:
                data = base64.b64decode(str(raw.get("data") or ""))
                values = array("f")
                values.frombytes(data)
            except (ValueError, TypeError):
                return []
            return [float(value) for value in values]
        if isinstance(raw, list):
            return cls._parse_embedding(json.dumps(raw))
        if isinstance(raw, str):
            return cls._parse_embedding(raw)
        return []

    def _payload_for_trash(self, row: sqlite3.Row) -> dict[str, Any]:
        """把一行记忆转成可 JSON 序列化的回收站载荷。

        F54：向量不再以 JSON 小数列表完整落库，改用 base64(float32) 编码，
        体积约为原 JSON 的 1/4；恢复时优先走新编码，同时兼容旧版 JSON 载荷。
        索引列（grams / norm）不需要进回收站，恢复时按正文重算。
        """
        payload: dict[str, Any] = {}
        for key in row.keys():  # noqa: SIM118 - sqlite3.Row 直接迭代会得到值而非列名
            if key in {"embedding_blob", "embedding_norm", "grams"}:
                continue
            value = row[key]
            if isinstance(value, (bytes, bytearray, memoryview)):
                # 坏数据可能是 BLOB：转成文本，保证回收站载荷一定可 JSON 序列化。
                value = bytes(value).decode("utf-8", "replace")
            payload[key] = value
        vector = self._row_embedding(row)
        if vector:
            payload["embedding"] = self._encode_trash_vector(vector)
        else:
            payload["embedding"] = "[]"
        return payload

    def restore_memory(self, trash_id: int) -> int | None:
        """从回收站恢复一条记忆，返回新 ID。"""
        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT payload, roles FROM memory_trash WHERE trash_id = ?",
                (int(trash_id),),
            ).fetchone()
            if row is None:
                return None
            try:
                payload = json.loads(row["payload"])
            except (json.JSONDecodeError, TypeError):
                logger.warning("[ActiveReply] 回收站载荷损坏，无法恢复: %s", trash_id)
                return None
            if not isinstance(payload, dict):
                return None
            try:
                roles = json.loads(row["roles"]) if row["roles"] else []
            except (json.JSONDecodeError, TypeError):
                roles = []
            if not isinstance(roles, list):
                roles = []
            content = str(payload.get("content", ""))
            # F54：新载荷是 base64(float32) 编码；旧载荷是 JSON 文本，二者都兼容。
            vector = self._decode_trash_vector(payload.get("embedding"))
            embedding_dim = (
                len(vector)
                if vector
                else self._safe_int(payload.get("embedding_dim"), 0)
            )
            # r3.3：恢复后重新计时——衰减基准与最近访问都记为现在，
            # created_at 仍保留载荷里的旧值（修复 r3.2 的 D01）。
            restored_at = time.time()
            conn.execute(
                """
                INSERT INTO memories (
                    content, embedding, embedding_dim, memory_time,
                    group_scope, group_id, platform_id, extra_metadata,
                    created_at, updated_at, memory_type, sender_id, sender_role,
                    source_msg_id, source_time, importance,
                    embedding_blob, embedding_norm, grams,
                    kind, entities, valid_until, confidence, source_batch,
                    last_access, decay_reset_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content,
                    "",
                    embedding_dim,
                    self._safe_timestamp(payload.get("memory_time"), restored_at),
                    payload.get("group_scope", ""),
                    payload.get("group_id", ""),
                    payload.get("platform_id", ""),
                    payload.get("extra_metadata", "{}"),
                    self._safe_timestamp(payload.get("created_at"), restored_at),
                    self._safe_timestamp(payload.get("updated_at"), restored_at),
                    payload.get("memory_type", "group"),
                    payload.get("sender_id", ""),
                    payload.get("sender_role", "member"),
                    payload.get("source_msg_id", ""),
                    payload.get("source_time", 0),
                    payload.get("importance", IMPORTANCE_DEFAULT),
                    self._encode_vector(vector) if vector else None,
                    math.sqrt(sum(value * value for value in vector))
                    if vector
                    else None,
                    self._build_grams(
                        " ".join(
                            [content, *self._parse_entities(payload.get("entities"))]
                        )
                    ),
                    payload.get("kind", "fact"),
                    payload.get("entities", "[]"),
                    payload.get("valid_until"),
                    payload.get("confidence", 0.5),
                    payload.get("source_batch", ""),
                    restored_at,
                    restored_at,
                ),
            )
            new_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            conn.executemany(
                "INSERT OR IGNORE INTO memory_roles(memory_id, role_id) VALUES (?, ?)",
                [(new_id, role) for role in roles],
            )
            conn.execute(
                "DELETE FROM memory_trash WHERE trash_id = ?", (int(trash_id),)
            )
            return new_id

    def latest_trash_id(self) -> int | None:
        """返回最近一次删除写入回收站的 ID。"""
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT MAX(trash_id) AS tid FROM memory_trash"
            ).fetchone()
        return int(row["tid"]) if row and row["tid"] is not None else None

    def purge_trash(self, days: float) -> int:
        """删除回收站中超过保留期的记录，返回删除条数。

        ``days <= 0`` 视为关闭自动清理，直接返回 0，避免误传 0 时清空整个
        回收站。保留期按 ``deleted_at``（删除时间，Unix 秒）计算。
        """
        try:
            retention = float(days)
        except (TypeError, ValueError):
            return 0
        if retention <= 0:
            return 0
        cutoff = time.time() - retention * 86400
        with self._lock, closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "DELETE FROM memory_trash WHERE deleted_at < ?", (cutoff,)
            )
            return int(cursor.rowcount or 0)

    def bump_access(self, memory_ids: list[int]) -> None:
        """命中记忆时更新最后访问时间与访问次数。"""
        ids = [int(item) for item in memory_ids if int(item) > 0]
        if not ids:
            return
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE memories SET last_access = ?, access_count = access_count + 1 "
                f"WHERE id IN ({placeholders})",
                [now, *ids],
            )

    @staticmethod
    def _merge_extra_metadata(
        old: dict[str, Any] | None, new: dict[str, Any] | None
    ) -> dict[str, Any]:
        """有界合并记忆元数据：证据/来源只增不减，其它键保留旧值（P2-1）。"""
        merged = dict(old) if isinstance(old, dict) else {}
        if not isinstance(new, dict):
            return merged

        def as_list(value: object) -> list[str]:
            """把字符串/序列归一成去重文本列表；其它类型忽略（L1）。"""
            if isinstance(value, str):
                raw = [value]
            elif isinstance(value, (list, tuple, set)):
                raw = list(value)
            else:
                return []
            result: list[str] = []
            for item in raw:
                text = str(item).strip()[:300]
                if text and text not in result:
                    result.append(text)
            return result

        for key, value in new.items():
            if key not in merged:
                merged[key] = value
            elif key in {"quotes", "sources"}:
                limit = 5 if key == "quotes" else 8
                current = as_list(merged.get(key))
                for text in as_list(value):
                    if text not in current and len(current) < limit:
                        current.append(text)
                merged[key] = current[:limit]
            elif key == "evidence":
                previous = str(merged.get("evidence") or "").strip()
                addition = str(value or "").strip()
                if addition and addition not in previous:
                    merged["evidence"] = (
                        f"{previous} | {addition}"[:1200]
                        if previous
                        else addition[:1200]
                    )

        return merged

    @staticmethod
    def _append_previous_content(
        metadata: dict[str, Any],
        *,
        old_content: str,
        old_meta: dict[str, Any],
        source: str,
    ) -> None:
        """把被覆盖的旧正文/旧证据追加进 ``extra_metadata.previous_content``（F05）。

        * 有界：最多保留最近 ``_PREVIOUS_CONTENT_LIMIT`` 条；
        * 单条正文/证据都截断，避免元数据无限膨胀；
        * 每条带 ``replaced_at`` 与 ``source``，便于回溯是谁在什么时候改的；
        * 只记录文本，不写旧向量（向量放回收站或重建，不塞进元数据）。
        """
        text = str(old_content or "").strip()
        if not text:
            return
        entries: list[dict[str, Any]] = []
        existing = metadata.get("previous_content")
        if isinstance(existing, list):
            for item in existing:
                if isinstance(item, dict) and str(item.get("content") or "").strip():
                    entries.append(dict(item))
                elif isinstance(item, str) and item.strip():
                    # 兼容早期可能写过的纯字符串快照。
                    entries.append({"content": item.strip()[:_PREVIOUS_CONTENT_MAX_LEN]})
        entry: dict[str, Any] = {
            "content": text[:_PREVIOUS_CONTENT_MAX_LEN],
            "replaced_at": time.time(),
            "source": str(source or "").strip()[:64] or "merge",
        }
        evidence = str(old_meta.get("evidence") or "").strip()
        if evidence:
            entry["evidence"] = evidence[:_PREVIOUS_EVIDENCE_MAX_LEN]
        entries.append(entry)
        metadata["previous_content"] = entries[-_PREVIOUS_CONTENT_LIMIT:]

    def merge_memory(
        self,
        memory_id: int,
        *,
        content: str = "",
        embedding: list[float] | None = None,
        entities: list[str] | None = None,
        importance: float | None = None,
        valid_until: float | None = None,
        role_ids: list[str] | None = None,
        confidence: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """把新信息合并进已有记忆，保留原 ID。

        用于「同一件事被重新总结」的场景：默认用新正文与新向量覆盖旧值，
        实体取并集、重要性取较大值、有效期取较晚者、角色取并集，避免重复行。

        当传入的 ``confidence`` 低于该行现有置信度时（后一次抽取更不确定），
        保留旧正文与旧向量，只合并实体 / 重要性 / 有效期 / 角色，防止一次低
        置信度的重新总结把可靠内容覆盖掉；不传 ``confidence`` 时行为与旧版
        一致。``role_ids`` 与主表更新在同一个事务内完成，任一失败一起回滚。
        ``extra_metadata`` 会与旧元数据做有界合并（quotes 去重≤5、sources 并集、
        evidence 拼接≤1200 字），用于把新证据/来源并入已有记忆（P2-1）。
        """
        try:
            target = int(memory_id)
        except (TypeError, ValueError):
            return False
        if target <= 0:
            return False
        try:
            new_confidence = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            new_confidence = None
        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (target,)
            ).fetchone()
            if row is None:
                return False
            old_entities = self._parse_entities(row["entities"])
            merged_entities = list(dict.fromkeys([*old_entities, *(entities or [])]))[
                :8
            ]
            expiry = self._safe_float(valid_until, 0.0) if valid_until else None
            importance_delta = (
                self._safe_float(importance, 0.0) if importance is not None else 0.0
            )
            merged_entities_json = json.dumps(merged_entities, ensure_ascii=False)
            old_metadata = self._parse_metadata(row["extra_metadata"])
            merged_metadata = self._merge_extra_metadata(old_metadata, extra_metadata)
            # 两个分支共用的「只增不减」字段：重要性取大、有效期取晚、实体并集。
            shared_fields = (
                "importance = MAX(importance, ?), "
                "valid_until = CASE WHEN ? IS NULL THEN valid_until "
                "WHEN valid_until IS NULL THEN ? "
                "ELSE MAX(valid_until, ?) END, "
                "entities = ?, updated_at = ?, extra_metadata = ? "
            )
            keep_old_content = new_confidence is not None and new_confidence < (
                self._safe_float(row["confidence"], 0.0)
            )
            if keep_old_content:
                # 低置信度合并：不动正文 / 向量，只吸收更完整的结构化信息。
                merged_metadata_json = json.dumps(merged_metadata, ensure_ascii=False)
                conn.execute(
                    "UPDATE memories SET " + shared_fields + "WHERE id = ?",
                    (
                        importance_delta,
                        expiry,
                        expiry,
                        expiry,
                        merged_entities_json,
                        time.time(),
                        merged_metadata_json,
                        target,
                    ),
                )
            else:
                old_content = str(row["content"] or "").strip()
                new_content = str(content or row["content"]).strip() or old_content
                vector = [
                    float(value) for value in (embedding or self._row_embedding(row))
                ]
                normalized_confidence = (
                    None
                    if new_confidence is None
                    else max(0.0, min(1.0, new_confidence))
                )
                if new_content != old_content:
                    # F05：覆盖前快照旧正文/旧证据，旧细节不再静默丢失。
                    source = (
                        str(old_metadata.get("source") or "").strip()
                        or str(row["source_msg_id"] or "").strip()
                        or str(row["source_batch"] or "").strip()
                    )
                    self._append_previous_content(
                        merged_metadata,
                        old_content=old_content,
                        old_meta=old_metadata,
                        source=source,
                    )
                merged_metadata_json = json.dumps(merged_metadata, ensure_ascii=False)
                conn.execute(
                    "UPDATE memories SET content = ?, embedding = '', "
                    "embedding_dim = ?, "
                    + shared_fields
                    + ", embedding_blob = ?, embedding_norm = ?, grams = ?, "
                    "confidence = COALESCE(?, confidence) WHERE id = ?",
                    (
                        new_content,
                        len(vector),
                        importance_delta,
                        expiry,
                        expiry,
                        expiry,
                        merged_entities_json,
                        time.time(),
                        merged_metadata_json,
                        self._encode_vector(vector) if vector else None,
                        math.sqrt(sum(value * value for value in vector))
                        if vector
                        else None,
                        self._build_grams(" ".join([new_content, *merged_entities])),
                        normalized_confidence,
                        target,
                    ),
                )
            clean_roles = self._normalize_role_ids(role_ids or [])
            if clean_roles:
                conn.executemany(
                    "INSERT OR IGNORE INTO memory_roles(memory_id, role_id) "
                    "VALUES (?, ?)",
                    [(target, role_id) for role_id in clean_roles],
                )
        return True

    def archive_memory(self, memory_id: int, reason: str = "manual") -> bool:
        """归档一条记忆（不删除，仍可回溯）；已归档或不存在时返回 False。

        ``reason`` 会写入 ``archive_reason`` 列，控制台据此显示遗忘原因。
        """
        try:
            target = int(memory_id)
        except (TypeError, ValueError):
            return False
        if target <= 0:
            return False
        safe_reason = str(reason or "manual").strip().lower()[:32] or "manual"
        with self._lock, closing(self._connect()) as conn, conn:
            archived_at = time.time()
            cursor = conn.execute(
                "UPDATE memories SET archived = 1, decay_reset_at = ?, "
                "archive_reason = ?, archived_at = ? "
                "WHERE id = ? AND archived = 0",
                (archived_at, safe_reason, archived_at, target),
            )
            return bool(cursor.rowcount and cursor.rowcount > 0)

    def archive_stale(
        self,
        *,
        days: float = 20.0,
        max_importance: float = 3.0,
        decay_rate: float = 0.01,
        decay_floor: float = 0.5,
        access_window_days: float = 30.0,
        access_max_count: int = 10,
        protected: float = IMPORTANCE_MAX,
    ) -> int:
        """归档低价值且长期未访问/未写入的记忆（r3.3 改写）。

        复用 ``_stale_archive_candidates`` 的同一套 SQL/有效重要性/最后活跃
        判定，避免候选预览和实际归档两套算法漂移；``decay_rate <= 0`` 表示
        直接用原始重要性判断。最后活跃时间优先取 ``last_access``，从未访问
        过则回落到 ``decay_reset_at`` / ``created_at``。
        """
        rate = self._safe_float(decay_rate, 0.0)
        targets = self._stale_archive_candidates(
            days=days,
            max_importance=max_importance,
            decay_enable=rate > 0,
            decay_rate=rate,
            decay_floor=decay_floor,
            access_window_days=access_window_days,
            access_max_count=access_max_count,
            protected=protected,
        )
        if not targets:
            return 0
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE memories SET archived = 1, decay_reset_at = ?, "
                "archive_reason = 'stale', archived_at = ? WHERE archived = 0 "
                f"AND id IN ({','.join('?' for _ in targets)})",
                [now, now, *targets],
            )
            return int(cursor.rowcount or 0)

    def unarchive_memory(
        self, memory_id: int, ttl_days: float | None = None
    ) -> dict[str, Any] | None:
        """取消归档并重置重要性衰减基准，可选按 TTL 恢复有效期（r3.3）。

        置 ``archived=0``、``decay_reset_at=now``、``updated_at=now``；当
        ``ttl_days`` 非空且当前有效期为空或已过期时，把 ``valid_until`` 续到
        ``now + ttl_days`` 天。返回更新后的完整记录，记录不存在时返回 None。
        """
        try:
            target = int(memory_id)
        except (TypeError, ValueError):
            return None
        if target <= 0:
            return None
        try:
            ttl = float(ttl_days) if ttl_days is not None else 0.0
        except (TypeError, ValueError):
            ttl = 0.0
        now = time.time()
        with self._lock, closing(self._connect()) as conn, conn:
            exists = conn.execute(
                "SELECT id FROM memories WHERE id = ?", (target,)
            ).fetchone()
            if exists is None:
                return None
            if ttl > 0:
                conn.execute(
                    "UPDATE memories SET archived = 0, archive_reason = '', "
                    "archived_at = NULL, decay_reset_at = ?, updated_at = ?, "
                    "valid_until = CASE WHEN valid_until IS NULL "
                    "OR valid_until <= ? THEN ? ELSE valid_until END "
                    "WHERE id = ?",
                    (now, now, now, now + ttl * 86400.0, target),
                )
            else:
                # r3.3m：关闭自适应 TTL 时，旧的有效期可能已过期；恢复时清空
                # 过期的 valid_until（视为永久），否则召回过滤 + 维护会立刻再归档。
                conn.execute(
                    "UPDATE memories SET archived = 0, archive_reason = '', "
                    "archived_at = NULL, decay_reset_at = ?, updated_at = ?, "
                    "valid_until = CASE WHEN valid_until IS NOT NULL "
                    "AND valid_until <= ? THEN NULL ELSE valid_until END "
                    "WHERE id = ?",
                    (now, now, now, target),
                )
        return self.get_memory(target)

    def delete_memory_permanently(self, memory_id: int) -> bool:
        """彻底删除一条记忆（不进回收站，角色关联级联删除，r3.3）。

        用于控制台「彻底删除」：记录不存在或 ID 非法时返回 False。
        """
        try:
            target = int(memory_id)
        except (TypeError, ValueError):
            return False
        if target <= 0:
            return False
        with self._lock, closing(self._connect()) as conn, conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (target,))
            return bool(cursor.rowcount and cursor.rowcount > 0)

    def weekly_forgetting_counts(
        self, *, now: float | None = None, days: float = 7.0
    ) -> dict[str, Any]:
        """统计最近 N 天新归档（遗忘）的记忆数量与原因分布（r3.3l）。

        使用独立的 ``archived_at`` 作为归档时间；r3.3 之前归档的旧行没有该
        时间，不计入（首周数字会偏少）。返回
        ``{"weekly_forgotten": total, "weekly_by_reason": {...}}``。
        """
        current = time.time() if now is None else float(now)
        cutoff = current - max(1.0, float(days)) * 86400.0
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT archive_reason, COUNT(*) AS cnt FROM memories "
                "WHERE archived = 1 AND archived_at IS NOT NULL "
                "AND archived_at >= ? GROUP BY archive_reason",
                (cutoff,),
            ).fetchall()
        by_reason = {
            str(row["archive_reason"] or "unknown"): int(row["cnt"]) for row in rows
        }
        return {
            "weekly_forgotten": int(sum(by_reason.values())),
            "weekly_by_reason": by_reason,
        }

    def forgetting_counts(
        self, *, now: float | None = None, expiring_days: float = 7.0
    ) -> dict[str, int]:
        """统计遗忘可见性数据（r3.3k，供记忆页顶部卡片）。

        Returns:
            ``expiring_soon``（未归档且 N 天内到期）、``expired_active``
            （未归档但已过期，等待维护归档）、``archived_count``（已归档）。
        """
        current = time.time() if now is None else float(now)
        window = max(0.0, float(expiring_days)) * 86400.0
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT "
                "SUM(CASE WHEN archived = 0 AND valid_until IS NOT NULL "
                "AND valid_until >= ? AND valid_until < ? THEN 1 ELSE 0 END) "
                "AS expiring_soon, "
                "SUM(CASE WHEN archived = 0 AND valid_until IS NOT NULL "
                "AND valid_until < ? THEN 1 ELSE 0 END) AS expired_active, "
                "SUM(CASE WHEN archived = 1 THEN 1 ELSE 0 END) AS archived_count "
                "FROM memories",
                (current, current + window, current),
            ).fetchone()
        return {
            "expiring_soon": int((row["expiring_soon"] if row else 0) or 0),
            "expired_active": int((row["expired_active"] if row else 0) or 0),
            "archived_count": int((row["archived_count"] if row else 0) or 0),
        }

    def purge_archived(self, days: float, limit: int = 500) -> int:
        """物理删除超过保留期的归档记忆，返回删除条数（r3.3）。

        ``days <= 0`` 表示关闭归档清理，直接返回 0；保留期按
        ``decay_reset_at``（恢复归档时重置）或 ``created_at`` 计算，物理删除且
        不写回收站。单次最多处理 ``limit`` 条，避免一次大事务卡住写入。
        """
        try:
            retention = float(days)
        except (TypeError, ValueError):
            return 0
        if retention <= 0:
            return 0
        cutoff = time.time() - retention * 86400
        batch = max(1, self._safe_int(limit, 500))
        with self._lock, closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT id FROM memories WHERE archived = 1 "
                "AND COALESCE(decay_reset_at, created_at) < ? "
                "ORDER BY id LIMIT ?",
                (cutoff, batch),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})", ids
            )
            return int(cursor.rowcount or 0)

    def archive_expired(self, now: float | None = None) -> int:
        """把有效期已过的计划/事件归档（不删除，仍可回溯）。"""
        cutoff = float(now if now is not None else time.time())
        with self._lock, closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE memories SET archived = 1, decay_reset_at = ?, "
                "archive_reason = 'expired', archived_at = ? WHERE archived = 0 "
                "AND valid_until IS NOT NULL AND valid_until < ?",
                (cutoff, cutoff, cutoff),
            )
            return int(cursor.rowcount or 0)

    def _similar_records(
        self,
        embedding: list[float],
        *,
        group_scope: str,
        scan_limit: int,
    ) -> list[dict[str, Any]]:
        """取限定作用域内近期记忆的相似度列表，按相似度降序。

        作用域语义与 ``search_memories(scope_mode="exact")`` 一致：非空作用域
        匹配「本作用域 + 全局记忆」，空作用域只匹配全局记忆。已归档、已被
        取代或已过有效期的记忆不参与比较：否则新记忆会被误判为「已有记忆」
        而丢弃，过期记忆也可能被 merge 延长有效期「复活」（P1-2）。
        候选先按记忆时间倒序截取 ``scan_limit`` 条，再在进程内批量算余弦。
        """
        if not embedding:
            return []
        scope = str(group_scope or "").strip()
        if scope:
            where = " WHERE (m.group_scope = ? OR m.group_scope = '')"
            params: list[Any] = [scope]
        else:
            where = " WHERE m.group_scope = ''"
            params = []
        where += (
            " AND m.archived = 0 AND m.superseded_by IS NULL"
            " AND (m.valid_until IS NULL OR m.valid_until >= ?)"
        )
        params.append(time.time())
        sql = (
            "SELECT m.id, m.content, m.embedding, m.embedding_blob, "
            "m.group_scope, m.memory_time, m.kind, m.entities "
            f"FROM memories m{where} ORDER BY m.memory_time DESC LIMIT ?"
        )
        params.append(max(1, int(scan_limit)))
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return []
        # 与召回共用批量余弦（numpy 可用时一次矩阵乘法）；维度不一致记 -1。
        scores = cosine_scores(
            [float(value) for value in embedding],
            [row["embedding_blob"] or row["embedding"] for row in rows],
        )
        order = sorted(range(len(rows)), key=lambda index: scores[index], reverse=True)
        return [
            {
                "memory_id": int(rows[index]["id"]),
                "content": rows[index]["content"],
                "group_scope": rows[index]["group_scope"],
                "kind": rows[index]["kind"],
                "entities": self._parse_entities(rows[index]["entities"]),
                "similarity": round(scores[index], 4),
            }
            for index in order
        ]

    def find_similar_memories(
        self,
        embedding: list[float],
        group_scope: str = "",
        threshold: float = 0.0,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """返回限定作用域内最相似的至多 ``limit`` 条近期记忆（相似度降序）。

        候选集合与单条相似查找完全一致，只是保留前 N 条，供多
        候选去重决策使用；``threshold`` 以下的条目直接丢弃（默认 0.0 会过滤
        掉无法比较的 -1 分，而不是把它当成「完全无关」）。
        """
        count = max(1, int(limit))
        records = self._similar_records(
            embedding, group_scope=group_scope, scan_limit=_SIMILAR_SCAN_LIMIT
        )
        return [item for item in records if item["similarity"] >= threshold][:count]

    def source_counts(self) -> dict[str, int]:
        """按 ``keyword``、``model``、``manual`` 等来源统计记忆数量。"""
        counts: dict[str, int] = {}
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute("SELECT extra_metadata FROM memories").fetchall()
        for row in rows:
            meta = self._parse_metadata(row["extra_metadata"])
            source = str(meta.get("source") or "manual")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def count_since(self, timestamp: float) -> int:
        """统计记忆时间不早于给定 Unix 时间的记录数量。"""
        # 统计指定 Unix 时间之后的记忆数，用于自动记忆面板的今日新增量。
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE memory_time >= ?",
                (float(timestamp),),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_stats(self) -> dict[str, Any]:
        """统计记忆总量、关联范围/角色数和最早、最近记忆时间。"""
        # 汇总条数、群/角色维度以及时间跨度供控制台状态卡展示。
        with self._lock, closing(self._connect()) as conn:
            total_row = conn.execute(
                "SELECT COUNT(*) AS total FROM memories"
            ).fetchone()
            total = int(total_row["total"]) if total_row else 0
            scope_row = conn.execute(
                "SELECT COUNT(DISTINCT group_scope) AS cnt FROM memories "
                "WHERE group_scope <> ''"
            ).fetchone()
            group_row = conn.execute(
                "SELECT COUNT(DISTINCT group_id) AS cnt FROM memories "
                "WHERE group_id <> ''"
            ).fetchone()
            role_row = conn.execute(
                "SELECT COUNT(DISTINCT role_id) AS cnt FROM memory_roles"
            ).fetchone()
            type_rows = conn.execute(
                "SELECT memory_type, COUNT(*) AS cnt FROM memories "
                "WHERE archived = 0 GROUP BY memory_type"
            ).fetchall()
            identity_rows = conn.execute(
                "SELECT sender_role, COUNT(*) AS cnt FROM memories "
                "WHERE archived = 0 GROUP BY sender_role"
            ).fetchall()
            archived_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE archived = 1"
            ).fetchone()
            range_row = conn.execute(
                "SELECT MIN(memory_time) AS oldest, MAX(memory_time) AS latest "
                "FROM memories"
            ).fetchone()

        oldest = (
            self._safe_float(range_row["oldest"], 0.0)
            if range_row and range_row["oldest"] is not None
            else None
        )
        latest = (
            self._safe_float(range_row["latest"], 0.0)
            if range_row and range_row["latest"] is not None
            else None
        )
        type_counts = {
            str(row["memory_type"] or "group"): int(row["cnt"]) for row in type_rows
        }
        identity_counts = {
            str(row["sender_role"] or "member"): int(row["cnt"])
            for row in identity_rows
        }
        return {
            "type_counts": {
                "personal": type_counts.get("personal", 0),
                "group": type_counts.get("group", 0),
                "private": type_counts.get("private", 0),
                "unknown": max(
                    0,
                    total
                    - sum(
                        type_counts.get(key, 0)
                        for key in ("personal", "group", "private")
                    )
                    - (int(archived_row["cnt"]) if archived_row else 0),
                ),
            },
            "identity_counts": {
                "owner": identity_counts.get("owner", 0),
                "group_owner": identity_counts.get("group_owner", 0),
                "group_admin": identity_counts.get("group_admin", 0),
                "member": identity_counts.get("member", 0),
            },
            "archived_count": int(archived_row["cnt"]) if archived_row else 0,
            "total_memories": total,
            "group_scope_count": int(scope_row["cnt"]) if scope_row else 0,
            "group_id_count": int(group_row["cnt"]) if group_row else 0,
            "role_count": int(role_row["cnt"]) if role_row else 0,
            "oldest_memory_time": oldest,
            "latest_memory_time": latest,
            "oldest_memory_time_iso": (
                self.format_timestamp_iso(oldest) if oldest is not None else None
            ),
            "latest_memory_time_iso": (
                self.format_timestamp_iso(latest) if latest is not None else None
            ),
            "display_timezone": self._timezone_name,
        }

    def consolidation_candidates(
        self,
        *,
        older_than_days: float = 7.0,
        group_scope: str = "",
        limit: int = 80,
        exact_scope: bool = False,
    ) -> list[dict[str, Any]]:
        """返回可参与巩固的旧记忆（含 embedding 与角色）。

        已过期（``valid_until`` 早于当前时间）的记忆不参与巩固，避免它们被
        合并后再写回时延长有效期「复活」（P1-2）。``exact_scope=True`` 时
        空作用域表示「只取全局层记忆」，否则空作用域表示不过滤（r3.3：巩固
        流程把全局层当成一个真实作用域处理，必须精确匹配）。
        """
        cutoff = time.time() - max(1.0, float(older_than_days)) * 86400
        now = time.time()
        where = [
            "m.archived = 0",
            "m.superseded_by IS NULL",
            "m.memory_time < ?",
            "(m.valid_until IS NULL OR m.valid_until >= ?)",
        ]
        params: list[Any] = [cutoff, now]
        if group_scope:
            where.append("m.group_scope = ?")
            params.append(str(group_scope))
        elif exact_scope:
            where.append("m.group_scope = ''")
        sql = (
            "SELECT m.* FROM memories m WHERE "
            + " AND ".join(where)
            + " ORDER BY m.memory_time ASC LIMIT ?"
        )
        params.append(max(1, int(limit)))
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
            ids = [int(row["id"]) for row in rows]
            roles_map: dict[int, list[str]] = {}
            if ids:
                placeholders = ",".join("?" for _ in ids)
                for item in conn.execute(
                    f"SELECT memory_id, role_id FROM memory_roles "
                    f"WHERE memory_id IN ({placeholders})",
                    ids,
                ).fetchall():
                    roles_map.setdefault(int(item["memory_id"]), []).append(
                        str(item["role_id"])
                    )
        items = []
        for row in rows:
            item = self._row_to_item(row, roles_map.get(int(row["id"]), []))
            item["embedding"] = self._row_embedding(row)
            items.append(item)
        return items

    def list_group_scopes(self, include_global: bool = False) -> list[str]:
        """按字典序返回最多 200 个记忆作用域（r3.3 支持包含全局层）。

        ``include_global=True`` 时结果包含全局作用域空字符串 ``""``（即使库里
        还没有全局记忆），供巩固流程把全局层也纳入处理。
        """
        # 返回有限数量的作用域，供记忆筛选下拉框与定期巩固使用。
        where = "" if include_global else " WHERE group_scope <> ''"
        sql = (
            "SELECT DISTINCT group_scope FROM memories"
            f"{where} ORDER BY group_scope LIMIT 200"
        )
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(sql).fetchall()
        scopes = [str(row["group_scope"]) for row in rows]
        if include_global and "" not in scopes:
            scopes.append("")
        return scopes

    # ------------------------------------------------------------------ #
    # 语义检索
    # ------------------------------------------------------------------ #
    def search_memories(
        self,
        *,
        query_embedding: list[float] | None = None,
        recall_k: int = 20,
        role_ids: list[str] | None = None,
        role_match_mode: str = "any",
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        start_time: float | None = None,
        end_time: float | None = None,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        max_results: int = 10,
        scan_limit: int | None = None,
        scope_mode: str = "all",
        scope_filter: ScopeFilter | None = None,
    ) -> list[dict[str, Any]]:
        """先按结构化条件筛选，再可选地执行有限窗口内的语义召回。

        执行分为三个阶段：SQL 使用参数化条件筛选作用域、群号、平台、时间和
        角色；按记忆时间倒序最多读取 ``scan_limit`` 条候选并批量取回角色；
        若提供查询向量，则对候选逐条计算余弦相似度并先截取 ``recall_k``，
        最后按时间或相关度排序并应用 ``max_results``。因此扫描上限和召回数
        是两层不同的限制，前者约束数据库读取量，后者约束语义排序后的候选量。

        读路径不拿 ``_lock``：每次查询用独立连接，SQLite 允许并发读，写操作仍由
        写锁串行；这样词法 FTS 超时后即使线程仍持锁，向量检索也能立即执行（H1）。

        Args:
            query_embedding: 查询向量；为空时不计算相似度，按时间返回结果。
            recall_k: 语义相关候选的截取数量；小于等于 0 时不做该层截断。
            role_ids: 可选角色 ID 列表。
            role_match_mode: ``any`` 命中任一角色，``all`` 要求所有角色都关联。
            start_time: 可选包含式时间下界（Unix 秒）。
            end_time: 可选包含式时间上界（Unix 秒）。
            group_scope: 可选精确的规范作用域。
            group_id: 可选精确群号。
            platform_id: 可选精确平台 ID。
            sort_by: ``relevance`` 优先按相似度；否则按记忆时间。
            sort_order: ``asc`` 正序，其他值按倒序处理。
            max_results: 最终最大结果数；小于等于 0 时不截断结果。
            scan_limit: SQL 阶段的最大候选数量；未指定时采用实例默认值。
            scope_mode: ``all`` 时空作用域表示不限制（供控制台全库列表使用）；
                ``exact`` 时空作用域只匹配全局记忆（``group_scope = ''``），
                自动回忆必须用这一档，否则会把别的群的记忆当成全局记忆召回。
            scope_filter: 可见性条件；传入后忽略 ``group_scope`` / ``scope_mode``，
                自动回忆与模型工具统一走这一条路径。

        Returns:
            记忆摘要字典列表，语义检索项额外含 ``similarity`` 分数。
        """
        roles = self._normalize_role_ids(role_ids or [])
        mode = "all" if str(role_match_mode).strip().lower() == "all" else "any"

        where: list[str] = ["m.archived = 0", "m.superseded_by IS NULL"]
        params: list[Any] = []
        # 有效期过滤：过期的计划/事件不再参与召回，fact 类没有有效期。
        where.append("(m.valid_until IS NULL OR m.valid_until >= ?)")
        params.append(time.time())
        if scope_filter is not None:
            # 统一可见性：本会话 ∪ 本人参与。条件由 memory_scope 生成，
            # 与工具侧的 is_visible 共用同一条规则。
            clause, values = scope_filter.clause("m")
            where.append(clause)
            params.extend(values)
        elif group_scope:
            where.append("m.group_scope = ?")
            params.append(str(group_scope).strip())
        elif str(scope_mode).strip().lower() == "exact":
            # 空作用域在 exact 档表示「全局记忆」，必须精确匹配，不能用
            # 省略条件代替——那会扫描所有群/私聊的记忆，造成跨群串味。
            where.append("m.group_scope = ''")
        if group_id:
            where.append("m.group_id = ?")
            params.append(str(group_id).strip())
        if platform_id:
            where.append("m.platform_id = ?")
            params.append(str(platform_id).strip())
        if start_time is not None:
            where.append("m.memory_time >= ?")
            params.append(float(start_time))
        if end_time is not None:
            where.append("m.memory_time <= ?")
            params.append(float(end_time))
        if roles:
            placeholders = ",".join("?" for _ in roles)
            if mode == "all":
                where.append(
                    f"m.id IN (SELECT memory_id FROM memory_roles "
                    f"WHERE role_id IN ({placeholders}) "
                    f"GROUP BY memory_id HAVING COUNT(DISTINCT role_id) = ?)"
                )
                params.extend(roles)
                params.append(len(roles))
            else:
                where.append(
                    f"m.id IN (SELECT DISTINCT memory_id FROM memory_roles "
                    f"WHERE role_id IN ({placeholders}))"
                )
                params.extend(roles)

        # 只在 SQL 中生成筛选子句，所有外部值仍通过 params 绑定；LIMIT 也受
        # scan_limit 约束，避免在 Python 层一次性载入整个记忆库再做排序。
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        base_sql = "SELECT m.* FROM memories m" + where_sql
        limit = (
            self.default_scan_limit if scan_limit is None else max(1, int(scan_limit))
        )
        params.append(limit)

        has_query = bool(query_embedding)
        query_vec = [float(value) for value in (query_embedding or [])]

        if has_query:
            # 只取 id 与向量先算相似度，再取回前 K 条完整行：
            # 避免为上千条候选构造 item，也避免把正文整页读进内存。
            light_sql = (
                "SELECT m.id AS id, m.embedding AS embedding, "
                "m.embedding_blob AS embedding_blob FROM memories m"
                + where_sql
                + " ORDER BY m.memory_time DESC LIMIT ?"
            )
            with closing(self._connect()) as conn:
                light_rows = conn.execute(light_sql, params).fetchall()
            if not light_rows:
                return []
            scores = cosine_scores(
                query_vec,
                [row["embedding_blob"] or row["embedding"] for row in light_rows],
            )
            order = sorted(
                range(len(light_rows)), key=lambda index: scores[index], reverse=True
            )
            if recall_k > 0:
                order = order[: int(recall_k)]
            selected_ids = [int(light_rows[index]["id"]) for index in order]
            score_map = {int(light_rows[index]["id"]): scores[index] for index in order}
            with closing(self._connect()) as conn:
                placeholders = ",".join("?" for _ in selected_ids)
                row_map = {
                    int(row["id"]): row
                    for row in conn.execute(
                        f"SELECT m.* FROM memories m WHERE m.id IN ({placeholders})",
                        selected_ids,
                    ).fetchall()
                }
            roles_map = self._roles_map(selected_ids)
            candidates: list[dict[str, Any]] = []
            for memory_id in selected_ids:
                row = row_map.get(memory_id)
                if row is None:
                    continue
                item = self._row_to_item(row, roles_map.get(memory_id, []))
                item["similarity"] = score_map.get(memory_id, -1.0)
                candidates.append(item)
        else:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    base_sql + " ORDER BY m.memory_time DESC LIMIT ?", params
                ).fetchall()
            if not rows:
                return []
            ids = [int(row["id"]) for row in rows]
            roles_map = self._roles_map(ids)
            candidates = []
            for row in rows:
                item = self._row_to_item(row, roles_map.get(int(row["id"]), []))
                item["similarity"] = None
                candidates.append(item)

        reverse = str(sort_order).strip().lower() != "asc"
        if str(sort_by).strip().lower() == "relevance" and has_query:
            candidates.sort(
                key=lambda item: (
                    item["similarity"]
                    if isinstance(item["similarity"], (int, float))
                    else -1.0
                ),
                reverse=reverse,
            )
        else:
            candidates.sort(key=lambda item: item["memory_time"], reverse=reverse)

        if max_results > 0:
            return candidates[: int(max_results)]
        return candidates

    def search_lexical(
        self,
        query: str,
        *,
        group_scope: str = "",
        scope_mode: str = "exact",
        limit: int = 20,
        scope_filter: ScopeFilter | None = None,
    ) -> list[dict[str, Any]]:
        """用 bigram 全文索引做关键词召回，不调用 embedding。

        只返回未归档、未被取代的记忆；作用域语义与 ``search_memories`` 一致。
        FTS5 不可用或查询没有有效检索词时返回空列表，调用方据此回退到向量检索。
        读路径不拿 ``_lock``（独立连接、SQLite 并发读），避免慢 FTS 阻塞向量检索（H1）。
        """
        grams = self._build_grams(query)
        if not grams or not self.fts_enabled:
            return []
        tokens = grams.split(" ")[:16]
        expression = " OR ".join(f'"{token}"' for token in tokens)
        # FTS5 的 MATCH 左操作数必须是表名本身，不能写成别名。
        where = ["memory_fts MATCH ?", "m.archived = 0", "m.superseded_by IS NULL"]
        where.append("(m.valid_until IS NULL OR m.valid_until >= ?)")
        params: list[Any] = [f"grams:({expression})", time.time()]
        if scope_filter is not None:
            clause, values = scope_filter.clause("m")
            where.append(clause)
            params.extend(values)
        else:
            scope = str(group_scope or "").strip()
            if scope:
                where.append("m.group_scope = ?")
                params.append(scope)
            elif str(scope_mode).strip().lower() == "exact":
                where.append("m.group_scope = ''")
        params.append(max(1, int(limit)))
        sql = (
            "SELECT m.* FROM memory_fts f JOIN memories m ON m.id = f.rowid "
            "WHERE " + " AND ".join(where) + " ORDER BY f.rank LIMIT ?"
        )
        with closing(self._connect()) as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.DatabaseError as exc:
                # F28：词法通道失败不能静默——启动自检会尝试修复，这里保留告警。
                if not self._lexical_error_logged:
                    self._lexical_error_logged = True
                    logger.warning("[ActiveReply] FTS 词法检索失败: %s", exc)
                else:
                    logger.debug("[ActiveReply] FTS 词法检索再次失败: %s", exc)
                return []
            if not rows:
                return []
            ids = [int(row["id"]) for row in rows]
            placeholders = ",".join("?" for _ in ids)
            role_rows = conn.execute(
                f"SELECT memory_id, role_id FROM memory_roles "
                f"WHERE memory_id IN ({placeholders}) ORDER BY memory_id, role_id",
                ids,
            ).fetchall()
        roles_map: dict[int, list[str]] = {}
        for role_row in role_rows:
            roles_map.setdefault(int(role_row["memory_id"]), []).append(
                str(role_row["role_id"])
            )
        return [
            self._row_to_item(row, roles_map.get(int(row["id"]), [])) for row in rows
        ]

    # ------------------------------------------------------------------ #
    # 维护
    # ------------------------------------------------------------------ #
    def cleanup_legacy_records(self) -> dict[str, Any]:
        """把毫秒时间戳规范化，并刷新时区相关的元数据（幂等）。"""
        updated = 0
        normalized_ms = 0
        metadata_refreshed = 0
        with self._lock, closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT id, memory_time, extra_metadata FROM memories ORDER BY id"
            ).fetchall()
            now = time.time()
            cleaned_at = self.format_timestamp_iso(now)
            for row in rows:
                memory_id = int(row["id"])
                normalized_time, converted = self._normalize_timestamp(
                    self._safe_float(row["memory_time"], 0.0)
                )
                if converted:
                    normalized_ms += 1

                metadata = self._parse_metadata(row["extra_metadata"])
                previous = metadata.get("_active_reply_cleanup")
                previous_cleaned_at = (
                    str(previous.get("cleaned_at") or "").strip()
                    if isinstance(previous, dict)
                    else ""
                )
                payload = {
                    "version": 1,
                    "timezone": self._timezone_name,
                    "memory_time_iso": self.format_timestamp_iso(normalized_time),
                    "cleaned_at": previous_cleaned_at or cleaned_at,
                }
                changed = previous != payload
                if changed:
                    metadata["_active_reply_cleanup"] = payload
                    metadata_refreshed += 1
                if not converted and not changed:
                    continue
                conn.execute(
                    "UPDATE memories SET memory_time = ?, extra_metadata = ?, "
                    "updated_at = ? WHERE id = ?",
                    (
                        normalized_time,
                        json.dumps(metadata, ensure_ascii=False),
                        now,
                        memory_id,
                    ),
                )
                updated += 1

        return {
            "status": "ok",
            "scanned": len(rows),
            "updated": updated,
            "normalized_millisecond_timestamps": normalized_ms,
            "metadata_refreshed": metadata_refreshed,
            "display_timezone": self._timezone_name,
        }
