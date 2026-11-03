"""记忆 RAG 的 SQLite 存储层。

设计取舍：
* 向量以 JSON 文本落库，检索时在进程内计算余弦相似度；
* 单次检索最多扫描 ``default_scan_limit`` 条最新记录，避免全表读取；
* 角色与记忆是多对多关系，单独建表并开启外键级联删除；
* 所有查询参数化，不拼接外部输入。

数据文件位于 ``data/plugin_data/astrbot_plugin_active_reply/memory_rag.db``。
"""

from __future__ import annotations

import json
import math
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

from .memory_conflict import entities_overlap, same_kind
from .memory_scope import ScopeFilter
from .vector_math import cosine_scores

DEFAULT_SCAN_LIMIT = 5000


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
        self.fts_error = ""
        self.set_display_timezone(display_timezone)
        self._init_schema()
        self.normalize_group_scopes()

    # ------------------------------------------------------------------ #
    # 基础设施
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        """创建配置好行映射和外键支持的数据库连接。"""
        # 每次操作独立连接，并开启外键以保证删除记忆时同步删除角色关联。
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        """确保记忆表、角色关联表及检索所需索引存在。"""
        # 创建记忆主表、角色关联表及常用筛选索引；已有数据库保持兼容。
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, closing(self._connect()) as conn, conn:
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
                CREATE INDEX IF NOT EXISTS idx_roles_role ON memory_roles(role_id);
                CREATE INDEX IF NOT EXISTS idx_roles_memory ON memory_roles(memory_id);

                CREATE TABLE IF NOT EXISTS memory_trash (
                    trash_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    roles TEXT NOT NULL DEFAULT '[]',
                    deleted_at REAL NOT NULL
                );
                """
            )
            self._ensure_columns(conn)
            self._ensure_fts(conn)

    _EXTRA_COLUMNS = {
        "memory_type": "TEXT NOT NULL DEFAULT 'group'",
        "sender_id": "TEXT NOT NULL DEFAULT ''",
        "sender_role": "TEXT NOT NULL DEFAULT 'member'",
        "source_msg_id": "TEXT NOT NULL DEFAULT ''",
        "source_time": "REAL NOT NULL DEFAULT 0",
        "importance": "REAL NOT NULL DEFAULT 0.5",
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
        except sqlite3.DatabaseError as exc:  # pragma: no cover - 依赖 SQLite 构建
            self.fts_enabled = False
            self.fts_error = str(exc)
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

    def rebuild_fts(self) -> int:
        """整体重建词法索引，返回补算 bigram 的记录数。"""
        if not self.fts_enabled:
            return 0
        with self._lock, closing(self._connect()) as conn, conn:
            return self._backfill_grams(conn)

    _MESSAGE_TYPE_SEGMENTS = {
        "group",
        "groupmessage",
        "friend",
        "friendmessage",
        "private",
    }

    def normalize_group_scopes(self) -> int:
        """把 ``platform:GroupMessage:123`` 之类的旧 scope 归一成 ``platform:123``。"""
        changed = 0
        with self._lock, closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT id, group_scope, group_id, platform_id FROM memories"
            ).fetchall()
            for row in rows:
                scope = str(row["group_scope"] or "").strip()
                parts = scope.split(":")
                if (
                    len(parts) < 3
                    or parts[1].lower() not in self._MESSAGE_TYPE_SEGMENTS
                ):
                    continue
                platform = parts[0].strip()
                group = parts[-1].strip()
                new_scope = f"{platform}:{group}" if platform and group else group
                conn.execute(
                    "UPDATE memories SET group_scope = ?, group_id = ?, platform_id = ? "
                    "WHERE id = ?",
                    (
                        new_scope,
                        str(row["group_id"] or group),
                        str(row["platform_id"] or platform),
                        int(row["id"]),
                    ),
                )
                changed += 1
        return changed

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
        """将 Unix 时间秒数转换为当前展示时区下的 ISO 8601 时间。"""
        # 将 Unix 秒时间戳按当前展示时区格式化为 ISO 8601 字符串。
        return datetime.fromtimestamp(float(timestamp), tz=self._tzinfo).isoformat()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_role_ids(role_ids: list[str] | tuple[str, ...]) -> list[str]:
        """清理角色 ID 并按输入顺序去重。"""
        # 清理空 ID 并按首次出现顺序去重，供关系表写入使用。
        result: list[str] = []
        seen: set[str] = set()
        for raw in role_ids or ():
            role_id = str(raw or "").strip()
            if not role_id or role_id in seen:
                continue
            seen.add(role_id)
            result.append(role_id)
        return result

    @staticmethod
    def _parse_embedding(raw: str | None) -> list[float]:
        """将存储的 JSON 向量解析为浮点列表，解析失败时返回空列表。"""
        # 将数据库中的 JSON 向量还原；损坏或不符合数组格式时返回空向量。
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
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
        except json.JSONDecodeError:
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
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """计算两个等维向量的余弦相似度；无效向量返回 -1。"""
        # 计算等维向量的余弦相似度；空向量、零向量和维数不一致视为无效。
        if not left or not right or len(left) != len(right):
            return -1.0
        dot = norm_left = norm_right = 0.0
        for a, b in zip(left, right, strict=False):
            dot += a * b
            norm_left += a * a
            norm_right += b * b
        if norm_left <= 0 or norm_right <= 0:
            return -1.0
        return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))

    @staticmethod
    def _normalize_timestamp(raw: float | int) -> tuple[float, bool]:
        """把毫秒时间戳换算为秒，并返回换算后的数值和是否转换标记。"""
        # 兼容秒/毫秒两种 Unix 时间戳，第二个返回值表示是否执行了换算。
        value = float(raw)
        if value > 1e12:  # 毫秒时间戳
            return value / 1000.0, True
        return value, False

    def _time_fields(self, timestamp: float) -> dict[str, Any]:
        """为记忆 API 生成时间戳、ISO 字符串及展示时区字段。"""
        # 生成 API/页面共用的时间字段，明确携带展示时区名称。
        dt = datetime.fromtimestamp(float(timestamp), tz=self._tzinfo)
        return {
            "memory_time": float(timestamp),
            "memory_time_iso": dt.isoformat(),
            "memory_time_timezone": self._timezone_name,
        }

    def _row_to_item(self, row: sqlite3.Row, role_ids: list[str]) -> dict[str, Any]:
        """将数据库行转换为包含角色和时区信息的普通字典。"""
        # 把 SQLite 行转换为不依赖 sqlite3.Row 的普通 API 字典。
        keys = row.keys()
        item: dict[str, Any] = {
            "memory_id": int(row["id"]),
            "content": str(row["content"]),
            "embedding_dim": int(row["embedding_dim"]),
            "group_scope": str(row["group_scope"] or ""),
            "group_id": str(row["group_id"] or ""),
            "platform_id": str(row["platform_id"] or ""),
            "related_role_ids": role_ids,
            "extra_metadata": self._parse_metadata(row["extra_metadata"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
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
            "source_time": float(row["source_time"] or 0)
            if "source_time" in keys
            else 0.0,
            "importance": float(row["importance"] or 0.5)
            if "importance" in keys
            else 0.5,
            "last_access": float(row["last_access"] or 0)
            if "last_access" in keys
            else 0.0,
            "access_count": int(row["access_count"] or 0)
            if "access_count" in keys
            else 0,
            "superseded_by": (
                int(row["superseded_by"]) if row["superseded_by"] is not None else None
            )
            if "superseded_by" in keys
            else None,
            "archived": int(row["archived"] or 0) if "archived" in keys else 0,
            "kind": str(row["kind"] or "fact") if "kind" in keys else "fact",
            "entities": self._parse_entities(row["entities"])
            if "entities" in keys
            else [],
            "valid_until": (
                float(row["valid_until"])
                if "valid_until" in keys and row["valid_until"] is not None
                else None
            ),
            "confidence": float(row["confidence"] or 0.5)
            if "confidence" in keys
            else 0.5,
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
            "group_admin": "群管理",
            "member": "群成员",
        }.get(item["sender_role"], "群成员")
        if item["source_time"]:
            item["source_time_iso"] = self.format_timestamp_iso(item["source_time"])
        item.update(self._time_fields(float(row["memory_time"])))
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
        """查找同作用域下正文完全相同的未归档记忆，用于快速去重。

        作用域必须精确匹配：否则两个群说同一句话时，后写入的那条会被当成
        重复丢弃，还会返回别的群的记忆 ID。
        """
        text = str(content or "").strip()
        if not text:
            return None
        scope = str(group_scope or "").strip()
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id FROM memories WHERE content = ? AND group_scope = ? "
                "AND archived = 0 ORDER BY id DESC LIMIT 1",
                (text, scope),
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
        importance: float = 0.5,
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
                    max(0.0, min(1.0, float(importance))),
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
            where.append("(m.content LIKE ? OR CAST(m.id AS TEXT) LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        group_scope = str(group_scope or "").strip()
        if group_scope:
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
        if memory_type:
            where.append("m.memory_type = ?")
            params.append(memory_type)
        identity = str(identity or "").strip()
        if identity:
            where.append("m.sender_role = ?")
            params.append(identity)
        if not include_archived:
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
        try:
            target = int(memory_id)
        except (TypeError, ValueError):
            return False
        if target <= 0:
            return False
        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (target,)
            ).fetchone()
            if row is None:
                return False
            payload = self._payload_for_trash(row)
            roles = [
                str(item["role_id"])
                for item in conn.execute(
                    "SELECT role_id FROM memory_roles WHERE memory_id = ?", (target,)
                ).fetchall()
            ]
            conn.execute(
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
        return True

    def _payload_for_trash(self, row: sqlite3.Row) -> dict[str, Any]:
        """把一行记忆转成可 JSON 序列化的回收站载荷。

        BLOB 无法直接 JSON 化，这里把向量回填成 JSON 文本，恢复时再重新编码；
        索引列（grams / norm）不需要进回收站，恢复时按正文重算。
        """
        payload: dict[str, Any] = {}
        for key in row.keys():  # noqa: SIM118 - sqlite3.Row 直接迭代会得到值而非列名
            if key in {"embedding_blob", "embedding_norm", "grams"}:
                continue
            payload[key] = row[key]
        vector = self._row_embedding(row)
        if vector:
            payload["embedding"] = json.dumps(vector, ensure_ascii=False)
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
            payload = json.loads(row["payload"])
            roles = json.loads(row["roles"]) if row["roles"] else []
            content = str(payload.get("content", ""))
            vector = self._parse_embedding(payload.get("embedding", "[]"))
            conn.execute(
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
                    content,
                    payload.get("embedding", "[]"),
                    payload.get("embedding_dim", 0),
                    payload.get("memory_time", time.time()),
                    payload.get("group_scope", ""),
                    payload.get("group_id", ""),
                    payload.get("platform_id", ""),
                    payload.get("extra_metadata", "{}"),
                    payload.get("created_at", time.time()),
                    payload.get("updated_at", time.time()),
                    payload.get("memory_type", "group"),
                    payload.get("sender_id", ""),
                    payload.get("sender_role", "member"),
                    payload.get("source_msg_id", ""),
                    payload.get("source_time", 0),
                    payload.get("importance", 0.5),
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

    def merge_memory(
        self,
        memory_id: int,
        *,
        content: str = "",
        embedding: list[float] | None = None,
        entities: list[str] | None = None,
        importance: float | None = None,
        valid_until: float | None = None,
    ) -> bool:
        """把新信息合并进已有记忆，保留原 ID。

        用于「同一件事被重新总结」的场景：正文与向量更新为较新版本，
        实体取并集、重要性取较大值、有效期取较晚者，避免重复行。
        """
        try:
            target = int(memory_id)
        except (TypeError, ValueError):
            return False
        if target <= 0:
            return False
        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (target,)
            ).fetchone()
            if row is None:
                return False
            new_content = str(content or row["content"]).strip() or row["content"]
            vector = [float(value) for value in (embedding or self._row_embedding(row))]
            old_entities = self._parse_entities(row["entities"])
            merged_entities = list(dict.fromkeys([*old_entities, *(entities or [])]))[
                :8
            ]
            expiry = float(valid_until) if valid_until else None
            conn.execute(
                "UPDATE memories SET content = ?, embedding = '', embedding_dim = ?, "
                "updated_at = ?, importance = MAX(importance, ?), "
                "valid_until = CASE WHEN ? IS NULL THEN valid_until "
                "WHEN valid_until IS NULL THEN ? "
                "ELSE MAX(valid_until, ?) END, "
                "embedding_blob = ?, embedding_norm = ?, grams = ?, entities = ? "
                "WHERE id = ?",
                (
                    new_content,
                    len(vector),
                    time.time(),
                    float(importance) if importance is not None else 0.0,
                    expiry,
                    expiry,
                    expiry,
                    self._encode_vector(vector) if vector else None,
                    math.sqrt(sum(value * value for value in vector))
                    if vector
                    else None,
                    self._build_grams(" ".join([new_content, *merged_entities])),
                    json.dumps(merged_entities, ensure_ascii=False),
                    target,
                ),
            )
        return True

    def supersede(self, old_id: int, new_id: int) -> None:
        """把旧记忆标记为被新记忆取代。"""
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE id = ?",
                (int(new_id), int(old_id)),
            )

    def archive_stale(self, *, days: float = 20.0, max_importance: float = 0.35) -> int:
        """归档低重要性且长期未访问的记忆。"""
        cutoff = time.time() - max(1.0, float(days)) * 86400
        with self._lock, closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE memories SET archived = 1 "
                "WHERE archived = 0 AND importance <= ? "
                "AND COALESCE(NULLIF(last_access, 0), memory_time) < ?",
                (float(max_importance), cutoff),
            )
            return int(cursor.rowcount or 0)

    def archive_expired(self, now: float | None = None) -> int:
        """把有效期已过的计划/事件归档（不删除，仍可回溯）。"""
        cutoff = float(now if now is not None else time.time())
        with self._lock, closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE memories SET archived = 1 WHERE archived = 0 "
                "AND valid_until IS NOT NULL AND valid_until < ?",
                (cutoff,),
            )
            return int(cursor.rowcount or 0)

    @staticmethod
    def should_supersede(
        old_kind: Any,
        old_entities: Any,
        new_kind: Any,
        new_entities: Any,
    ) -> bool:
        """新记忆能否取代相似旧记忆：同类型且实体有交集（规则见 memory_conflict）。"""
        return same_kind(old_kind, new_kind) and entities_overlap(
            old_entities, new_entities
        )

    def find_similar_memory(
        self,
        embedding: list[float],
        *,
        group_scope: str = "",
        threshold: float = 0.92,
        scan_limit: int = 2000,
    ) -> dict[str, Any] | None:
        """在限定作用域的近期记忆中查找最相似项，用于避免重复写入。

        作用域语义与 ``search_memories(scope_mode="exact")`` 一致：非空作用域
        匹配「本作用域 + 全局记忆」，空作用域只匹配全局记忆。空作用域若不限制
        过滤，写一条全局记忆时会和别的群的记忆比较，进而把它当作冲突项取代掉。
        已归档或已被取代的记忆不参与比较：否则新记忆会被误判为「已有记忆」
        而丢弃，或再次取代一条早已失效的旧记忆。

        Args:
            embedding: 待比较的查询向量。
            group_scope: 记忆作用域；为空表示这是全局记忆。
            threshold: 相似度最低阈值。
            scan_limit: 最多扫描的近期记录数。

        Returns:
            达到阈值的最佳记忆摘要（含 ``kind`` 与 ``entities``）；没有匹配项
            时返回 None。
        """
        if not embedding:
            return None
        scope = str(group_scope or "").strip()
        if scope:
            where = " WHERE (m.group_scope = ? OR m.group_scope = '')"
            params: list[Any] = [scope]
        else:
            where = " WHERE m.group_scope = ''"
            params = []
        where += " AND m.archived = 0 AND m.superseded_by IS NULL"
        sql = (
            "SELECT m.id, m.content, m.embedding, m.embedding_blob, "
            "m.group_scope, m.memory_time, m.kind, m.entities "
            f"FROM memories m{where} ORDER BY m.memory_time DESC LIMIT ?"
        )
        params.append(max(1, int(scan_limit)))
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return None
        # 与召回共用批量余弦（numpy 可用时一次矩阵乘法）；维度不一致记 -1。
        scores = cosine_scores(
            [float(value) for value in embedding],
            [row["embedding_blob"] or row["embedding"] for row in rows],
        )
        best_index = max(range(len(rows)), key=lambda index: scores[index])
        best_score = scores[best_index]
        if best_score < threshold:
            return None
        row = rows[best_index]
        return {
            "memory_id": int(row["id"]),
            "content": row["content"],
            "group_scope": row["group_scope"],
            "kind": row["kind"],
            "entities": self._parse_entities(row["entities"]),
            "similarity": round(best_score, 4),
        }

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
            float(range_row["oldest"])
            if range_row and range_row["oldest"] is not None
            else None
        )
        latest = (
            float(range_row["latest"])
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
        self, *, older_than_days: float = 7.0, group_scope: str = "", limit: int = 80
    ) -> list[dict[str, Any]]:
        """返回可参与巩固的旧记忆（含 embedding 与角色）。"""
        cutoff = time.time() - max(1.0, float(older_than_days)) * 86400
        where = ["m.archived = 0", "m.superseded_by IS NULL", "m.memory_time < ?"]
        params: list[Any] = [cutoff]
        if group_scope:
            where.append("m.group_scope = ?")
            params.append(str(group_scope))
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

    def list_group_scopes(self) -> list[str]:
        """按字典序返回最多 200 个非空记忆作用域。"""
        # 返回有限数量的非空作用域，供记忆筛选下拉框使用。
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT group_scope FROM memories "
                "WHERE group_scope <> '' ORDER BY group_scope LIMIT 200"
            ).fetchall()
        return [str(row["group_scope"]) for row in rows]

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
            with self._lock, closing(self._connect()) as conn:
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
            with self._lock, closing(self._connect()) as conn:
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
            with self._lock, closing(self._connect()) as conn:
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
        with self._lock, closing(self._connect()) as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.DatabaseError:
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
                    float(row["memory_time"])
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
