"""记忆抽取调度器。

把「批量攒满、冷却期满、空闲兜底」三类触发与分段执行从 ``main.py`` 拆出来，
只通过插件对象访问运行态。所有窗口规划仍是 ``extract_plan`` 的纯函数。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from astrbot.api import logger

from .extract_plan import plan_batch, plan_sweep, select_chunk
from .history_format import (
    cooldown_ok,
    find_pending_start,
    is_private_origin,
    parse_line,
    render_for_extract,
)


class ExtractionRunner:
    """驱动某个插件实例的记忆抽取。"""

    def __init__(self, plugin: Any) -> None:
        """保存插件引用并初始化任务/在跑集合。"""
        self.plugin = plugin
        self.task: asyncio.Task | None = None
        self.inflight: set[str] = set()

    # ------------------------------------------------------------------ #
    # 触发
    # ------------------------------------------------------------------ #
    def schedule_if_due(self, event: Any, cfg: Any) -> None:
        """按「累计消息 + 冷却」触发：攒满 batch 且冷却结束才总结。"""
        plugin = self.plugin
        if not (cfg.memory_rag.enable and cfg.memory_rag.auto_extract_enable):
            return
        origin = event.unified_msg_origin
        # 私聊走 on_private_message 的专用路径（带 private scope），后台调度
        # 不能把私聊当群会话抽取，否则用户 ID 会被当成群号（F01）。
        if is_private_origin(origin):
            return
        history = plugin.state.memory_history(origin)
        if not history:
            return
        start = find_pending_start(
            history, plugin.state.last_extract_msg.get(origin, "")
        )
        pending = len(history) - start
        if not plan_batch(
            pending,
            cfg.memory_rag.batch_messages,
            cooldown_ok(
                time.time(),
                plugin.state.last_extract_at.get(origin, 0.0),
                cfg.memory_rag.auto_extract_interval_minutes,
            ),
        ):
            return
        logger.info(
            "[ActiveReply] 自动记忆触发 | origin=%s pending=%s batch=%s",
            origin,
            pending,
            cfg.memory_rag.batch_messages,
        )
        plugin._journal(origin, f"累计 {pending} 条，触发片段总结（{pending} 条）")
        self._start(
            origin,
            cfg,
            history[start:],
            reason="batch",
            self_id=plugin._resolve_self_id(event),
        )

    def _start(
        self,
        origin: str,
        cfg: Any,
        lines: list[str],
        *,
        reason: str,
        self_id: str = "",
    ) -> None:
        """启动一次积压总结；同一会话同时只允许一个在跑。"""
        if not lines or origin in self.inflight:
            return
        if not self_id:
            for line in lines:
                parsed = parse_line(line)
                if parsed is not None and parsed.is_self and parsed.sender_id:
                    self_id = parsed.sender_id
                    break
        self.inflight.add(origin)
        self.plugin.state.last_extract_at[origin] = time.time()
        self.plugin._spawn(
            self._run(origin, cfg, list(lines), reason, self_id),
            name=f"areply-extract-{reason}",
        )

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    async def _run(
        self,
        origin: str,
        cfg: Any,
        lines: list[str],
        reason: str,
        self_id: str,
    ) -> None:
        """把积压消息按字数上限分段总结，逐段推进游标，保证不丢。"""
        plugin = self.plugin
        try:
            rest = list(lines)
            while rest:
                filtered = plugin._filter_extract_window(rest)
                if len(filtered) < 3:
                    # 0 条才是“整段被过滤”，推进游标；只有 1–2 条有效消息时
                    # 暂存等待后续消息，不能把它们连同积压一起永久丢弃。
                    if not filtered:
                        skipped_id = plugin._last_message_id(rest)
                        if skipped_id:
                            plugin.state.last_extract_msg[origin] = skipped_id
                        logger.debug(
                            "[ActiveReply] 积压无有效消息，跳过 | origin=%s rest=%s",
                            origin,
                            len(rest),
                        )
                    else:
                        logger.debug(
                            "[ActiveReply] 积压有效消息不足，暂存 | origin=%s valid=%s",
                            origin,
                            len(filtered),
                        )
                    break
                chunk, _ = select_chunk(
                    filtered,
                    cfg.memory_rag.extract_max_chars,
                    lambda line: len(render_for_extract(line)) + 1,
                )
                if len(chunk) < 3:
                    # 单条消息就超过字数上限：无法组成有效窗口，推进游标避免卡死。
                    # 首条可能是无消息号的机器人行，回退到整段最后一条消息 ID。
                    skipped_id = plugin._last_message_id(chunk) or plugin._last_message_id(rest)
                    if skipped_id:
                        plugin.state.last_extract_msg[origin] = skipped_id
                    logger.warning(
                        "[ActiveReply] 积压消息超过字数上限，已跳过 | origin=%s chunk=%s",
                        origin,
                        len(chunk),
                    )
                    break
                diagnostics: dict[str, object] = {}
                status = await plugin._extract_memories(
                    origin, chunk, cfg, self_id=self_id, diagnostics=diagnostics
                )
                if status == "failed":
                    plugin._schedule_extract_retry(
                        origin, chunk, cfg, 1, self_id=self_id
                    )
                    return
                if status == "skipped":
                    skip_reason = str(diagnostics.get("reason") or "")
                    # 无实质内容/有效消息不足是永久跳过：推进游标，否则同一段
                    # 窗口会在每次冷却后被反复扫描。无模型/存储未就绪则保留重试。
                    if skip_reason in {
                        "valid_messages_too_few",
                        "no_substantive_content",
                        "empty_prompt",
                    }:
                        skipped_id = plugin._last_message_id(chunk)
                        if skipped_id:
                            plugin.state.last_extract_msg[origin] = skipped_id
                        last_line = chunk[-1]
                        index = next(
                            (
                                pos
                                for pos, line in enumerate(rest)
                                if line is last_line
                            ),
                            None,
                        )
                        if index is None:
                            break
                        rest = rest[index + 1 :]
                        continue
                    logger.info(
                        "[ActiveReply] 积压总结跳过 | origin=%s reason=%s skip=%s",
                        origin,
                        reason,
                        skip_reason or "unknown",
                    )
                    return
                last_id = plugin._last_message_id(chunk)
                if last_id:
                    plugin.state.last_extract_msg[origin] = last_id
                last_line = chunk[-1]
                index = next(
                    (pos for pos, line in enumerate(rest) if line is last_line), None
                )
                if index is None:
                    break
                rest = rest[index + 1 :]
            if rest:
                logger.debug(
                    "[ActiveReply] 积压暂存 | origin=%s rest=%s reason=%s",
                    origin,
                    len(rest),
                    reason,
                )
        except Exception as exc:
            logger.warning(f"[ActiveReply] 积压总结失败: {exc}", exc_info=True)
        finally:
            self.inflight.discard(origin)

    # ------------------------------------------------------------------ #
    # 后台扫描
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """启动后台扫描任务（重复调用安全）。"""
        if self.task is not None and not self.task.done():
            return
        self.task = asyncio.create_task(self._loop())
        self.task.add_done_callback(self._clear_task)

    def stop(self) -> None:
        """停止后台扫描并清空在跑集合。"""
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = None
        self.inflight.clear()

    def _clear_task(self, task: asyncio.Task) -> None:
        """扫任务结束后清掉引用，允许下次重载重新启动。"""
        if self.task is task:
            self.task = None

    async def _loop(self) -> None:
        """每 30 秒检查一次；只有满足条件的会话才会真正触发抽取。"""
        await asyncio.sleep(20)
        while True:
            try:
                await self.sweep()
            except Exception as exc:
                logger.warning(f"[ActiveReply] 抽取扫描失败: {exc}")
            await asyncio.sleep(30)

    async def sweep(self) -> None:
        """检查冷却期满与空闲兜底两类触发。"""
        plugin = self.plugin
        now = time.time()
        for origin in list(plugin.state.origin_lru.keys()):
            if origin in self.inflight:
                continue
            # P1：私聊记忆由 on_private_message 专用路径按 ``平台:private:用户``
            # 写入并自带 scope；后台扫描只处理群会话，避免把私聊用户 ID 当群号
            # 抽成可跨群召回的记忆（F01）。
            if is_private_origin(origin):
                continue
            history = plugin.state.memory_history(origin)
            if not history:
                continue
            start = find_pending_start(
                history, plugin.state.last_extract_msg.get(origin, "")
            )
            pending = len(history) - start
            if pending < 3:
                continue
            cfg = plugin._cfg_for_origin(origin)
            if not (cfg.memory_rag.enable and cfg.memory_rag.auto_extract_enable):
                continue
            gap_minutes = cfg.memory_rag.window_gap_minutes
            reason = plan_sweep(
                now=now,
                last_chat=plugin.state.last_chat_at.get(origin, 0.0),
                last_flush_at=plugin.state.gap_flushed_at.get(origin, 0.0),
                gap_minutes=gap_minutes,
                pending=pending,
                batch=cfg.memory_rag.batch_messages,
                last_extract_at=plugin.state.last_extract_at.get(origin, 0.0),
                interval_minutes=cfg.memory_rag.auto_extract_interval_minutes,
            )
            if reason == "idle":
                plugin.state.gap_flushed_at[origin] = now
                plugin._journal(
                    origin,
                    f"对话空闲 {int(gap_minutes)} 分钟，触发片段总结（{pending} 条）",
                )
            elif reason == "cooldown":
                plugin._journal(origin, f"冷却结束，积压 {pending} 条，触发总结")
            else:
                continue
            self._start(origin, cfg, history[start:], reason=reason)


__all__ = ["ExtractionRunner"]
