"""AstrBot Web API 与旧版 Quart 请求对象的兼容层。

AstrBot 新版插件页面通过 ``astrbot.api.web.request`` 读取请求，较早版本则依赖
Quart 的 ``request`` 全局对象。本模块在导入时选择可用实现，并把 query、JSON
请求体和响应数据归一化成稳定接口，使页面 handler 无需分支判断框架版本。

响应约定为 HTTP 200 配合 ``{"success": bool, ...}`` JSON 字段，由前端检查
``success`` 并展示 ``error``。这避免不同框架在异常响应格式上的差异向业务层扩散。
"""

from __future__ import annotations

import json
from typing import Any

try:  # AstrBot >= 4.14 左右提供的新 API
    from astrbot.api.web import request as _request  # type: ignore

    NEW_WEB_API = True
except Exception:  # pragma: no cover - 兼容旧版本
    try:
        from quart import request as _request  # type: ignore

        NEW_WEB_API = False
    except Exception:
        _request = None  # type: ignore[assignment]
        NEW_WEB_API = False


def _query_source() -> Any:
    """获取当前请求的查询参数映射，兼容 AstrBot 与 Quart 的属性名。"""
    # 新版请求对象使用 query，旧版 Quart 使用 args；均不可用时返回空映射。
    if _request is None:
        return {}
    return getattr(_request, "query", None) or getattr(_request, "args", None) or {}


def query_get(name: str, default: Any = None, cast: Any = None) -> Any:
    """读取 query 参数，支持类型转换。"""
    source = _query_source()
    try:
        value = source.get(name, default)
    except Exception:
        return default
    if value is None or cast is None:
        return value
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def header_get(name: str, default: str = "") -> str:
    """读取请求头，宿主不支持 headers 或读取异常时返回默认值。

    开发者令牌改走 ``X-Dev-Token`` 请求头后，各版本请求对象的 headers
    实现差异（属性缺失、大小写敏感、非映射对象）都收敛在这里，避免
    handler 里散落 try/except。
    """
    if _request is None:
        return default
    try:
        headers = getattr(_request, "headers", None)
        if headers is None:
            return default
        getter = getattr(headers, "get", None)
        if not callable(getter):
            return default
        value = getter(name, default)
    except Exception:
        return default
    return default if value is None else str(value)


class JsonBodyError(ValueError):
    """请求体存在但不是合法 JSON 对象。

    与「请求体缺失/为空」区分开：后者仍返回调用方给的默认字典，前者由
    handler 转成明确的“请求体不是合法 JSON”错误，避免排障时被误报成
    “缺少参数”。
    """


async def _read_raw_body() -> str | None:
    """尽量读取原始请求体文本；宿主不提供读取接口时返回 None。"""
    # Quart 的 get_data() 与新版 AstrBot 请求对象都可能有该方法；取不到时
    # 交给下面的 json()/get_json() 回退分支处理。
    get_data = getattr(_request, "get_data", None)
    if not callable(get_data):
        return None
    try:
        raw = await get_data()
    except Exception:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        return raw
    return None


async def read_json(default: dict[str, Any] | None = None) -> dict[str, Any]:
    """读取并规范化 JSON 请求体，确保 handler 总是收到字典。

    新版 AstrBot 与 Quart 的解析方法和默认值参数不同，因此分别调用各自 API。
    请求对象不可用、正文为空或顶层不是对象时返回调用方提供的字典默认值；
    正文存在但 JSON 语法错误时抛出 :class:`JsonBodyError`，由 handler 返回
    统一错误响应。若默认值本身不是字典，则安全回落为空字典。

    Args:
        default: 请求体缺失或为空时使用的默认字典。

    Returns:
        解析得到的 JSON 对象，或规范化后的默认字典。

    Raises:
        JsonBodyError: 请求体存在但不是合法 JSON。
    """
    fallback = default if isinstance(default, dict) else {}
    if _request is None:
        return fallback
    text = await _read_raw_body()
    if text is not None:
        if not text.strip():
            return fallback
        try:
            data: Any = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise JsonBodyError(f"请求体不是合法 JSON：{exc}") from exc
        return data if isinstance(data, dict) else fallback
    # 旧宿主没有原始正文读取接口，无法区分「空正文」与「语法错误」；
    # 统一按语法错误上报，避免把用户输入问题误报成缺少参数。
    try:
        if NEW_WEB_API:
            try:
                data = await _request.json()
            except TypeError:
                # 个别版本要求显式传 default 关键字。
                data = await _request.json(default=fallback)
        else:  # pragma: no cover - 旧版本分支
            data = await _request.get_json()
    except Exception as exc:
        raise JsonBodyError(f"请求体不是合法 JSON：{exc}") from exc
    return data if isinstance(data, dict) else fallback


def ok(data: Any = None, **extra: Any) -> dict[str, Any]:
    """构造页面接口的成功响应，可附带 data 和额外顶层字段。"""
    # 页面接口统一用 success 字段传递成功状态，便于前端跨版本处理。
    payload: dict[str, Any] = {"success": True}
    if data is not None:
        payload["data"] = data
    payload.update(extra)
    return payload


def fail(message: str) -> dict[str, Any]:
    """构造供前端识别并显示的统一失败响应。"""
    # 错误也采用稳定 JSON 结构，由页面统一显示 error 文本。
    return {"success": False, "error": str(message)}
