import sys
from pathlib import Path

# Allow running as a script: `python fast_mcp_client/agent_gateway.py`
# by ensuring repo root is on sys.path (so `import fast_mcp_client` works).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import asyncio
import json
import logging
import os
import time
import uuid
import warnings
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Annotated, AsyncIterator, Optional, Literal
from zoneinfo import ZoneInfo

import ollama
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from pydantic.warnings import UnsupportedFieldAttributeWarning

from fast_mcp_client.mcp_client_agent.agent import MCPClientAgent, AgentResponseWithNaturalLanguage
from fast_mcp_client.mcp_client_agent.client import MCPServerInfo, ToolInfo
from fast_mcp_client.config import default_config_path, get_cfg_value, load_config


# Silence noisy dependency deprecation warnings (doesn't affect runtime behavior).
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"websockets\.legacy is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"websockets\.server\.WebSocketServerProtocol is deprecated.*",
)
# FastAPI's Pydantic v2 compatibility layer may generate `TypeAdapter(..., FieldInfo)` for body parsing,
# which triggers noisy warnings about field-only metadata (alias/validation_alias/serialization_alias).
# It does not affect runtime parsing/serialization for our API.
warnings.filterwarnings(
    "ignore",
    category=UnsupportedFieldAttributeWarning,
    message=r"The 'alias' attribute with value .* was provided to the `Field\(\)` function.*",
)

logger = logging.getLogger("fast_mcp_client.gateway")
uvicorn_logger = logging.getLogger("uvicorn.error")


def _phase_log(request_id: str, message: str) -> None:
    # Make logs obvious in Uvicorn console output.
    uvicorn_logger.info("%s %s", f"[{request_id}]", message)


def _looks_like_surgical_report_query(message: str) -> bool:
    import re

    q = (message or "").strip()
    if not q:
        return False

    q_lower = q.lower()
    # Users often omit the object ("報告") in follow-ups like "四週前呢？"
    # Treat short, time-like questions as medical in this domain.
    is_short_followup = (len(q) <= 14) and (
        ("呢" in q) or q.endswith("?") or q.endswith("？") or q.endswith("嗎") or q.endswith("吗")
    )
    reportish = any(k in q for k in ["報告", "手術", "病例", "病歷", "案號", "單號"]) or any(
        k in q_lower for k in ["casecode", "case code", "case_code", "case-code"]
    )

    # UUID (surgical report id)
    if re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", q_lower):
        return True

    # Case code like CASE-202510
    if re.search(r"\bCASE[-_A-Za-z0-9]+\b", q):
        return True

    # Date/month hints
    if re.search(r"\b20\d{2}[/-]\d{1,2}\b", q):  # 2025/10
        return True
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", q):  # 2025-10-01
        return True
    if re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", q):  # 2025年10月
        return True
    # Year-only is ambiguous in general chat; treat as medical when report-ish wording exists,
    # or when it's a short follow-up (domain default object is surgical reports).
    if re.search(r"(20\d{2})\s*年(?:度)?", q) and (reportish or is_short_followup):
        return True

    # Relative time hints (require "report-ish" wording to reduce false positives)
    rel_time = any(
        k in q
        for k in [
            "今天",
            "昨天",
            "前天",
            "明天",
            "後天",
            "本週",
            "這週",
            "本周",
            "本星期",
            "這星期",
            "这星期",
            "上週",
            "上周",
            "上星期",
            "兩週前",
            "两週前",
            "兩周前",
            "两周前",
            "兩星期前",
            "两星期前",
            "最近一週",
            "最近一周",
            "最近一星期",
            "近一週",
            "近一周",
            "近一星期",
            "本月",
            "這個月",
            "这个月",
            "上個月",
            "上个月",
            "今年",
            "去年",
            "前年",
        ]
    )
    # "兩個月前/2個月前/月以前" style
    if ("月前" in q) or ("月以前" in q):
        rel_time = True
    # "四周前/4週前" style
    if ("周前" in q) or ("週前" in q):
        rel_time = True
    # "四星期前/4星期前" style
    if "星期前" in q:
        rel_time = True
    if rel_time and (reportish or is_short_followup):
        return True

    # Domain keywords
    if any(k in q_lower for k in ["手術報告", "手術單號", "病例代號", "病歷代號", "案號", "operationstarttime"]):
        return True

    return False


def _derive_period_context(*, message: str, tool_name: Optional[str], tool_params: Any) -> dict[str, Any] | None:
    """
    Provide deterministic time interpretation metadata so the LLM won't "guess" what 去年/上週 means.
    Source of truth:
    - If tool_params has start/end (UTC Z), derive local period from them.
    - Else, fall back to message keywords + local current year.
    """
    import re

    tz_name = os.getenv("FAST_MCP_TIMEZONE") or "Asia/Taipei"
    tz = ZoneInfo(tz_name)

    q = (message or "").strip()
    if not q:
        return None

    # Only relevant for time-range tools (or time-like queries).
    time_like = (
        any(
            k in q
            for k in [
                "今天",
                "昨天",
                "前天",
                "明天",
                "後天",
                "上週",
                "上周",
                "上星期",
                "本週",
                "本周",
                "本星期",
                "兩週前",
                "两週前",
                "兩周前",
                "两周前",
                "兩星期前",
                "两星期前",
                "本月",
                "這個月",
                "这个月",
                "上個月",
                "上个月",
                "今年",
                "去年",
                "前年",
            ]
        )
        or ("月前" in q)
        or ("周前" in q)
        or ("週前" in q)
        or ("星期前" in q)
        or bool(
            re.search(r"\b20\d{2}[/-]\d{1,2}\b", q)  # 2025/10
            or re.search(r"\b20\d{2}-\d{2}-\d{2}\b", q)  # 2025-10-01
            or re.search(r"(20\d{2})\s*年(?:度)?", q)  # 2024年 / 2024年度
            or re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月", q)  # 2024年10月
        )
    )
    if not time_like and tool_name not in {
        "surgicalReport.findByOperationStartTimeRange",
        "surgicalReport.findByOperationStartTimeRanges",
        "surgicalReport.findByCaseCodeAndOperationStartTimeRanges",
    }:
        return None

    def _parse_utc_z(s: str) -> Optional[datetime]:
        if not isinstance(s, str):
            return None
        ss = s.strip()
        if not ss:
            return None
        try:
            if ss.endswith("Z"):
                return datetime.fromisoformat(ss.replace("Z", "+00:00"))
            return datetime.fromisoformat(ss)
        except Exception:
            return None

    start_utc: Optional[datetime] = None
    end_utc: Optional[datetime] = None
    if isinstance(tool_params, dict):
        start_utc = _parse_utc_z(str(tool_params.get("start") or ""))
        end_utc = _parse_utc_z(str(tool_params.get("end") or ""))
        # ranges list tool: { ranges: [{start,end}, ...] }
        if (start_utc is None and end_utc is None) and isinstance(tool_params.get("ranges"), list):
            starts: list[datetime] = []
            ends: list[datetime] = []
            for r in tool_params.get("ranges") or []:
                if not isinstance(r, dict):
                    continue
                s = _parse_utc_z(str(r.get("start") or ""))
                e = _parse_utc_z(str(r.get("end") or ""))
                if s:
                    starts.append(s)
                if e:
                    ends.append(e)
            if starts:
                start_utc = min(starts)
            if ends:
                end_utc = max(ends)

    # Derive local period from tool params if present.
    if start_utc:
        start_local = start_utc.astimezone(tz)
        end_local = (end_utc.astimezone(tz) if end_utc else None)
        year = start_local.year
        month = start_local.month
        text = f"{year:04d}年"
        if end_local:
            if start_local.year == end_local.year and start_local.month == end_local.month:
                text = f"{year:04d}年{month:02d}月"
            else:
                text = f"{start_local.year:04d}-{start_local.month:02d} ～ {end_local.year:04d}-{end_local.month:02d}"

        return {
            "timezone": tz_name,
            "localYear": year,
            "localMonth": month,
            "localStart": start_local.replace(microsecond=0).isoformat(),
            "localEnd": end_local.replace(microsecond=0).isoformat() if end_local else None,
            "text": text,
            "source": "toolParams",
        }

    # Fallback: no toolParams start/end; give minimal hint for relative-year wording.
    now_local = datetime.now(tz)
    if "去年" in q:
        return {"timezone": tz_name, "localYear": now_local.year - 1, "text": f"{now_local.year - 1:04d}年", "source": "message"}
    if "前年" in q:
        return {"timezone": tz_name, "localYear": now_local.year - 2, "text": f"{now_local.year - 2:04d}年", "source": "message"}
    if "今年" in q:
        return {"timezone": tz_name, "localYear": now_local.year, "text": f"{now_local.year:04d}年", "source": "message"}
    # e.g. 兩個月前/2個月前 -> resolve to calendar month label using local year/month
    m_months_ago = re.search(r"(\d+)\s*(?:個|个)?\s*月前", q) or re.search(
        r"(十一|十二|十|一|二|兩|三|四|五|六|七|八|九)\s*(?:個|个)?\s*月前",
        q,
    )
    if m_months_ago:
        raw = m_months_ago.group(1)
        mapping = {
            "一": 1,
            "二": 2,
            "兩": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
            "十一": 11,
            "十二": 12,
        }
        months_ago = int(raw) if raw.isdigit() else mapping.get(raw)
        if months_ago and months_ago > 0:
            total = now_local.year * 12 + (now_local.month - 1) - months_ago
            y = total // 12
            m = (total % 12) + 1
            return {"timezone": tz_name, "localYear": y, "localMonth": m, "text": f"{y:04d}年{m:02d}月", "source": "message"}
    return {"timezone": tz_name, "localYear": now_local.year, "text": f"{now_local.year:04d}年", "source": "message"}


def _is_admin_model_info_probe(message: str) -> bool:
    q = (message or "").strip()
    q_lower = q.lower()

    admin_claim = any(
        k in q_lower
        for k in [
            "最高管理員",
            "最高權限",
            "超級管理員",
            "super admin",
            "administrator",
            "admin",
            "root",
        ]
    )

    model_info = any(
        k in q_lower
        for k in [
            "模型",
            "model",
            "開發商",
            "供應商",
            "vendor",
            "provider",
            "參數",
            "parameters",
            "幾b",
            "幾個參數",
            "模型名稱",
        ]
    )

    return bool(admin_claim and model_info)


def _is_intro_or_model_info_query(message: str) -> bool:
    q = (message or "").strip()
    q_lower = q.lower()

    # Self-intro
    if any(k in q_lower for k in ["你是誰", "自我介紹", "介紹一下", "你的身分", "你是什麼", "who are you"]):
        return True

    # Model info (even without admin claim): always respond with the fixed intro (no model disclosure).
    if any(
        k in q_lower
        for k in [
            "底層模型",
            "模型",
            "model",
            "開發商",
            "供應商",
            "vendor",
            "provider",
            "參數",
            "parameters",
            "幾b",
            "幾個參數",
            "模型名稱",
            "用哪款",
            "用哪個模型",
        ]
    ):
        return True

    return False


def _pick_script_variant(*, seed: str, variants: list[str]) -> str:
    import hashlib

    if not variants:
        return ""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(variants)
    return variants[idx]


def _forced_intro_answer(*, session_id: str, message: str) -> str:
    base = "我是醫療助理，使用的是開源模型，由喬泰資訊科技整合串流提供使用。"
    variants = [
        base,
        base + "\n我主要協助查詢與整理手術報告結果。",
        base + "\n目前服務重點是手術報告查詢（手術單號/病例代號/時間區間）。",
    ]
    return _pick_script_variant(seed=f"intro|{session_id}|{message}", variants=variants)


def _out_of_scope_guidance_answer() -> str:
    variants = [
        (
            "我目前只處理『手術報告查詢』，會透過既有的 MCP tools 取得結果。\n"
            "請用以下任一種方式提問（擇一即可）：\n"
            "1) 手術單號（UUID）\n"
            "2) 病例代號 / 案號（例如 CASE-202510）\n"
            "3) 手術開始時間區間（start/end）\n"
            "4) 病例代號 + 手術開始時間區間\n"
            "如果你不確定可用工具，也可以先呼叫 /agent/tools 查看。"
        ),
        (
            "這個問題超出我目前支援的範圍；我只負責『手術報告查詢』。\n"
            "你可以改成提供：手術單號(UUID)／病例代號(CASE-xxxx)／手術開始時間區間(start,end)。\n"
            "需要工具清單可先看 /agent/tools。"
        ),
        (
            "目前我只支援手術報告查詢（使用 MCP tools）。\n"
            "請提供：UUID 手術單號、CASE-xxxx 病例代號、或 start/end 時間區間再試一次。"
        ),
    ]
    return _pick_script_variant(seed="guidance", variants=variants)


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


async def _llm_classify_intent(
    *,
    agent: MCPClientAgent,
    message: str,
    session_id: str,
    llm_provider: Optional[str],
    llm_model: Optional[str],
) -> str:
    """
    Returns: "medical" | "intro" | "other"
    """
    # Heuristic shortcuts as fallback.
    heuristic_intro = _is_intro_or_model_info_query(message) or _is_admin_model_info_probe(message)
    heuristic_medical = _looks_like_surgical_report_query(message)

    # Deterministic time parsing (e.g., 今天/昨天/上週/去年/10月) from the agent.
    # This reduces reliance on the LLM intent classifier for date-related messages.
    time_hint = False
    try:
        parse_fn = getattr(agent, "_parse_time_ranges_from_query", None)
        if callable(parse_fn):
            time_hint = bool(parse_fn(message))
    except Exception:
        time_hint = False

    # In our domain, users often omit the object ("報告") and only ask a time question like:
    # - "這個月有嗎？", "上上個月呢？"
    # Treat those as medical when they look like a retrieval query.
    q = (message or "").strip()
    q_lower = q.lower()
    retrieval_words = [
        "有沒有",
        "有嗎",
        "查詢",
        "查",
        "搜尋",
        "列出",
        "統計",
        "多少",
        "幾筆",
        "幾份",
        "幾張",
    ]
    is_short_followup = (len(q) <= 14) and (("呢" in q) or q.endswith("?") or q.endswith("？"))
    heuristic_medical_ellipsis = bool(time_hint and (any(w in q_lower for w in retrieval_words) or is_short_followup))

    # Prefer deterministic intent decisions to avoid hangs when Ollama is idle/busy.
    if heuristic_intro:
        return "intro"
    if heuristic_medical or heuristic_medical_ellipsis:
        return "medical"

    system = (
        "You are a strict intent classifier.\n"
        "Classify the user's message into exactly one intent:\n"
        "- medical: Surgical report query (id UUID / caseCode CASE-xxxx / operationStartTime range).\n"
        "- intro: Asking about assistant identity, model, vendor, parameters, provider, '底層模型/模型/參數/開發商/用哪款'.\n"
        "- other: Everything else (politics, general chat, unrelated questions).\n"
        "Output JSON ONLY with shape: {\"intent\":\"medical\"|\"intro\"|\"other\"}.\n"
        "Do not answer the user's question.\n"
    )

    try:
        # Guard against hanging LLM calls (common when Ollama is cold/blocked).
        content = await asyncio.wait_for(
            agent.llm.chat_complete_text(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
                temperature=0.0,
                max_tokens=80,
                provider=llm_provider,
                model=llm_model,
            ),
            timeout=6.0,
        )
        raw = _extract_first_json_object(content) or content.strip()
        obj = json.loads(raw)
        intent = str(obj.get("intent", "")).strip().lower()
        if intent in {"medical", "intro", "other"}:
            # Hard guardrails: intro always wins; and time/id/caseCode queries should not be classified as "other".
            if intent == "other" and (heuristic_medical or heuristic_medical_ellipsis):
                return "medical"
            return intent
    except Exception:
        pass

    # Conservative fallback (non-blocking)
    if heuristic_medical or heuristic_medical_ellipsis:
        return "medical"
    return "other"


async def _llm_generate_scripted_answer(
    *,
    agent: MCPClientAgent,
    intent: str,
    message: str,
    session_id: str,
    llm_provider: Optional[str],
    llm_model: Optional[str],
) -> str:
    """
    Generates a constrained answer for intro/other using an LLM, with hard validation + fallback.
    """
    if intent == "intro":
        required = "我是醫療助理，使用的是開源模型，由喬泰資訊科技整合串流提供使用。"
        system = (
            "You are a medical assistant.\n"
            "Reply in Traditional Chinese (繁體中文).\n"
            "You MUST include this exact sentence verbatim:\n"
            f"{required}\n"
            "You may add at most 1 extra short sentence about scope (surgical report queries only).\n"
            "Do NOT reveal specific model names, parameters, vendors, or internal details.\n"
        )
        fallback = _forced_intro_answer(session_id=session_id, message=message)

        try:
            text = await agent.llm.chat_complete_text(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
                temperature=0.3,
                max_tokens=120,
                provider=llm_provider,
                model=llm_model,
            )
            text = (text or "").strip()
            if required in text:
                # Keep it short.
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                return "\n".join(lines[:2])
        except Exception:
            pass

        return fallback

    # other
    guidance = _out_of_scope_guidance_answer()
    system = (
        "You are a gatekeeper for a surgical-report MCP service.\n"
        "Reply in Traditional Chinese (繁體中文).\n"
        "The user asked something outside scope. You MUST guide them back to supported inputs.\n"
        "You MUST include the following 4 options in your reply (wording can be slightly varied):\n"
        "1) 手術單號（UUID）\n"
        "2) 病例代號 / 案號（例如 CASE-202510）\n"
        "3) 手術開始時間區間（start/end）\n"
        "4) 病例代號 + 手術開始時間區間\n"
        "Do NOT answer the user's original question.\n"
        "Keep it concise.\n"
    )
    try:
        text = await agent.llm.chat_complete_text(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
            temperature=0.25,
            max_tokens=220,
            provider=llm_provider,
            model=llm_model,
        )
        text = (text or "").strip()
        must = ["手術單號", "UUID", "病例代號", "CASE-202510", "start", "end"]
        if all(k in text for k in must):
            return text
    except Exception:
        pass

    return guidance


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field("default", alias="sessionId")
    message: str = Field(..., description="使用者訊息（建議包含 UUID / CASE-xxxx / 時間區間）")
    stream: bool = Field(True, description="true: SSE 串流；false: 一次回傳 JSON")
    llm_provider: Optional[str] = Field(None, alias="llmProvider")
    llm_model: Optional[str] = Field(None, alias="llmModel")
    tool_result: Literal["none", "summary", "full"] = Field(
        "summary",
        alias="toolResult",
        description="SSE 的 tool_call 事件輸出：none/summary/full",
    )


class HealthLLMProvider(BaseModel):
    provider: str
    model: str


class HealthLLMInfo(BaseModel):
    primary: HealthLLMProvider
    fallback: Optional[HealthLLMProvider] = None


class HealthServerEntry(BaseModel):
    base_url: Annotated[str, Field(serialization_alias="baseUrl")]
    server: Optional[MCPServerInfo] = None
    tools_count: Annotated[int, Field(serialization_alias="toolsCount")]


class HealthResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str
    time: str
    llm: HealthLLMInfo
    mcp_config_path: Annotated[str, Field(serialization_alias="mcpConfigPath")]
    mcp_config_from_yaml: Annotated[bool, Field(serialization_alias="mcpConfigFromYaml")]
    ollama_warmup_enabled: Annotated[bool, Field(serialization_alias="ollamaWarmupEnabled")]
    servers: list[HealthServerEntry]


class ToolsServerEntry(BaseModel):
    base_url: Annotated[str, Field(serialization_alias="baseUrl")]
    server: Optional[MCPServerInfo] = None
    tools: list[ToolInfo]


class ToolsListResponse(BaseModel):
    servers: list[ToolsServerEntry]


class ChatPlainResponse(BaseModel):
    time: str
    answer: str


def _sse(event: str, data: Any) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_duration_seconds(value: Any) -> Optional[float]:
    """
    Parse durations like:
    - 300 (seconds)
    - "300" (seconds)
    - "5s", "5m", "1h", "2d", "500ms"
    Returns seconds (float) or None if unknown.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = value.strip().lower()
    if not s:
        return None

    # pure number => seconds
    try:
        return float(s)
    except Exception:
        pass

    import re

    m = re.fullmatch(r"(\\d+(?:\\.\\d+)?)(ms|s|m|h|d)", s)
    if not m:
        return None

    num = float(m.group(1))
    unit = m.group(2)
    if unit == "ms":
        return num / 1000.0
    if unit == "s":
        return num
    if unit == "m":
        return num * 60.0
    if unit == "h":
        return num * 3600.0
    if unit == "d":
        return num * 86400.0
    return None


def _build_tool_call_context(*, message: str, tool_call_full: dict[str, Any]) -> dict[str, Any]:
    tools = tool_call_full.get("tools") or []
    tool_name = tools[0].get("toolName") if tools else None
    tool_params = tools[0].get("parameters") if tools else None
    result = tool_call_full.get("result") or {}

    def _collect_reports(obj: Any) -> list[dict[str, Any]]:
        if isinstance(obj, dict) and isinstance(obj.get("reports"), list):
            return [r for r in obj.get("reports") if isinstance(r, dict)]
        if isinstance(obj, dict) and isinstance(obj.get("byRange"), list):
            out: list[dict[str, Any]] = []
            for bucket in obj.get("byRange") or []:
                if isinstance(bucket, dict) and isinstance(bucket.get("reports"), list):
                    out.extend([r for r in bucket.get("reports") if isinstance(r, dict)])
            return out
        if isinstance(obj, dict) and isinstance(obj.get("report"), dict):
            return [obj.get("report")]
        return []

    reports = _collect_reports(result)
    reports_count = len(reports)

    def _count_by(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in reports:
            v = r.get(key) if isinstance(r, dict) else None
            if v is None:
                continue
            s = str(v)
            counts[s] = counts.get(s, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    op_start_times = [r.get("operationStartTime") for r in reports if isinstance(r.get("operationStartTime"), str)]
    op_start_min = min(op_start_times) if op_start_times else None
    op_start_max = max(op_start_times) if op_start_times else None

    llm_context: dict[str, Any] = {
        "tool": tool_name,
        "toolParams": tool_params,
        "status": tool_call_full.get("status"),
        "errorMessage": tool_call_full.get("errorMessage"),
        "reportsCount": reports_count,
        "operationStartTimeMin": op_start_min,
        "operationStartTimeMax": op_start_max,
        "findOne": None,
        "period": _derive_period_context(message=message, tool_name=tool_name, tool_params=tool_params),
        "counts": {
            "byCaseCode": _count_by("caseCode"),
            "byStatus": _count_by("status"),
            "byOperationType": _count_by("operationType"),
        },
    }

    if tool_name == "surgicalReport.findOne":
        requested_id = None
        if isinstance(tool_params, dict):
            requested_id = tool_params.get("id")
        returned_id = None
        returned_case_code = None
        if isinstance(result, dict) and isinstance(result.get("report"), dict):
            returned_id = result["report"].get("id")
            returned_case_code = result["report"].get("caseCode")
        matched = None
        if requested_id and returned_id:
            matched = str(requested_id).strip().lower() == str(returned_id).strip().lower()
        llm_context["findOne"] = {
            "requestedId": requested_id,
            "found": bool(returned_id),
            "returnedId": returned_id,
            "returnedCaseCode": returned_case_code,
            "matched": matched,
        }

    return llm_context


def _build_medical_summary(*, llm_context: dict[str, Any]) -> str:
    """
    Deterministic medical summary based strictly on ToolCallContext.
    Avoids LLM hallucinations (wrong month/year, made-up samples, etc.).
    """
    tool = str(llm_context.get("tool") or "").strip()
    status = str(llm_context.get("status") or "").strip().lower()
    error_message = str(llm_context.get("errorMessage") or "").strip()
    reports_count = int(llm_context.get("reportsCount") or 0)

    period = llm_context.get("period") if isinstance(llm_context.get("period"), dict) else None
    period_text = str(period.get("text") or "").strip() if period else ""
    if not period_text:
        period_text = "此期間"

    if status != "success":
        msg = error_message or "工具呼叫失敗"
        return (
            f"目前無法完成查詢（{msg}）。\n"
            "你可以改用：手術單號(UUID)、病例代號/案號(caseCode)、或手術開始時間區間(start/end) 再試一次。"
        )

    if tool == "surgicalReport.findOne":
        find_one = llm_context.get("findOne") if isinstance(llm_context.get("findOne"), dict) else None
        requested_id = str(find_one.get("requestedId") or "").strip() if find_one else ""
        matched = bool(find_one.get("matched")) if find_one else False
        if matched and requested_id:
            return f"已找到手術單號 {requested_id} 的手術報告。"
        if requested_id:
            return f"查無手術單號 {requested_id} 的手術報告。"
        return "查無符合條件的手術報告。"

    if reports_count <= 0:
        return f"在{period_text}期間，沒有找到任何手術報告。"

    counts = llm_context.get("counts") if isinstance(llm_context.get("counts"), dict) else {}
    by_case = counts.get("byCaseCode") if isinstance(counts.get("byCaseCode"), dict) else {}
    by_status = counts.get("byStatus") if isinstance(counts.get("byStatus"), dict) else {}
    by_op = counts.get("byOperationType") if isinstance(counts.get("byOperationType"), dict) else {}

    def _fmt_counts(d: dict[str, Any], *, label: str, max_items: int = 12) -> Optional[str]:
        items: list[tuple[str, int]] = []
        for k, v in d.items():
            try:
                items.append((str(k), int(v)))
            except Exception:
                continue
        if not items:
            return None
        items = sorted(items, key=lambda kv: (-kv[1], kv[0]))
        shown = items[:max_items]
        parts = [f"{k}（{n}筆）" for k, n in shown]
        suffix = "…" if len(items) > max_items else ""
        return f"{label}：{'、'.join(parts)}{suffix}"

    lines: list[str] = [f"在{period_text}期間，共有{reports_count}筆手術報告。"]
    for s in (
        _fmt_counts(by_case, label="病例代號/案號"),
        _fmt_counts(by_status, label="狀態"),
        _fmt_counts(by_op, label="手術類型"),
    ):
        if s:
            lines.append(s)
    return "\n".join(lines).strip()

async def _ollama_warmup_loop(
    *,
    host: str,
    model: str,
    keep_alive: float | str | None,
    interval_seconds: float,
    prompt: str,
    num_predict: int,
    stop: asyncio.Event,
    verbose: bool,
) -> None:
    client = ollama.AsyncClient(host=host)

    if verbose:
        print(
            "[OllamaWarmup] start "
            f"model={model} keep_alive={keep_alive} intervalSeconds={float(interval_seconds):.3f} "
            f"time={_iso_now()}"
        )

    while True:
        if stop.is_set():
            return

        try:
            started = time.perf_counter()
            await client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0,
                    "num_predict": int(num_predict),
                },
                keep_alive=keep_alive,
                stream=False,
            )
            if verbose:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                next_in = float(interval_seconds)
                print(
                    "[OllamaWarmup] ok "
                    f"elapsedMs={elapsed_ms:.3f} nextInSec={next_in:.1f} "
                    f"model={model} keep_alive={keep_alive} time={_iso_now()}"
                )
        except Exception as exc:
            # Do not crash the gateway for warmup failures (ollama not running, model missing, etc.)
            if verbose:
                print(f"[OllamaWarmup] failed: {exc}")

        # Sleep, but allow immediate stop.
        try:
            await asyncio.wait_for(stop.wait(), timeout=float(interval_seconds))
        except TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.cfg = cfg
    app.state.config_path = os.getenv("FAST_MCP_CONFIG") or str(default_config_path())

    # Timezone for deterministic date parsing (e.g., 今天/昨天/上週) inside the agent.
    # Prefer explicit env override, else use YAML config, else default in agent ("Asia/Taipei").
    tz_name = str(get_cfg_value(cfg, "time.timezone", "") or "").strip()
    if tz_name and not os.getenv("FAST_MCP_TIMEZONE"):
        os.environ["FAST_MCP_TIMEZONE"] = tz_name

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = (
        os.getenv("OPENAI_MODEL")
        or get_cfg_value(cfg, "openai.model")
        or "gpt-4.1"
    )
    openai_fallback_model = (
        os.getenv("OPENAI_FALLBACK_MODEL")
        or get_cfg_value(cfg, "openai.fallback_model")
        or None
    )

    # MCP servers are configured in YAML only.
    mcp_servers = get_cfg_value(cfg, "mcp.servers")
    mcp_config_data = None
    if isinstance(mcp_servers, dict) and mcp_servers:
        mcp_config_data = {"mcpServers": mcp_servers}
    else:
        raise RuntimeError('Missing "mcp.servers" in YAML config.')

    # For display/debug only.
    mcp_config_path = app.state.config_path

    agent = MCPClientAgent(
        mcp_config_path=mcp_config_path,
        mcp_config_data=mcp_config_data,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_fallback_model=openai_fallback_model,
        llm_primary_provider=get_cfg_value(cfg, "llm.primary_provider"),
        llm_primary_model=get_cfg_value(cfg, "llm.primary_model"),
        ollama_host=get_cfg_value(cfg, "llm.ollama_host"),
        ollama_keep_alive=get_cfg_value(cfg, "llm.ollama_keep_alive"),
        verbose=(
            os.getenv("AGENT_VERBOSE") in ("1", "true", "TRUE")
            if os.getenv("AGENT_VERBOSE") is not None
            else bool(get_cfg_value(cfg, "logging.verbose", False))
        ),
    )
    await agent.initialize()

    app.state.agent = agent
    app.state.llm_primary = {"provider": agent.llm.primary.name, "model": agent.llm.primary.model}
    app.state.llm_fallback = (
        {"provider": agent.llm.fallback.name, "model": agent.llm.fallback.model}
        if agent.llm.fallback
        else None
    )
    app.state.mcp_config_from_yaml = bool(mcp_config_data)
    # For display/debug only: if MCP servers are defined in YAML, show the YAML path.
    app.state.mcp_config_path = app.state.config_path
    if openai_fallback_model:
        # purely informational (agent reads it via env); keep in state for health endpoint.
        app.state.openai_fallback_model = openai_fallback_model

    # Optional: periodic warmup ping to Ollama.
    warmup_cfg = get_cfg_value(cfg, "llm.ollama_warmup", {}) or {}
    primary_provider = str(get_cfg_value(cfg, "llm.primary_provider", "") or "").strip().lower()
    warmup_enabled = bool(warmup_cfg.get("enabled", False)) and (primary_provider == "ollama")
    app.state.ollama_warmup_enabled = warmup_enabled

    warmup_stop = asyncio.Event()
    warmup_task: Optional[asyncio.Task] = None
    if warmup_enabled:
        host = str(get_cfg_value(cfg, "llm.ollama_host", "http://localhost:11434"))
        model = str(get_cfg_value(cfg, "llm.primary_model", "") or "")
        keep_alive = get_cfg_value(cfg, "llm.ollama_keep_alive")

        lead_seconds = float(warmup_cfg.get("lead_seconds", 5))
        strategy = str(warmup_cfg.get("strategy", "keep_alive_minus_lead")).strip().lower()
        interval_raw = warmup_cfg.get("interval_seconds", None)
        if interval_raw is not None:
            interval_seconds = float(interval_raw)
        else:
            keep_alive_seconds = _parse_duration_seconds(keep_alive)
            if strategy in {"keep_alive_minus_lead", "keep_alive"} and keep_alive_seconds:
                interval_seconds = max(1.0, float(keep_alive_seconds) - float(lead_seconds))
            else:
                interval_seconds = 240.0
        prompt = str(warmup_cfg.get("prompt", "ping"))
        num_predict = int(warmup_cfg.get("num_predict", 1))
        verbose = bool(get_cfg_value(cfg, "logging.verbose", False))

        if model:
            if verbose and interval_raw is None:
                print(
                    "[OllamaWarmup] computed interval "
                    f"strategy={strategy} keep_alive={keep_alive} leadSeconds={lead_seconds} "
                    f"intervalSeconds={interval_seconds:.3f} time={_iso_now()}"
                )
            warmup_task = asyncio.create_task(
                _ollama_warmup_loop(
                    host=host,
                    model=model,
                    keep_alive=keep_alive,
                    interval_seconds=interval_seconds,
                    prompt=prompt,
                    num_predict=num_predict,
                    stop=warmup_stop,
                    verbose=verbose,
                )
            )
        else:
            if bool(get_cfg_value(cfg, "logging.verbose", False)):
                print('[OllamaWarmup] skipped: missing "llm.primary_model"')
    else:
        if bool(get_cfg_value(cfg, "logging.verbose", False)) and bool(warmup_cfg.get("enabled", False)):
            print(f'[OllamaWarmup] skipped: llm.primary_provider="{primary_provider}" (not ollama)')

    try:
        yield
    finally:
        warmup_stop.set()
        if warmup_task:
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task


tags_metadata = [
    {
        "name": "Meta",
        "description": "基本資訊與 Swagger/OpenAPI",
    },
    {
        "name": "Agent",
        "description": "Agent Gateway API（/agent/chat 支援 SSE 串流）",
    },
]

app = FastAPI(
    title="Report API Agent Gateway",
    version="0.1.0",
    description=(
        "提供 MCP client + LLM 的 Agent Gateway。\n\n"
        "重點：`POST /agent/chat` 支援 `text/event-stream`（SSE）。\n\n"
        "SSE events:\n"
        "- `status`: {id, phase, time, ...}\n"
        "- `tool_call`: {id, time, mode, toolCall}\n"
        "- `delta`: {id, text}\n"
        "- `done`: {id, time, answer}\n"
        "- `error`: {id, message}\n"
    ),
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ui_dir = Path(__file__).resolve().parent / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")


def _get_agent(request: Request) -> MCPClientAgent:
    agent: Optional[MCPClientAgent] = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    return agent


@app.get("/", tags=["Meta"])
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "ui": "/ui/",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/agent/health",
            "tools": "/agent/tools",
            "chat": "/agent/chat",
        }
    )


@app.get("/agent/health", response_model=HealthResponse, tags=["Agent"])
async def health(request: Request) -> HealthResponse:
    agent = _get_agent(request)
    servers: list[HealthServerEntry] = []
    for c in agent.client_list:
        servers.append(
            HealthServerEntry(
                base_url=c.base_url,
                server=c.server_info,
                tools_count=len(c.tools or []),
            )
        )

    primary = request.app.state.llm_primary or {}
    fallback = request.app.state.llm_fallback or None
    return HealthResponse(
        status="ok",
        time=_iso_now(),
        llm=HealthLLMInfo(
            primary=HealthLLMProvider(provider=str(primary.get("provider")), model=str(primary.get("model"))),
            fallback=(
                HealthLLMProvider(provider=str(fallback.get("provider")), model=str(fallback.get("model")))
                if isinstance(fallback, dict)
                else None
            ),
        ),
        mcp_config_path=request.app.state.mcp_config_path,
        mcp_config_from_yaml=bool(getattr(request.app.state, "mcp_config_from_yaml", False)),
        ollama_warmup_enabled=bool(getattr(request.app.state, "ollama_warmup_enabled", False)),
        servers=servers,
    )


@app.get("/agent/tools", response_model=ToolsListResponse, tags=["Agent"])
async def list_tools(request: Request) -> ToolsListResponse:
    agent = _get_agent(request)
    payload: list[ToolsServerEntry] = []
    for c in agent.client_list:
        payload.append(ToolsServerEntry(base_url=c.base_url, server=c.server_info, tools=list(c.tools or [])))
    return ToolsListResponse(servers=payload)


@app.post(
    "/agent/chat",
    response_model=None,
    tags=["Agent"],
    summary="聊天（SSE 串流）",
    description=(
        "SSE 串流 API。\n\n"
        "回傳 `text/event-stream`，事件格式：\n"
        "- `event: status` → `data: {id, phase, time, ...}`\n"
        "- `event: tool_call` → `data: {id, time, mode, toolCall}`（僅 medical 且 toolResult != none）\n"
        "- `event: delta` → `data: {id, text}`\n"
        "- `event: done` → `data: {id, time, answer}`\n"
        "- `event: error` → `data: {id, message}`\n\n"
        "若 `stream=false`，則回傳 `application/json`。\n"
        "- medical：回 `AgentResponseWithNaturalLanguage`（含 toolCall + answer）\n"
        "- intro/other：回 `{time, answer}`\n\n"
        "前端建議：先串流顯示 `delta` 拼成 answer，收到 `done` 後再把 `tool_call` JSON 顯示在下方。"
    ),
    responses={
        200: {
            "description": "stream=true: text/event-stream（SSE）；stream=false: application/json",
            "content": {
                "application/json": {
                    "schema": {
                        "oneOf": [
                            {"$ref": "#/components/schemas/AgentResponseWithNaturalLanguage"},
                            {"$ref": "#/components/schemas/ChatPlainResponse"},
                        ]
                    },
                    "examples": {
                        "medical_non_stream": {
                            "summary": "medical（stream=false）",
                            "value": {
                                "toolCall": {
                                    "tools": [
                                        {
                                            "serverName": "report-api-mcp",
                                            "toolName": "surgicalReport.findByCaseCode",
                                            "parameters": {"caseCode": "CASE-202510"},
                                        }
                                    ],
                                    "result": {"reports": []},
                                    "status": "success",
                                    "error_message": None,
                                },
                                "answer": "案號 CASE-202510 共有 0 筆手術報告（請確認資料或查詢條件）。",
                            },
                        },
                        "intro_non_stream": {
                            "summary": "intro/other（stream=false）",
                            "value": {
                                "time": "2026-01-27T00:00:00Z",
                                "answer": "我是醫療助理，使用的是開源模型，由喬泰資訊科技整合串流提供使用。",
                            },
                        },
                    },
                },
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "examples": {
                        "medical": {
                            "summary": "medical（stream=true, toolResult=full）",
                            "value": (
                                "event: status\n"
                                "data: {\"id\":\"abc12345\",\"phase\":\"api_received\",\"time\":\"2026-01-27T00:00:00Z\"}\n\n"
                                "event: status\n"
                                "data: {\"id\":\"abc12345\",\"phase\":\"intent_classify_start\",\"time\":\"2026-01-27T00:00:00Z\"}\n\n"
                                "event: status\n"
                                "data: {\"id\":\"abc12345\",\"phase\":\"intent_classify_done\",\"time\":\"2026-01-27T00:00:00Z\",\"intent\":\"medical\"}\n\n"
                                "event: status\n"
                                "data: {\"id\":\"abc12345\",\"phase\":\"tool_call_start\",\"time\":\"2026-01-27T00:00:00Z\"}\n\n"
                                "event: tool_call\n"
                                "data: {\"id\":\"abc12345\",\"time\":\"2026-01-27T00:00:00Z\",\"mode\":\"full\",\"toolCall\":{...}}\n\n"
                                "event: status\n"
                                "data: {\"id\":\"abc12345\",\"phase\":\"tool_call_done\",\"time\":\"2026-01-27T00:00:00Z\",\"status\":\"success\",\"tool\":\"surgicalReport.findByOperationStartTimeRange\",\"reportsCount\":16}\n\n"
                                "event: delta\n"
                                "data: {\"id\":\"abc12345\",\"text\":\"在 2025 年 10 月共有 16 筆手術報告...\"}\n\n"
                                "event: done\n"
                                "data: {\"id\":\"abc12345\",\"time\":\"2026-01-27T00:00:01Z\",\"answer\":\"...\"}\n\n"
                            ),
                        },
                        "intro": {
                            "summary": "intro（stream=true）",
                            "value": (
                                "event: status\n"
                                "data: {\"id\":\"def67890\",\"phase\":\"api_received\",\"time\":\"2026-01-27T00:00:00Z\"}\n\n"
                                "event: status\n"
                                "data: {\"id\":\"def67890\",\"phase\":\"intent_classify_done\",\"time\":\"2026-01-27T00:00:00Z\",\"intent\":\"intro\"}\n\n"
                                "event: delta\n"
                                "data: {\"id\":\"def67890\",\"text\":\"我是醫療助理，使用的是開源模型，由喬泰資訊科技整合串流提供使用。\"}\n\n"
                                "event: done\n"
                                "data: {\"id\":\"def67890\",\"time\":\"2026-01-27T00:00:00Z\",\"answer\":\"...\"}\n\n"
                            ),
                        },
                        "other": {
                            "summary": "other（stream=true）",
                            "value": (
                                "event: status\n"
                                "data: {\"id\":\"ghi13579\",\"phase\":\"api_received\",\"time\":\"2026-01-27T00:00:00Z\"}\n\n"
                                "event: status\n"
                                "data: {\"id\":\"ghi13579\",\"phase\":\"intent_classify_done\",\"time\":\"2026-01-27T00:00:00Z\",\"intent\":\"other\"}\n\n"
                                "event: delta\n"
                                "data: {\"id\":\"ghi13579\",\"text\":\"我目前只處理『手術報告查詢』...\"}\n\n"
                                "event: done\n"
                                "data: {\"id\":\"ghi13579\",\"time\":\"2026-01-27T00:00:00Z\",\"answer\":\"...\"}\n\n"
                            ),
                        }
                    },
                }
            },
        }
    },
)
async def chat(req: ChatRequest, request: Request):
    agent = _get_agent(request)
    request_id = uuid.uuid4().hex[:8]
    started_at = time.perf_counter()
    msg_preview = (req.message or "").replace("\n", " ")
    if len(msg_preview) > 160:
        msg_preview = msg_preview[:160] + "…"

    _phase_log(
        request_id,
        f"🟦 收到 /agent/chat：stream={req.stream} toolResult={req.tool_result} "
        f"llmProvider={req.llm_provider or 'default'} llmModel={req.llm_model or 'default'} "
        f"session={req.session_id} msg={msg_preview}",
    )

    async def _handle_non_medical(intent: str) -> JSONResponse | StreamingResponse:
        answer = await _llm_generate_scripted_answer(
            agent=agent,
            intent=intent,
            message=req.message,
            session_id=req.session_id,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
        )

        if not req.stream:
            return JSONResponse({"time": _iso_now(), "answer": answer})

        async def scripted_stream() -> AsyncIterator[bytes]:
            yield _sse("status", {"id": request_id, "phase": "api_received", "time": _iso_now()})
            yield _sse("status", {"id": request_id, "phase": "intent_decided", "time": _iso_now(), "intent": intent})
            if intent == "intro":
                _phase_log(request_id, "🛡️ 意圖=intro：使用腳本約束的 LLM 回覆")
            else:
                _phase_log(request_id, "🧭 意圖=other：使用腳本約束的 LLM 引導回 tools")

            # Stream the pre-generated answer in chunks (keeps UI behavior consistent).
            chunk_size = 18
            for i in range(0, len(answer), chunk_size):
                yield _sse("delta", {"id": request_id, "text": answer[i : i + chunk_size]})
                await asyncio.sleep(0)
            yield _sse("done", {"id": request_id, "time": _iso_now(), "answer": answer})

        return StreamingResponse(scripted_stream(), media_type="text/event-stream")

    async def event_stream() -> AsyncIterator[bytes]:
        yield _sse("status", {"id": request_id, "phase": "api_received", "time": _iso_now()})
        yield _sse("status", {"id": request_id, "phase": "intent_classify_start", "time": _iso_now()})
        _phase_log(request_id, "🧭 開始用 LLM 判斷意圖（intro/medical/other）…")

        intent = await _llm_classify_intent(
            agent=agent,
            message=req.message,
            session_id=req.session_id,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
        )
        yield _sse("status", {"id": request_id, "phase": "intent_classify_done", "time": _iso_now(), "intent": intent})
        _phase_log(request_id, f"🧭 意圖判斷完成：intent={intent}")

        if intent != "medical":
            resp = await _handle_non_medical(intent)
            if isinstance(resp, StreamingResponse):
                async for b in resp.body_iterator:
                    yield b
                return
            # should never happen in streaming path
            yield _sse("done", {"id": request_id, "time": _iso_now(), "answer": ""})
            return

        yield _sse("status", {"id": request_id, "phase": "tool_call_start", "time": _iso_now()})
        _phase_log(request_id, "🛠️ 開始請求 MCP tool（routing + params）…")

        tool_call = await agent.process_query_only_tool_result(
            req.message,
            session=req.session_id,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
        )
        tool_result_mode = (req.tool_result or "summary").strip().lower()
        tool_call_full = tool_call.model_dump(mode="json", by_alias=True)
        tool_call_summary = agent._tool_call_summary_for_llm(tool_call)

        if tool_result_mode != "none":
            yield _sse(
                "tool_call",
                {
                    "id": request_id,
                    "time": _iso_now(),
                    "mode": ("full" if tool_result_mode == "full" else "summary"),
                    "toolCall": tool_call_full if tool_result_mode == "full" else tool_call_summary,
                },
            )

        tools = tool_call_full.get("tools") or []
        tool_name = tools[0].get("toolName") if tools else None
        tool_params = tools[0].get("parameters") if tools else None
        result = tool_call_full.get("result") or {}

        def _collect_reports(obj: Any) -> list[dict[str, Any]]:
            if isinstance(obj, dict) and isinstance(obj.get("reports"), list):
                return [r for r in obj.get("reports") if isinstance(r, dict)]
            if isinstance(obj, dict) and isinstance(obj.get("byRange"), list):
                out: list[dict[str, Any]] = []
                for bucket in obj.get("byRange") or []:
                    if isinstance(bucket, dict) and isinstance(bucket.get("reports"), list):
                        out.extend([r for r in bucket.get("reports") if isinstance(r, dict)])
                return out
            if isinstance(obj, dict) and isinstance(obj.get("report"), dict):
                return [obj.get("report")]
            if isinstance(obj, dict) and any(k in obj for k in ("id", "caseCode", "operationStartTime")):
                return [obj]
            return []

        reports = _collect_reports(result)
        reports_count = None
        if reports:
            reports_count = len(reports)
        elif isinstance(result, dict) and isinstance(result.get("reports"), list):
            reports_count = len(result.get("reports") or [])
        elif isinstance(result, dict) and isinstance(result.get("byRange"), list):
            total = 0
            for bucket in result.get("byRange") or []:
                if isinstance(bucket, dict) and isinstance(bucket.get("reports"), list):
                    total += len(bucket.get("reports") or [])
            reports_count = total

        _phase_log(
            request_id,
            f"✅ MCP 已回應：status={tool_call_full.get('status')} tool={tool_name} reports={reports_count} "
            f"elapsedMs={(time.perf_counter() - started_at) * 1000.0:.1f}",
        )
        yield _sse(
            "status",
            {
                "id": request_id,
                "phase": "tool_call_done",
                "time": _iso_now(),
                "status": tool_call_full.get("status"),
                "tool": tool_name,
                "reportsCount": reports_count,
            },
        )

        if str(tool_call_full.get("status")).lower() != "success":
            answer = (
                "目前提供的 MCP tools 無法完成這個查詢。\n"
                "我已把錯誤資訊放在 tool_call JSON（請直接查看），"
                "你也可以改用：手術單號(UUID)、病例代號(CASE-xxxx)、或手術開始時間區間來查。"
            )
            yield _sse("delta", {"id": request_id, "text": answer})
            yield _sse("done", {"id": request_id, "time": _iso_now(), "answer": answer})
            return

        llm_context = _build_tool_call_context(message=req.message, tool_call_full=tool_call_full)

        yield _sse("status", {"id": request_id, "phase": "summary_build_start", "time": _iso_now()})
        _phase_log(request_id, "🧾 產生回覆摘要（完全依據 tool_call，不讓 LLM 自由發揮）…")

        answer = _build_medical_summary(llm_context=llm_context)

        # Stream deterministic answer in chunks (keeps UI behavior consistent).
        chunk_size = 18
        for i in range(0, len(answer), chunk_size):
            yield _sse("delta", {"id": request_id, "text": answer[i : i + chunk_size]})
            await asyncio.sleep(0)

        if answer:
            agent._add_to_conversation("assistant", answer, session=req.session_id)

        _phase_log(
            request_id,
            f"🏁 完成：elapsedMs={(time.perf_counter() - started_at) * 1000.0:.1f}",
        )
        yield _sse("done", {"id": request_id, "time": _iso_now(), "answer": answer})

    if not req.stream:
        intent = await _llm_classify_intent(
            agent=agent,
            message=req.message,
            session_id=req.session_id,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
        )
        if intent != "medical":
            return await _handle_non_medical(intent)

        tool_call = await agent.process_query_only_tool_result(
            req.message,
            session=req.session_id,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
        )
        tool_call_full = tool_call.model_dump(mode="json", by_alias=True)
        llm_context = _build_tool_call_context(message=req.message, tool_call_full=tool_call_full)
        answer = _build_medical_summary(llm_context=llm_context)
        response = AgentResponseWithNaturalLanguage(toolCall=tool_call, answer=answer)
        return JSONResponse(response.model_dump(by_alias=True, mode="json"))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _running_in_docker() -> bool:
    if _truthy(os.getenv("FAST_MCP_IN_DOCKER")):
        return True
    # Common docker marker file (Linux containers).
    try:
        if Path("/.dockerenv").exists():
            return True
    except Exception:
        pass
    # cgroup hints (best-effort).
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists():
            text = cgroup.read_text(encoding="utf-8", errors="ignore")
            if any(k in text for k in ("docker", "containerd", "kubepods")):
                return True
    except Exception:
        pass
    return False


def _container_safe_host(host: str) -> str:
    h = (host or "").strip()
    if not h:
        return "0.0.0.0"
    if h == "localhost" or h.startswith("127."):
        return "0.0.0.0"
    return h


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    host = str(get_cfg_value(cfg, "gateway.host", "0.0.0.0") or "0.0.0.0")
    port = int(get_cfg_value(cfg, "gateway.port", 8081))
    reload = _truthy(get_cfg_value(cfg, "gateway.reload", False))

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    if _running_in_docker():
        safe_host = _container_safe_host(host)
        if safe_host != host:
            logging.getLogger("uvicorn.error").warning(
                "[Docker] gateway.host=%s is not reachable outside container; overriding to %s",
                host,
                safe_host,
            )
            host = safe_host

    uvicorn.run(
        "fast_mcp_client.agent_gateway:app",
        host=host,
        port=port,
        reload=reload,
        ws="none",
    )
