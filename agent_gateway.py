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

import ollama
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

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

    # Domain keywords
    if any(k in q_lower for k in ["手術報告", "手術單號", "病例代號", "病歷代號", "案號", "operationstarttime"]):
        return True

    return False


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
        content = await agent.llm.chat_complete_text(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
            temperature=0.0,
            max_tokens=80,
            provider=llm_provider,
            model=llm_model,
        )
        raw = _extract_first_json_object(content) or content.strip()
        obj = json.loads(raw)
        intent = str(obj.get("intent", "")).strip().lower()
        if intent in {"medical", "intro", "other"}:
            return intent
    except Exception:
        pass

    if heuristic_intro:
        return "intro"
    if heuristic_medical:
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

    session_id: Annotated[
        str,
        Field(validation_alias="sessionId", serialization_alias="sessionId"),
    ] = "default"
    message: str = Field(..., description="使用者訊息（建議包含 UUID / CASE-xxxx / 時間區間）")
    stream: bool = Field(True, description="true: SSE 串流；false: 一次回傳 JSON")
    llm_provider: Annotated[
        Optional[str],
        Field(validation_alias="llmProvider", serialization_alias="llmProvider"),
    ] = None
    llm_model: Annotated[
        Optional[str],
        Field(validation_alias="llmModel", serialization_alias="llmModel"),
    ] = None
    tool_result: Annotated[
        Literal["none", "summary", "full"],
        Field(validation_alias="toolResult", serialization_alias="toolResult"),
    ] = Field("summary", description="SSE 的 tool_call 事件輸出：none/summary/full")


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

    while True:
        if stop.is_set():
            return

        try:
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
                print(f"[OllamaWarmup] ok model={model} keep_alive={keep_alive} time={_iso_now()}")
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
    warmup_enabled = bool(warmup_cfg.get("enabled", False))
    app.state.ollama_warmup_enabled = warmup_enabled

    warmup_stop = asyncio.Event()
    warmup_task: Optional[asyncio.Task] = None
    if warmup_enabled:
        host = str(get_cfg_value(cfg, "llm.ollama_host", "http://localhost:11434"))
        model = str(get_cfg_value(cfg, "llm.primary_model", "") or "")
        keep_alive = get_cfg_value(cfg, "llm.ollama_keep_alive")
        interval_seconds = float(warmup_cfg.get("interval_seconds", 240))
        prompt = str(warmup_cfg.get("prompt", "ping"))
        num_predict = int(warmup_cfg.get("num_predict", 1))
        verbose = bool(get_cfg_value(cfg, "logging.verbose", False))

        if model:
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
        "前端建議：先串流顯示 `delta` 拼成 answer，收到 `done` 後再把 `tool_call` JSON 顯示在下方。"
    ),
    responses={
        200: {
            "description": "SSE events（text/event-stream）",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "examples": {
                        "sample": {
                            "summary": "SSE sample",
                            "value": (
                                "event: status\n"
                                "data: {\"id\":\"abc12345\",\"phase\":\"api_received\",\"time\":\"2026-01-27T00:00:00Z\"}\n\n"
                                "event: status\n"
                                "data: {\"id\":\"abc12345\",\"phase\":\"tool_call_start\",\"time\":\"2026-01-27T00:00:00Z\"}\n\n"
                                "event: tool_call\n"
                                "data: {\"id\":\"abc12345\",\"time\":\"2026-01-27T00:00:00Z\",\"mode\":\"full\",\"toolCall\":{...}}\n\n"
                                "event: delta\n"
                                "data: {\"id\":\"abc12345\",\"text\":\"在 2025 年 10 月共有 16 筆手術報告...\"}\n\n"
                                "event: done\n"
                                "data: {\"id\":\"abc12345\",\"time\":\"2026-01-27T00:00:01Z\",\"answer\":\"...\"}\n\n"
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

        def _count_by(key: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for r in reports:
                v = r.get(key)
                if v is None:
                    continue
                s = str(v)
                counts[s] = counts.get(s, 0) + 1
            return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

        op_start_times = [r.get("operationStartTime") for r in reports if isinstance(r.get("operationStartTime"), str)]
        op_start_min = min(op_start_times) if op_start_times else None
        op_start_max = max(op_start_times) if op_start_times else None

        llm_context = {
            "tool": tool_name,
            "toolParams": tool_params,
            "status": tool_call_full.get("status"),
            "errorMessage": tool_call_full.get("errorMessage"),
            "reportsCount": reports_count,
            "operationStartTimeMin": op_start_min,
            "operationStartTimeMax": op_start_max,
            "findOne": None,
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

        llm_context_json = agent._truncate_text(
            json.dumps(llm_context, ensure_ascii=False, indent=2),
            limit=8000,
        )
        system_prompt = (
            "You are a helpful assistant.\n"
            "You MUST answer in Traditional Chinese (繁體中文) unless the user clearly uses another language.\n"
            "Given the user's query and the MCP tool call context JSON, produce a concise and useful answer.\n"
            "IMPORTANT: The frontend already receives full tool results via the `tool_call` SSE event.\n"
            "Therefore, your answer MUST be summary/overview only: totals, statistics, high-level findings.\n"
            "Terminology:\n"
            "- 病例代號/案號 = caseCode.\n"
            "- When counting reports, the unit MUST be 「筆」 (e.g., 「16 筆手術報告」). Do NOT use 份/張/倝.\n"
            "DO NOT include any examples / samples (e.g., '部分樣本').\n"
            "DO NOT list individual reports, IDs, file paths, or per-item timestamps.\n"
            "If tool is surgicalReport.findOne and context.findOne.matched is true, clearly say the requested report was found.\n"
            "You MUST ONLY use information present in ToolCallContext JSON.\n"
            "If the user asks something outside the available MCP tools, politely guide them back to supported inputs: id(UUID), caseCode(CASE-xxxx), time range.\n"
            "If the user asks for full details, politely tell them to inspect the tool_call JSON.\n"
            "If the tool call failed, explain the error and suggest the next action.\n"
            "For streaming: output plain text only (no JSON), and do not repeat the tool JSON.\n"
        )

        yield _sse("status", {"id": request_id, "phase": "llm_stream_start", "time": _iso_now()})
        _phase_log(request_id, "🧠 開始串流 LLM 回覆…")

        answer_parts: list[str] = []
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message},
                {"role": "system", "content": f"ToolCallContext (JSON):\n{llm_context_json}"},
            ]
            async for delta in agent.llm.chat_stream_deltas(
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                provider=req.llm_provider,
                model=req.llm_model,
            ):
                answer_parts.append(delta)
                yield _sse("delta", {"id": request_id, "text": delta})
        except Exception as exc:
            logger.exception("[%s] llm_stream_failed", request_id)
            yield _sse("error", {"id": request_id, "message": str(exc)})

        answer = "".join(answer_parts).strip()
        if answer:
            # Guard against rare incorrect unit characters produced by some LLMs.
            answer = answer.replace("倝", "筆")
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

        response = await agent.process_query_with_natural_response(
            req.message,
            session=req.session_id,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
        )
        return JSONResponse(response.model_dump(by_alias=True, mode="json"))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    host = get_cfg_value(cfg, "gateway.host", "0.0.0.0")
    port = int(get_cfg_value(cfg, "gateway.port", 8081))
    reload = _truthy(get_cfg_value(cfg, "gateway.reload", False))

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    uvicorn.run(
        "fast_mcp_client.agent_gateway:app",
        host=host,
        port=port,
        reload=reload,
        ws="none",
    )
