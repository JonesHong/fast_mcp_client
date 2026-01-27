from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .client import SingleMCPClient
from .llm import LLMBackend, LLMRouter, OllamaChatProvider, OpenAIChatProvider


class ConversationMessage(BaseModel):
    """對話訊息"""
    role: str = Field(..., description="角色：user, assistant, system")
    content: str = Field(..., description="訊息內容")
    timestamp: datetime = Field(default_factory=datetime.now, description="時間戳")

class ToolCallStatusEnum(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    MISSING_PARAMETERS = "missing_parameters"
    SERVER_INTERNAL_ERROR = "server_internal_error"

class ToolCallInfo(BaseModel):
    """工具調用資訊"""
    server_name: str = Field(..., description="MCP 伺服器名稱", alias="serverName")
    tool_name: str = Field(..., description="工具名稱", alias="toolName")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="工具參數", alias="parameters")

    model_config = ConfigDict(
        populate_by_name=True
    )

class ToolCallResult(BaseModel):
    """工具調用結果"""
    tools: Optional[List[ToolCallInfo]] = Field(..., description="調用工具資訊列表")
    result: Dict[str, Any] = Field(..., description="調用結果")
    status: ToolCallStatusEnum = Field(..., description="是否成功")
    error_message: Optional[str] = Field(None, description="錯誤訊息")

class AgentResponseWithNaturalLanguage(BaseModel):
    """Agent 結果（含 tool result + 最終自然語言回答）"""
    tool_call: ToolCallResult = Field(..., alias="toolCall")
    answer: str = Field(..., description="LLM 生成的自然語言回答")

    model_config = ConfigDict(
        populate_by_name=True
    )

class BaseNode(BaseModel):
    server_name: Optional[str] = Field(None, description="MCP server name")
    clarification_needed: Optional[str] = Field(None, description="Clarification question or error detail")

class ServerDecisionNode(BaseNode):
    pass

class ToolDecisionNode(BaseNode):
    tool_name: Optional[str] = Field(None, description="MCP tool name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")

class ParameterClarification(BaseModel):
    """參數澄清請求"""
    tool_name: str = Field(..., description="需要澄清參數的工具名稱")
    missing_parameters: List[str] = Field(..., description="缺少的必要參數")
    current_parameters: Dict[str, Any] = Field(..., description="已提取的參數")
    clarification_questions: Dict[str, str] = Field(..., description="參數澄清問題")
    user_language: str = Field("zh", description="用戶語言")

class ConversationContext(BaseModel):
    """對話上下文"""
    pending_tool: Optional[str] = Field(None, description="待執行的工具")
    pending_parameters: Dict[str, Any] = Field(default_factory=dict, description="待補齊的參數")
    awaiting_clarification: bool = Field(False, description="是否等待參數澄清")
    clarification_request: Optional[ParameterClarification] = Field(None, description="參數澄清請求")

class MCPClientAgent:
    def __init__(
        self,
        mcp_config_path: str,
        mcp_config_data: Optional[dict] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        openai_fallback_model: Optional[str] = None,
        llm_primary_provider: Optional[str] = None,
        llm_primary_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_keep_alive: float | str | None = None,
        verbose: bool = False,
        prompt_dir: Optional[str] = None,
    ):
        self.mcp_config_path = mcp_config_path
        self.mcp_config_data = mcp_config_data
        self.openai_model = openai_model
        self.verbose = verbose
        default_prompt_dir = Path(__file__).resolve().parent / "prompts"
        self.prompt_dir = Path(prompt_dir) if prompt_dir else default_prompt_dir
        self._prompt_cache: dict[str, str] = {}
        self.client_list: list[SingleMCPClient] = []
        self.is_initialized = False
        self.conversation_history: Dict[str, List[ConversationMessage]] = {}
        self.conversation_context: Dict[str, ConversationContext] = {}

        # LLM routing:
        # - Primary: env/args selectable (default openai)
        # - Fallback: OpenAI if OPENAI_API_KEY is available
        env_primary = os.getenv("LLM_PRIMARY_PROVIDER")
        primary_provider = (llm_primary_provider or env_primary or "openai").strip().lower()

        # Ollama defaults (native ollama python client)
        ollama_host = (
            ollama_host
            or os.getenv("OLLAMA_HOST")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        # Some setups use an OpenAI-compatible URL like http://localhost:11434/v1;
        # normalize it for the native ollama python client.
        if isinstance(ollama_host, str):
            ollama_host = ollama_host.rstrip("/")
            if ollama_host.endswith("/v1"):
                ollama_host = ollama_host[:-3]
        ollama_model = (
            llm_primary_model if primary_provider == "ollama" and llm_primary_model else os.getenv("OLLAMA_MODEL")
        ) or "qwen2.5:7b-instruct"

        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if openai_api_key is not None:
            if str(openai_api_key).strip() in {"", "your_openai_api_key_here"}:
                openai_api_key = None
        openai_model_env = os.getenv("OPENAI_MODEL")
        openai_fallback_model_env = openai_fallback_model or os.getenv("OPENAI_FALLBACK_MODEL")
        openai_model_effective = (
            llm_primary_model if primary_provider == "openai" and llm_primary_model else openai_model_env
        ) or openai_model
        openai_fallback_model = openai_fallback_model_env or openai_model_env or openai_model

        if primary_provider == "ollama":
            primary = LLMBackend(
                name="ollama",
                provider=OllamaChatProvider(host=ollama_host, keep_alive=ollama_keep_alive or os.getenv("OLLAMA_KEEP_ALIVE")),
                model=ollama_model,
            )
        elif primary_provider == "openai":
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set (required for LLM_PRIMARY_PROVIDER=openai).")
            primary = LLMBackend(
                name="openai",
                provider=OpenAIChatProvider(api_key=openai_api_key),
                model=openai_model_effective,
            )
        else:
            raise ValueError(f"Unsupported LLM_PRIMARY_PROVIDER: {primary_provider}")

        fallback: Optional[LLMBackend] = None
        if openai_api_key:
            fallback = LLMBackend(
                name="openai",
                provider=OpenAIChatProvider(api_key=openai_api_key),
                model=openai_fallback_model,
            )
            # If primary is already openai, keep fallback disabled.
            if primary.name == "openai":
                fallback = None

        self.llm = LLMRouter(primary=primary, fallback=fallback)
        

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[MCPClientAgent] {message}")

    async def initialize(self) -> None:
        if self.is_initialized:
            return
        if self.mcp_config_data is not None:
            config_data = self.mcp_config_data
        else:
            with open(self.mcp_config_path, "r") as f:
                config_data = json.load(f)
        for server_name, config in config_data.get("mcpServers", {}).items():
            type = config.get("type", "stdio")
            if type != "streamable_http" and type != "streamable-http":
                self._log(f"Only 'streamable_http' type is supported. Skipping server '{server_name}' of type '{type}'.")
                continue
            url = config.get("url")
            authorization = config.get("headers", {}).get("Authorization") or config.get("headers", {}).get("authorization")
            if not url:
                self._log(f"No URL found for server '{server_name}'. Skipping.")
                continue
            self._log(f"Loading MCP client for server '{server_name}' at URL '{url}'.")
            try:
                client = await SingleMCPClient.create(
                    base_url=url,
                    authorization=authorization,
                    verbose=self.verbose,
                )
                self.client_list.append(client)
            except Exception as e:
                self._log(f"Failed to initialize client for server '{server_name}': {e}")
        self.is_initialized = True
    
    def get_all_clients(self) -> list[SingleMCPClient]:
        return self.client_list

    def _add_to_conversation(self, role: str, content: str, session: str = "default") -> None:
        message = ConversationMessage(role=role, content=content)
        if session not in self.conversation_history:
            self.conversation_history[session] = []
        self.conversation_history[session].append(message)
        if len(self.conversation_history[session]) > 10:
            self.conversation_history[session] = self.conversation_history[session][-10:]

    def _truncate_text(self, text: str, limit: int = 500) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...(truncated)"

    def _tool_call_summary_for_llm(self, tool_call: ToolCallResult) -> dict:
        payload = tool_call.model_dump(mode="json", by_alias=True)
        result = payload.get("result") or {}

        # Compact large report lists to avoid huge prompts.
        if isinstance(result, dict) and isinstance(result.get("reports"), list):
            reports = result.get("reports") or []
            sample = []
            for r in reports[:5]:
                if isinstance(r, dict):
                    sample.append({
                        k: r.get(k)
                        for k in ["id", "caseCode", "operationStartTime", "status", "file_path"]
                        if k in r
                    })
                else:
                    sample.append(r)
            result = dict(result)
            result.pop("reports", None)
            result["reportsCount"] = len(reports)
            result["reportsSample"] = sample
            payload["result"] = result

        # Compact byRange similarly.
        if isinstance(result, dict) and isinstance(result.get("byRange"), list):
            by_range = []
            for bucket in (result.get("byRange") or [])[:5]:
                if isinstance(bucket, dict):
                    bucket_reports = bucket.get("reports") or []
                    by_range.append({
                        "range": bucket.get("range"),
                        "reportsCount": len(bucket_reports) if isinstance(bucket_reports, list) else None,
                        "reportsSample": [
                            {
                                k: r.get(k)
                                for k in ["id", "caseCode", "operationStartTime", "status", "file_path"]
                                if k in r
                            }
                            for r in (bucket_reports[:3] if isinstance(bucket_reports, list) else [])
                            if isinstance(r, dict)
                        ],
                    })
                else:
                    by_range.append(bucket)
            result = dict(payload.get("result") or {})
            result["byRange"] = by_range
            payload["result"] = result

        return payload

    def _parse_json_to_model(self, model_cls: type[BaseNode], raw_text: str) -> BaseNode:
        try:
            return model_cls.model_validate_json(raw_text)
        except Exception:
            extracted = self._extract_first_json_object(raw_text)
            if extracted:
                try:
                    return model_cls.model_validate_json(extracted)
                except Exception:
                    pass

            # Fallback: look for fenced ```json blocks
            import re
            fence = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
            if fence:
                try:
                    return model_cls.model_validate_json(fence.group(1))
                except Exception:
                    pass

        return model_cls(clarification_needed=f"JSON parse failed; raw={self._truncate_text(raw_text, limit=800)}")

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        """
        Extract the first balanced JSON object substring from an arbitrary LLM output.
        This is more robust than greedy regex for models that append extra tokens/blocks.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None

    def _load_prompt(self, filename: str) -> str:
        if filename in self._prompt_cache:
            return self._prompt_cache[filename]
        prompt_path = self.prompt_dir / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompt_text = prompt_path.read_text(encoding="utf-8")
        self._prompt_cache[filename] = prompt_text
        return prompt_text

    def reload_prompts(self) -> None:
        self._prompt_cache.clear()

    def _get_time_context(self, current_time: Optional[datetime] = None) -> str:
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        current_time_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        current_date_str = current_time.strftime("%Y-%m-%d")
        current_year = current_time.year
        current_month = current_time.month

        start_of_today = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_week = start_of_today - timedelta(days=current_time.weekday())
        start_of_month = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_of_year = current_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

        return f"""
CURRENT TIME CONTEXT (Use this for time parameter extraction):
- Current Time (UTC): {current_time_str}
- Current Date: {current_date_str}
- Current Year: {current_year}
- Current Month: {current_month}
- Start of Today: {start_of_today.strftime("%Y-%m-%dT%H:%M:%SZ")}
- Start of This Week: {start_of_week.strftime("%Y-%m-%dT%H:%M:%SZ")}
- Start of This Month: {start_of_month.strftime("%Y-%m-%dT%H:%M:%SZ")}
- Start of This Year: {start_of_year.strftime("%Y-%m-%dT%H:%M:%SZ")}

TIME EXTRACTION GUIDELINES:
- "今天" / "today" → startTime: {start_of_today.strftime("%Y-%m-%dT%H:%M:%SZ")}, endTime: {current_time_str}
- "本週" / "this week" → startTime: {start_of_week.strftime("%Y-%m-%dT%H:%M:%SZ")}, endTime: {current_time_str}
- "這個月" / "this month" → startTime: {start_of_month.strftime("%Y-%m-%dT%H:%M:%SZ")}, endTime: {current_time_str}
- "今年" / "this year" → startTime: {start_of_year.strftime("%Y-%m-%dT%H:%M:%SZ")}, endTime: {current_time_str}
- "最近一週" / "past week" → startTime: {(current_time - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")}, endTime: {current_time_str}
- "最近一個月" / "past month" → startTime: {(current_time - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")}, endTime: {current_time_str}
- Always use ISO format: YYYY-MM-DDTHH:MM:SSZ
"""

    def _get_system_prompt(self, server_name: Optional[str] = None) -> str:
        tool_entries: list[str] = []
        for client in self.client_list:
            if not client.tools:
                continue
            client_server_name = client.server_info.name if client.server_info else "unknown"
            if server_name and client_server_name != server_name:
                continue
            for tool in client.tools:
                required = [
                    name for name, prop in tool.input_properties.items() if prop.required
                ]
                param_lines = []
                for name, prop in tool.input_properties.items():
                    requirement = "REQUIRED" if prop.required else "OPTIONAL"
                    enum_text = f", enum={prop.enum}" if prop.enum else ""
                    param_lines.append(
                        f"  - {name}: {prop.type} ({requirement}){enum_text} - {prop.description}"
                    )
                param_text = "\n".join(param_lines) if param_lines else "  - No parameters"
                tool_entries.append(
                    f"""
Server: {client_server_name}
Tool Name: {tool.name}
Description: {tool.description}
Required Parameters: {", ".join(required) if required else "None"}
Parameters:
{param_text}
"""
                )

        template = self._load_prompt("tool_selector.prompt")
        return template.format(
            time_context=self._get_time_context(),
            tool_entries="".join(tool_entries),
        )

    def _get_server_router_prompt(self) -> str:
        server_entries: list[str] = []
        for client in self.client_list:
            if not client.server_info:
                continue
            tool_names = [tool.name for tool in client.tools or []]
            tool_summary = ", ".join(tool_names) if tool_names else "No tools"
            server_entries.append(
                f"""
Server Name: {client.server_info.name}
Instructions: {client.server_info.instructions}
Tools: {tool_summary}
"""
            )

        template = self._load_prompt("server_router.prompt")
        return template.format(server_entries="".join(server_entries))

    def _find_client_for_tool(self, server_name: str, tool_name: str) -> Optional[SingleMCPClient]:
        for client in self.client_list:
            if not client.server_info:
                continue
            if client.server_info.name != server_name:
                continue
            if client.tools and any(tool.name == tool_name for tool in client.tools):
                return client
        return None

    def _find_client_by_server_name(self, server_name: str) -> Optional[SingleMCPClient]:
        for client in self.client_list:
            if not client.server_info:
                continue
            if client.server_info.name == server_name:
                return client
        return None

    async def _decide_tool_and_extract_params_with_server(
        self,
        user_query: str,
        server_name: str,
        session: str = "default",
        *,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> ToolDecisionNode:
        messages = [{"role": "system", "content": self._get_system_prompt(server_name=server_name)}]
        for msg in self.conversation_history.get(session, [])[-6:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_query})

        try:
            response_content = await self.llm.chat_complete_text(
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                provider=llm_provider,
                model=llm_model,
            )
            decision = self._parse_json_to_model(ToolDecisionNode, response_content)
            # This function is already scoped to a specific MCP server, so do not allow
            # the model to override the server selection.
            decision.server_name = server_name

            if (
                decision.clarification_needed
                and str(decision.clarification_needed).startswith("JSON parse failed")
                and self.llm.fallback is not None
                and (llm_provider or self.llm.primary.name) != "openai"
            ):
                response_content = await self.llm.chat_complete_text(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800,
                    provider="openai",
                    model=None,
                )
                decision = self._parse_json_to_model(ToolDecisionNode, response_content)
                decision.server_name = server_name

            inferred = self._infer_tool_from_query(user_query)
            if inferred:
                inferred_tool, inferred_params = inferred
                tool_name_raw = (decision.tool_name or "").strip()
                q = user_query.lower()
                is_tools_intent = any(k in q for k in ["list tools", "tools", "tool", "工具", "有哪些工具", "列出工具"])
                if (not tool_name_raw) or (tool_name_raw == "__list_tools__" and not is_tools_intent):
                    decision.tool_name = inferred_tool
                    decision.parameters = inferred_params
                    decision.clarification_needed = None

            # If the LLM responded with a clarification for a "list tools" request,
            # fall back to the internal "__list_tools__" tool while still going through the LLM step.
            tool_name_raw = (decision.tool_name or "").strip()
            if tool_name_raw.lower() in {"__list_tools__", "list_tools", "list-tools", "listtools"}:
                decision.tool_name = "__list_tools__"
                decision.parameters = {}
                decision.clarification_needed = None
                return decision

            if not tool_name_raw:
                q = user_query.lower()
                if any(k in q for k in ["list tools", "tools", "tool", "工具", "有哪些工具", "列出"]):
                    decision.tool_name = "__list_tools__"
                    decision.parameters = {}
                    decision.clarification_needed = None
                    return decision

            # Post-normalize common domain intent: CASE-xxx is almost always a caseCode, not a SurgicalReport id.
            # This prevents the LLM from mistakenly calling findOne(id="CASE-202510") when the user wants a list.
            import re

            q = user_query
            q_lower = q.lower()
            m_case = re.search(r"\bCASE[-_A-Za-z0-9]+\b", q)
            case_code = m_case.group(0) if m_case else None

            wants_case_list = case_code and any(
                k in q_lower
                for k in [
                    "案號",
                    "病例代號",
                    "病歷代號",
                    "casecode",
                    "case code",
                    "所有",
                    "全部",
                    "列表",
                    "清單",
                    "all",
                ]
            )
            mentions_id = any(k in q_lower for k in ["手術單號", "單號", "report id", "id:"])
            id_param = str(decision.parameters.get("id") or "")

            if decision.tool_name == "surgicalReport.findOne" and case_code and not mentions_id:
                # If the provided id looks like a case code, treat it as caseCode query.
                if wants_case_list or id_param == case_code:
                    if self._find_client_for_tool(server_name, "surgicalReport.findByCaseCode") is not None:
                        decision.tool_name = "surgicalReport.findByCaseCode"
                        decision.parameters = {"caseCode": case_code}
                        decision.clarification_needed = None
                        tool_name_raw = decision.tool_name

            tool_name = tool_name_raw
            if tool_name == "__list_tools__":
                return decision

            client = self._find_client_for_tool(server_name, tool_name or "")
            if tool_name and client is None:
                decision.tool_name = None
                decision.clarification_needed = (
                    f"Unknown tool '{tool_name}' for server '{server_name}'."
                )
            return decision
        except Exception as exc:
            inferred = self._infer_tool_from_query(user_query)
            if inferred:
                tool_name, params = inferred
                return ToolDecisionNode(
                    server_name=server_name,
                    tool_name=tool_name,
                    parameters=params,
                )
            return ToolDecisionNode(clarification_needed=f"LLM request failed: {exc}")

    async def _decide_server(
        self,
        user_query: str,
        session: str = "default",
        *,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> ServerDecisionNode:
        messages = [{"role": "system", "content": self._get_server_router_prompt()}]
        for msg in self.conversation_history.get(session, [])[-4:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_query})

        try:
            response_content = await self.llm.chat_complete_text(
                messages=messages,
                temperature=0.0,
                max_tokens=200,
                provider=llm_provider,
                model=llm_model,
            )
            decision = self._parse_json_to_model(ServerDecisionNode, response_content)
            if (
                decision.clarification_needed
                and str(decision.clarification_needed).startswith("JSON parse failed")
                and self.llm.fallback is not None
                and (llm_provider or self.llm.primary.name) != "openai"
            ):
                response_content = await self.llm.chat_complete_text(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=200,
                    provider="openai",
                    model=None,
                )
            decision = self._parse_json_to_model(ServerDecisionNode, response_content)

            if not decision.server_name:
                # If only one server is available, default to it (robust against weak JSON outputs).
                available = [c.server_info.name for c in self.client_list if c.server_info]
                if len(available) == 1:
                    decision.server_name = available[0]
                    decision.clarification_needed = None

            return decision
        except Exception as exc:
            # If only one server is available, keep working even if LLM fails.
            available = [c.server_info.name for c in self.client_list if c.server_info]
            if len(available) == 1:
                return ServerDecisionNode(server_name=available[0])
            return ServerDecisionNode(clarification_needed=f"LLM request failed: {exc}")

    def _infer_tool_from_query(self, user_query: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """
        Minimal heuristic fallback when the model doesn't return usable JSON.
        Keeps the system usable with weaker local models.
        """
        import re

        q = user_query
        q_lower = q.lower()

        # list tools intent
        if any(k in q_lower for k in ["list tools", "tools", "tool", "工具", "有哪些工具", "列出工具"]):
            return "__list_tools__", {}

        # find one by UUID
        m_id = re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", q_lower)
        if m_id:
            return "surgicalReport.findOne", {"id": m_id.group(0)}

        # case code
        m_case = re.search(r"\bCASE[-_A-Za-z0-9]+\b", q)
        case_code = m_case.group(0) if m_case else None

        # date range like 2025-10-01 到 2025-10-31
        dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", q)
        if len(dates) >= 2:
            start_date, end_date = dates[0], dates[1]
            params: Dict[str, Any] = {
                "start": f"{start_date}T00:00:00Z",
                "end": f"{end_date}T23:59:59Z",
            }
            if case_code:
                params["caseCode"] = case_code
            return "surgicalReport.findByOperationStartTimeRange", params

        # month like 2025/10
        m_month = re.search(r"\b(20\d{2})[/-](\d{1,2})\b", q)
        if m_month:
            year = int(m_month.group(1))
            month = int(m_month.group(2))
            if 1 <= month <= 12:
                from datetime import datetime, timezone

                start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

                params = {
                    "start": start.isoformat().replace("+00:00", "Z"),
                    "end": end.isoformat().replace("+00:00", "Z"),
                }
                if case_code:
                    params["caseCode"] = case_code
                return "surgicalReport.findByOperationStartTimeRange", params

        if case_code and any(
            k in q_lower
            for k in [
                "查詢",
                "搜尋",
                "列表",
                "reports",
                "report",
                "手術報告",
                "報告",
                "全部",
                "所有",
                "案號",
                "病例代號",
                "病歷代號",
                "casecode",
                "case code",
            ]
        ):
            return "surgicalReport.findByCaseCode", {"caseCode": case_code}

        return None

    async def _execute_tool(
        self,
        server_name: str,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> ToolCallResult:
        print("Executing tool:", server_name, tool_name, parameters)

        if tool_name == "__list_tools__":
            client = self._find_client_by_server_name(server_name)
            if client is None:
                return ToolCallResult(
                    tools=None,
                    result={},
                    status=ToolCallStatusEnum.FAILURE,
                    error_message="Unknown server",
                )
            return ToolCallResult(
                tools=[],
                result={
                    "server": {
                        "name": client.server_info.name if client.server_info else server_name,
                        "version": client.server_info.version if client.server_info else "",
                        "instructions": client.server_info.instructions if client.server_info else None,
                    },
                    "tools": [t.model_dump(by_alias=True) for t in (client.tools or [])],
                },
                status=ToolCallStatusEnum.SUCCESS,
            )

        client = self._find_client_for_tool(server_name, tool_name)
        if client is None:
            return ToolCallResult(
                tools=[ToolCallInfo(serverName=server_name, toolName=tool_name, parameters=parameters)],
                result={},
                status=ToolCallStatusEnum.FAILURE,
                error_message="Unknown server or tool",
            )
        try:
            validation_results = client._params_valid(tool_name, parameters)
            invalid_fields = [
                field for field, result in validation_results.items() if not result.is_valid
            ]
            if invalid_fields:
                missing_fields = [
                    field for field in invalid_fields
                    if validation_results[field].error_message and "Field required" in validation_results[field].error_message
                ]
                status = ToolCallStatusEnum.MISSING_PARAMETERS if missing_fields else ToolCallStatusEnum.FAILURE
                return ToolCallResult(
                    tools=[ToolCallInfo(serverName=server_name, toolName=tool_name, parameters=parameters)],
                    result={"validation": {k: v.model_dump(by_alias=True) for k, v in validation_results.items()}},
                    status=status,
                    error_message="Parameter validation failed",
                )
            result = await client.call_tool(tool_name, parameters)
            return ToolCallResult(
                tools=[ToolCallInfo(serverName=server_name, toolName=tool_name, parameters=parameters)],
                result=result,
                status=ToolCallStatusEnum.SUCCESS,
            )
        except Exception as exc:
            return ToolCallResult(
                tools=[ToolCallInfo(serverName=server_name, toolName=tool_name, parameters=parameters)],
                result={},
                status=ToolCallStatusEnum.FAILURE,
                error_message=str(exc),
            )

    async def process_query_only_tool_result(
        self,
        user_query: str,
        session: str = "default",
        *,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> ToolCallResult:
        self._add_to_conversation("user", user_query, session=session)
        if session not in self.conversation_context:
            self.conversation_context[session] = ConversationContext()

        routing = await self._decide_server(
            user_query,
            session=session,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        routed_server = routing.server_name
        if not routed_server:
            clarification = routing.clarification_needed or "無法理解請求"
            return ToolCallResult(
                tools=None,
                result={},
                status=ToolCallStatusEnum.FAILURE,
                error_message=clarification,
            )

        decision = await self._decide_tool_and_extract_params_with_server(
            user_query,
            routed_server,
            session=session,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        if decision.tool_name is None or decision.server_name is None:
            clarification = decision.clarification_needed or "無法理解請求"
            return ToolCallResult(
                tools=None,
                result={},
                status=ToolCallStatusEnum.FAILURE,
                error_message=clarification,
            )

        return await self._execute_tool(
            decision.server_name,
            decision.tool_name,
            decision.parameters,
        )

    async def process_query_with_natural_response(
        self,
        user_query: str,
        session: str = "default",
        *,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> AgentResponseWithNaturalLanguage:
        """
        Like process_query_only_tool_result(), but also asks the LLM to generate a
        human-readable final answer based on the tool result.
        """
        tool_call = await self.process_query_only_tool_result(
            user_query,
            session=session,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

        summary = self._tool_call_summary_for_llm(tool_call)
        tool_json = self._truncate_text(json.dumps(summary, ensure_ascii=False, indent=2), limit=8000)

        system_prompt = (
            "You are a helpful assistant.\n"
            "You MUST answer in Traditional Chinese (繁體中文) unless the user clearly uses another language.\n"
            "Given the user's query and the MCP tool call result JSON, produce a concise and useful answer.\n"
            "If the tool result contains many items, summarize counts and show a small sample.\n"
            "If the tool call failed, explain the error and suggest the next action.\n"
        )

        try:
            answer = await self.llm.chat_complete_text(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                    {"role": "system", "content": f"ToolCallResult (JSON):\n{tool_json}"},
                ],
                temperature=0.2,
                max_tokens=800,
                provider=llm_provider,
                model=llm_model,
            )
        except Exception as exc:
            answer = f"（LLM 回覆失敗）{exc}"

        self._add_to_conversation("assistant", answer, session=session)
        return AgentResponseWithNaturalLanguage(toolCall=tool_call, answer=answer)
    
