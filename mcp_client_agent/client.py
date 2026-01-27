import asyncio
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter, ValidationError
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.json_schema_type import json_schema_to_type
from mcp.types import Tool

class MCPServerInfo(BaseModel):
    name: str
    version: str
    instructions: Optional[str] = None

class PropertyItem(BaseModel):
    name: str
    type: str
    enum: Optional[list[str]] = None
    description: str
    required: Optional[bool] = None
    optional: Optional[bool] = None

class ToolInfo(BaseModel):
    name: str = Field(...)
    description: str = Field(...)
    input_properties: dict[str, PropertyItem] = Field(..., alias="inputProperties")

    model_config = ConfigDict(
        populate_by_name=True
    )

class ParamValidationInfo(BaseModel):
    is_valid: bool
    error_message: Optional[str] = Field(None, alias="errorMessage")

    model_config = ConfigDict(
        populate_by_name=True
    )

class ParamValidationError(Exception):
    pass

class ToolNotFoundError(Exception):
    pass

class ToolInfoFactory:
    @staticmethod
    def from_tools(tools: list[Tool]) -> list[ToolInfo]:
        tool_infos = []
        for tool in tools:
            properties = []
            # check required and optional
            required_set = set(tool.inputSchema.get("required", []))
            # all properties in input_properties
            for prop_name, prop_info in tool.inputSchema.get("properties", {}).items():
                properties.append(
                    PropertyItem(
                        name=prop_name,
                        type=prop_info.get("type", "unknown"),
                        description=prop_info.get("description", ""),
                        enum=prop_info.get("enum"),
                        required=True if prop_name in required_set else None,
                        optional=True if prop_name not in required_set else None
                    )
                )
            tool_infos.append(
                ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    input_properties={p.name: p for p in properties}
                )
            )
        return tool_infos

class SingleMCPClient:

    @classmethod
    async def create(
        cls,
        base_url: str,
        authorization: str | None = None,
        *,
        timeout: float | None = None,
        init_timeout: float | None = None,
        retries: int = 0,
        retry_backoff: float = 0.25,
        verbose: bool = False,
    ) -> "SingleMCPClient":
        self = cls(
            base_url,
            authorization,
            timeout=timeout,
            init_timeout=init_timeout,
            retries=retries,
            retry_backoff=retry_backoff,
            verbose=verbose,
        )
        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        transport = StreamableHttpTransport(
            url=base_url,
            headers=headers
        )
        self._client = Client(
            transport=transport,
            timeout=self.timeout,
            init_timeout=self.init_timeout,
        )
        self._log("Initializing MCP client...")
        try:
            async with self._client as client:
                info = client.initialize_result
                self.server_info = MCPServerInfo(
                    name=info.serverInfo.name,
                    version=info.serverInfo.version,
                    instructions=info.instructions,
                )
                self._log(
                    f"Connected to MCP server: {self.server_info.name} (version {self.server_info.version})"
                )
                self._log(f"Instructions: {self.server_info.instructions}")
                tools = await client.list_tools()
                tool_infos = ToolInfoFactory.from_tools(tools)
                self.tools = tool_infos
                self._tool_validators = self._build_tool_validators(tools)
            self._is_initialized = True
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MCP client: {e}") from e


    def __init__(
        self,
        base_url: str,
        authorization: str | None = None,
        *,
        timeout: float | None = None,
        init_timeout: float | None = None,
        retries: int = 0,
        retry_backoff: float = 0.25,
        verbose: bool = False,
    ) -> None:
        self.base_url = base_url
        self.authorization = authorization
        self.timeout = timeout
        self.init_timeout = init_timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.verbose = verbose
        self._is_initialized = False
        self._client: Client | None = None
        self.server_info: MCPServerInfo | None = None
        self.tools: list[ToolInfo] | None = None
        self._tool_validators: dict[str, TypeAdapter] = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _build_tool_validators(self, tools: list[Tool]) -> dict[str, TypeAdapter]:
        validators: dict[str, TypeAdapter] = {}
        for tool in tools:
            schema = tool.inputSchema or {}
            if schema.get("type") == "object":
                param_type = json_schema_to_type(schema, name=f"{tool.name}Input")
            else:
                param_type = json_schema_to_type(schema)
            validators[tool.name] = TypeAdapter(param_type)
        return validators
    
    def _params_valid(self, tool_name: str, params: dict) -> dict[str, ParamValidationInfo]:
        if tool_name not in [tool.name for tool in self.tools or []]:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found.")
        tool_info = next(tool for tool in self.tools or [] if tool.name == tool_name)
        validator = self._tool_validators.get(tool_name)
        if validator is None:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found.")
        per_field_results: dict[str, ParamValidationInfo] = {
            prop_name: ParamValidationInfo(is_valid=True)
            for prop_name in tool_info.input_properties.keys()
        }
        try:
            validator.validate_python(params)
        except ValidationError as exc:
            for error in exc.errors():
                loc = error.get("loc", [])
                field = str(loc[0]) if loc else "input"
                per_field_results[field] = ParamValidationInfo(
                    is_valid=False,
                    error_message=error.get("msg", "Invalid parameter."),
                )
        for param_name in params.keys():
            if param_name not in per_field_results:
                per_field_results[param_name] = ParamValidationInfo(
                    is_valid=False,
                    error_message=f"Unknown parameter: '{param_name}'",
                )
        return per_field_results

    async def call_tool(self, tool_name: str, params: dict) -> dict:
        if not self._is_initialized or self._client is None:
            raise RuntimeError("Client is not initialized.")
        # check if tool exists
        if self.tools is None or not any(tool.name == tool_name for tool in self.tools):
            raise ToolNotFoundError(f"Tool '{tool_name}' not found.")
        # validate parameters
        validation_results = self._params_valid(tool_name, params)
        if any(not result.is_valid for result in validation_results.values()):
            error_json = [ f'{{"field": "{field}", "error": "{result.error_message}"}}' for field, result in validation_results.items() if not result.is_valid]
            raise ParamValidationError(f"Parameter validation failed: {', '.join(error_json)}")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                async with self._client as client:
                    if self.timeout is None:
                        result = await client.call_tool(tool_name, params)
                    else:
                        result = await asyncio.wait_for(
                            client.call_tool(tool_name, params),
                            timeout=self.timeout,
                        )
                    return self._normalize_tool_result(result)
            except (ParamValidationError, ToolNotFoundError):
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(self.retry_backoff * (2 ** attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Tool call failed unexpectedly.")

    def _normalize_tool_result(self, result) -> dict:
        return result.structured_content or {}
