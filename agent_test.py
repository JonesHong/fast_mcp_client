import asyncio
import sys
import os
from pathlib import Path
import argparse

# Allow running as a script: `python fast_mcp_client/agent_test.py`
# by ensuring repo root is on sys.path (so `import fast_mcp_client` works).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fast_mcp_client.mcp_client_agent.agent import MCPClientAgent
from fast_mcp_client.config import get_cfg_value, load_config


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fast MCP Client Agent test runner")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw ToolCallResult objects (JSON-like) instead of natural language replies.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path (default: fast_mcp_client/config.yaml or env FAST_MCP_CONFIG)",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="Primary LLM provider: ollama | openai (default: yaml llm.primary_provider)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Primary LLM model (ollama: model tag, openai: model name).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL") or get_cfg_value(cfg, "openai.model") or "gpt-4.1"
    openai_fallback_model = os.getenv("OPENAI_FALLBACK_MODEL") or get_cfg_value(cfg, "openai.fallback_model")

    llm_provider = args.llm_provider or os.getenv("LLM_PRIMARY_PROVIDER") or get_cfg_value(cfg, "llm.primary_provider") or "openai"
    llm_model = args.llm_model or os.getenv("LLM_PRIMARY_MODEL") or get_cfg_value(cfg, "llm.primary_model")
    if not llm_model:
        if str(llm_provider).strip().lower() == "ollama":
            llm_model = os.getenv("OLLAMA_MODEL") or "qwen2.5:7b-instruct"
        else:
            llm_model = "gpt-4.1"

    # MCP servers are configured in YAML only.
    mcp_servers = get_cfg_value(cfg, "mcp.servers")
    mcp_config_data = None
    if isinstance(mcp_servers, dict) and mcp_servers:
        mcp_config_data = {"mcpServers": mcp_servers}
    else:
        raise ValueError('Missing "mcp.servers" in YAML config.')

    mcp_config_path = str((repo_root / "fast_mcp_client" / "config.yaml").resolve())

    agent = MCPClientAgent(
        mcp_config_path=mcp_config_path, 
        mcp_config_data=mcp_config_data,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_fallback_model=openai_fallback_model,
        llm_primary_provider=llm_provider,
        llm_primary_model=llm_model,
        ollama_host=get_cfg_value(cfg, "llm.ollama_host"),
        ollama_keep_alive=get_cfg_value(cfg, "llm.ollama_keep_alive"),
        verbose=bool(get_cfg_value(cfg, "logging.verbose", True)),
    )
    await agent.initialize()
    print("MCP Client Agent initialized with clients:")
    for client in agent.client_list:
        print(f"- Connected to MCP server at {client.base_url} with {len(client.tools or [])} tools.")

    print("-----")
    query = "請列出目前可用的 MCP tools。"
    print(f"User 1: {query}")
    if args.raw:
        response = await agent.process_query_only_tool_result(
            query,
            session="user123",
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        print("Agent 1 (raw):")
        print(response)
    else:
        response = await agent.process_query_with_natural_response(
            query,
            session="user123",
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        print("Assistant 1:")
        print(response.answer)

    print("-----")
    query2 = "請查詢 2025/10 的手術報告（operationStartTime 在 2025-10-01 到 2025-10-31）。"
    print(f"User 2: {query2}")
    if args.raw:
        response2 = await agent.process_query_only_tool_result(
            query2,
            session="user123",
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        print("Agent 2 (raw):")
        print(response2)
    else:
        response2 = await agent.process_query_with_natural_response(
            query2,
            session="user123",
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        print("Assistant 2:")
        print(response2.answer)

    print("-----")
    query3 = "請查詢病例代號 CASE-202510 的手術報告（operationStartTime 在 2025-10-01 到 2025-10-31）。"
    print(f"User 3: {query3}")
    if args.raw:
        response3 = await agent.process_query_only_tool_result(
            query3,
            session="user123",
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        print("Agent 3 (raw):")
        print(response3)
    else:
        response3 = await agent.process_query_with_natural_response(
            query3,
            session="user123",
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        print("Assistant 3:")
        print(response3.answer)

    print("-----")
    query4 = "案號 CASE-202510 的所有報告"
    print(f"User 4: {query4}")
    if args.raw:
        response4 = await agent.process_query_only_tool_result(
            query4,
            session="user123",
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        print("Agent 4 (raw):")
        print(response4)
    else:
        response4 = await agent.process_query_with_natural_response(
            query4,
            session="user123",
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        print("Assistant 4:")
        print(response4.answer)

if __name__ == "__main__":
    asyncio.run(main())
