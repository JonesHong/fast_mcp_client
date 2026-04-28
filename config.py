from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def repo_root() -> Path:
    # fast_mcp_client/config.py -> repo root = parents[1]
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    return repo_root() / "fast_mcp_client" / "config.yaml"


def load_config(path: str | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else Path(os.getenv("FAST_MCP_CONFIG") or default_config_path())
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path

    if not cfg_path.exists():
        return apply_env_overrides({})

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml config (expected mapping): {cfg_path}")

    return apply_env_overrides(data)


def get_cfg_value(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _env_truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay environment variables on top of YAML config.
    Env vars take precedence so the same image can be reconfigured at deploy time.
    """
    cfg = dict(cfg or {})

    # Gateway binding
    if (v := os.getenv("GATEWAY_HOST")) is not None:
        cfg.setdefault("gateway", {})["host"] = v
    if (v := os.getenv("GATEWAY_PORT")) is not None:
        try:
            cfg.setdefault("gateway", {})["port"] = int(v)
        except ValueError:
            pass
    if (v := os.getenv("GATEWAY_RELOAD")) is not None:
        cfg.setdefault("gateway", {})["reload"] = _env_truthy(v)

    # Logging
    if (v := os.getenv("LOGGING_VERBOSE")) is not None:
        cfg.setdefault("logging", {})["verbose"] = _env_truthy(v)

    # Time
    if (v := os.getenv("FAST_MCP_TIMEZONE")) is not None:
        cfg.setdefault("time", {})["timezone"] = v

    # MCP servers - quick mode (single server via env)
    mcp_url = os.getenv("MCP_SERVER_URL")
    if mcp_url:
        server: Dict[str, Any] = {
            "type": os.getenv("MCP_SERVER_TYPE", "streamable-http"),
            "url": mcp_url,
            "disabled": False,
        }
        auth = os.getenv("MCP_SERVER_AUTH")
        if auth:
            server["headers"] = {"Authorization": auth}
        name = os.getenv("MCP_SERVER_NAME", "report-api")
        cfg["mcp"] = {"servers": {name: server}}

    # MCP servers - advanced mode (full JSON map; overrides quick mode if set)
    mcp_json = os.getenv("MCP_SERVERS_JSON")
    if mcp_json:
        import json
        try:
            servers = json.loads(mcp_json)
            if isinstance(servers, dict) and servers:
                cfg["mcp"] = {"servers": servers}
        except Exception as exc:
            print(f"[config] MCP_SERVERS_JSON parse failed: {exc}")

    return cfg

