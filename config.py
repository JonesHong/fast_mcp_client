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
        return {}

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml config (expected mapping): {cfg_path}")

    return data


def get_cfg_value(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
