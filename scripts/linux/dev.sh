#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "[dev] cwd=$ROOT"
echo "[dev] starting gateway (reload=true)"

python -m uvicorn fast_mcp_client.agent_gateway:app --host 0.0.0.0 --port 8081 --reload

