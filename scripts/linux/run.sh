#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "[run] cwd=$ROOT"
echo "[run] starting gateway (reload=false)"

python -m uvicorn fast_mcp_client.agent_gateway:app --host 0.0.0.0 --port 8081

