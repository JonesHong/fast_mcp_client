#!/usr/bin/env bash
set -euo pipefail

BASE="${FAST_MCP_BASEURL:-http://localhost:8081}"
BASE="${BASE%/}"

echo "[smoke] base=$BASE"
curl -fsS "$BASE/agent/health" && echo
curl -fsS "$BASE/agent/tools" && echo

