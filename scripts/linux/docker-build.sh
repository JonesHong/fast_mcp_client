#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${FAST_MCP_IMAGE:-fast-mcp-client:local}"
echo "[docker-build] image=$IMAGE"
docker build -t "$IMAGE" .

