#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${FAST_MCP_IMAGE:-fast-mcp-client:local}"
NAME="${FAST_MCP_CONTAINER:-fast-mcp-client}"
PORT="${FAST_MCP_PORT:-8081}"

if [[ ! -f "$ROOT/config.yaml" ]]; then
  echo "[docker-run] missing config.yaml; copy config.example.yaml -> config.yaml first" >&2
  exit 1
fi

echo "[docker-run] name=$NAME image=$IMAGE port=$PORT"
echo "[docker-run] mounting config.yaml -> /app/fast_mcp_client/config.yaml"

docker run --rm -it \
  --name "$NAME" \
  -p "$PORT:8081" \
  -e FAST_MCP_CONFIG=/app/fast_mcp_client/config.yaml \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  -v "$ROOT/config.yaml:/app/fast_mcp_client/config.yaml:ro" \
  "$IMAGE"

