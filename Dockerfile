FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (kept minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# The repository root (this folder) is copied into /app/fast_mcp_client
WORKDIR /app/fast_mcp_client

COPY ./requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r ./requirements.txt

COPY . .

# Default config location inside container:
# - If you mount your own config, set FAST_MCP_CONFIG=/app/fast_mcp_client/config.yaml (or other path)
ENV FAST_MCP_CONFIG=/app/fast_mcp_client/config.yaml
# Hint for runtime: treat this environment as "in docker" for safe host binding.
ENV FAST_MCP_IN_DOCKER=1

EXPOSE 8081

# Basic container healthcheck (does not require MCP server to be reachable)
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=10 \
  CMD curl -fsS http://127.0.0.1:8081/agent/health || exit 1

CMD ["python", "agent_gateway.py"]
