import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Allow running as a script: `python fast_mcp_client/gateway_smoke_test.py`
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fast_mcp_client.agent_gateway import app


def main() -> None:
    with TestClient(app) as client:
        health = client.get("/agent/health")
        print("GET /agent/health:", health.status_code)
        print(health.json())

        tools = client.get("/agent/tools")
        print("GET /agent/tools:", tools.status_code)
        data = tools.json()
        servers = data.get("servers") or []
        print("servers:", len(servers))
        if servers:
            print("firstServer.toolsCount:", len(servers[0].get("tools") or []))


if __name__ == "__main__":
    main()
