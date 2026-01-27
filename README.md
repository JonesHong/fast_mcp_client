# fast_mcp_client (Agent Gateway)

這個資料夾提供一個 **FastAPI Agent Gateway**：前端只需要連到 Gateway（不是直接連 MCP server），Gateway 內部會用 MCP client 呼叫 tools，並用 LLM 產生「聊天式」回覆（支援 SSE 串流）。

## 先決條件

- MCP server 已啟動：`http://localhost:3007/mcp`
- （可選）Ollama 已啟動：`http://localhost:11434`
- （可選）OpenAI fallback：設定 `OPENAI_API_KEY`

## 設定檔（YAML）

統一設定在 `fast_mcp_client/config.yaml`（可用 `FAST_MCP_CONFIG` 指定其他路徑），包含：
- Gateway host/port
- MCP servers（已改為 YAML 管理）
- Ollama/OpenAI 模型偏好
- （可選）Ollama 定時 warm-up（避免閒置被卸載）

要啟用 warm-up：把 `llm.ollama_warmup.enabled` 設為 `true`，並調整 `interval_seconds`。

## 安裝依賴

```powershell
pip install -r fast_mcp_client/requirements.txt
```

## 啟動 Gateway

```powershell
# 預設走 OpenAI（需要 OPENAI_API_KEY）
python -m uvicorn fast_mcp_client.agent_gateway:app --host 0.0.0.0 --port 8081
```

也可以直接：

```powershell
python fast_mcp_client/agent_gateway.py
```

（建議用上面這個方式啟動；我們在程式內已關閉 WebSocket 支援 `ws=none`，可避免某些 websockets 套件的 deprecation 警告。）

若要 **優先使用 Ollama（OpenAI 當最後防線）**：

```powershell
$env:LLM_PRIMARY_PROVIDER="ollama"
$env:OLLAMA_HOST="http://localhost:11434"
$env:OLLAMA_MODEL="qwen2.5:7b-instruct"
# optional: 有 OPENAI_API_KEY 才會啟用 fallback
# optional: 指定 fallback 用的 OpenAI model（避免你帳號不支援預設 model）
# $env:OPENAI_FALLBACK_MODEL="gpt-4.1-mini"
python -m uvicorn fast_mcp_client.agent_gateway:app --host 0.0.0.0 --port 8081
```

## API（2+1）

- `GET /agent/health`
- `GET /agent/tools`
- `POST /agent/chat`（SSE 串流，或 `stream=false` 走 JSON）

## Swagger / OpenAPI

Gateway 啟動後可直接給前端同事使用：

- Swagger UI：`GET /docs`
- OpenAPI JSON：`GET /openapi.json`

### Response 形狀（方便 data binding）

- `GET /agent/health`
  - `status`, `time`
  - `llm.primary` / `llm.fallback`（`{ provider, model }`）
  - `servers[]`：`{ baseUrl, server, toolsCount }`
- `GET /agent/tools`
  - `servers[]`：`{ baseUrl, server, tools[] }`
  - `tools[]`：`{ name, description, inputProperties }`
- `POST /agent/chat`（SSE）
  - `status`：`{ id, phase, time, ... }`
  - `tool_call`：`{ id, time, mode, toolCall }`
  - `delta`：`{ id, text }`
  - `done`：`{ id, time, answer }`

### /agent/chat 請求欄位

- `sessionId`：session
- `message`：使用者訊息
- `stream`：是否 SSE（預設 `true`）
- `llmProvider` / `llmModel`：可選，覆蓋 YAML
- `toolResult`：`"none" | "summary" | "full"`（預設 `"summary"`；SSE 時決定 tool_call 事件帶回多少內容）

## 測試（非串流 JSON）

```powershell
curl.exe -s -X POST "http://localhost:8081/agent/chat" `
  -H "Content-Type: application/json" `
  -d "{\"sessionId\":\"u1\",\"message\":\"請列出目前可用的 MCP tools。\",\"stream\":false}"
```

## 測試（SSE 串流）

```powershell
curl.exe -N -X POST "http://localhost:8081/agent/chat" `
  -H "Content-Type: application/json" `
  -d "{\"sessionId\":\"u1\",\"message\":\"請查詢 2025/10 的手術報告\",\"stream\":true,\"toolResult\":\"full\"}"
```

SSE events：
- `status`：階段訊息（tool call / llm stream）
- `tool_call`：工具呼叫摘要（精簡後）
- `delta`：LLM 回覆增量文字
- `done`：結束（含最終 answer）

## 逐一比較不同開源模型（per-request 指定）

`POST /agent/chat` 可加：
- `llmProvider`: `"ollama"` 或 `"openai"`
- `llmModel`: 模型名稱（Ollama 請用 `ollama list` 看到的 tag）

例如：

```powershell
curl.exe -N -X POST "http://localhost:8081/agent/chat" `
  -H "Content-Type: application/json" `
  -d "{\"sessionId\":\"u1\",\"message\":\"請列出目前可用的 MCP tools。\",\"llmProvider\":\"ollama\",\"llmModel\":\"llama3.1:8b-instruct\"}"
```
