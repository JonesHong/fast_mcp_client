(() => {
  const $ = (id) => document.getElementById(id);

  const baseUrlEl = $("baseUrl");
  const sessionIdEl = $("sessionId");
  const llmProviderEl = $("llmProvider");
  const llmModelEl = $("llmModel");
  const toolResultFullEl = $("toolResultFull");

  const messagesEl = $("messages");
  const eventLogEl = $("eventLog");
  const clearLogBtn = $("clearLog");
  const toggleLogOrderBtn = $("toggleLogOrder");
  const chatForm = $("chatForm");
  const messageInput = $("messageInput");
  const sendBtn = $("sendBtn");
  const stopBtn = $("stopBtn");

  let currentAbortController = null;
  let logOrder = "desc"; // desc: newest first
  const logLines = [];

  const PHASE_LABELS = {
    api_received: "已收到請求",
    tool_call_start: "開始呼叫 MCP",
    tool_call_done: "MCP 已回應",
    llm_stream_start: "LLM 串流回覆中",
  };

  function nowStr() {
    return new Date().toLocaleString();
  }

  function appendLog(line) {
    logLines.push(line);
    renderLog();
  }

  function renderLog() {
    const lines = logOrder === "desc" ? [...logLines].reverse() : logLines;
    eventLogEl.textContent = lines.join("\n") + (lines.length ? "\n" : "");
    if (logOrder === "desc") {
      eventLogEl.scrollTop = 0;
    } else {
      eventLogEl.scrollTop = eventLogEl.scrollHeight;
    }
  }

  function scrollMessagesToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function makeBubble({ role, title, meta }) {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${role}`;

    const header = document.createElement("div");
    header.className = "bubble-title";

    const badge = document.createElement("span");
    badge.className = `badge ${role}`;
    badge.textContent = title;

    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta || "";

    header.appendChild(badge);
    header.appendChild(metaEl);

    const content = document.createElement("div");
    content.className = "content";

    bubble.appendChild(header);
    bubble.appendChild(content);

    return { bubble, content, metaEl };
  }

  function prettyJson(obj) {
    try {
      return JSON.stringify(obj, null, 2);
    } catch {
      return String(obj);
    }
  }

  function parseSseBlock(block) {
    const lines = block.split("\n");
    let event = "message";
    const dataLines = [];

    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice("event:".length).trim();
        continue;
      }
      if (line.startsWith("data:")) {
        dataLines.push(line.slice("data:".length).trimStart());
        continue;
      }
    }

    const dataRaw = dataLines.join("\n");
    let data = dataRaw;
    try {
      data = JSON.parse(dataRaw);
    } catch {
      // keep raw text
    }

    return { event, data };
  }

  async function chatOnce({ message }) {
    const baseUrl = (baseUrlEl.value || "").replace(/\/+$/, "");
    const sessionId = sessionIdEl.value || "default";
    const llmProvider = (llmProviderEl.value || "").trim() || undefined;
    const llmModel = (llmModelEl.value || "").trim() || undefined;
    const toolResult = toolResultFullEl.checked ? "full" : "summary";

    const user = makeBubble({
      role: "user",
      title: "User",
      meta: nowStr(),
    });
    user.content.textContent = message;
    messagesEl.appendChild(user.bubble);

    const assistant = makeBubble({
      role: "assistant",
      title: "Assistant",
      meta: "",
    });
    assistant.content.textContent = "";
    messagesEl.appendChild(assistant.bubble);
    scrollMessagesToBottom();

    const toolDetails = document.createElement("details");
    toolDetails.className = "tool";
    const toolSummary = document.createElement("summary");
    toolSummary.textContent = "tool_call（done 後顯示）";
    const toolPre = document.createElement("pre");
    toolPre.textContent = "";
    toolDetails.appendChild(toolSummary);
    toolDetails.appendChild(toolPre);
    assistant.bubble.appendChild(toolDetails);

    const body = {
      sessionId,
      message,
      stream: true,
      toolResult,
    };
    if (llmProvider) body.llmProvider = llmProvider;
    if (llmModel) body.llmModel = llmModel;

    appendLog(`[${nowStr()}] POST ${baseUrl}/agent/chat ${prettyJson(body)}`);

    currentAbortController = new AbortController();
    sendBtn.disabled = true;
    stopBtn.disabled = false;

    let buffered = "";
    let requestId = null;
    let toolCallPayload = null;

    try {
      const resp = await fetch(`${baseUrl}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: currentAbortController.signal,
      });

      if (!resp.ok || !resp.body) {
        const text = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffered += decoder.decode(value, { stream: true });

        const parts = buffered.split("\n\n");
        buffered = parts.pop() || "";

        for (const part of parts) {
          const trimmed = part.trim();
          if (!trimmed) continue;

          const { event, data } = parseSseBlock(trimmed);
          appendLog(`[${nowStr()}] event=${event} data=${typeof data === "string" ? data : JSON.stringify(data)}`);

          if (data && typeof data === "object" && data.id && !requestId) {
            requestId = data.id;
          }

          if (event === "status" && data && typeof data === "object") {
            const phase = data.phase || "status";
            const label = PHASE_LABELS[phase] || phase;
            assistant.metaEl.textContent = `request=${data.id || requestId || "-"} · ${label}`;
            continue;
          }

          if (event === "delta" && data && typeof data === "object") {
            assistant.content.textContent += data.text || "";
            scrollMessagesToBottom();
            continue;
          }

          if (event === "tool_call" && data && typeof data === "object") {
            toolCallPayload = data;
            continue;
          }

          if (event === "error") {
            const msg = data && typeof data === "object" ? data.message : String(data);
            assistant.metaEl.textContent = `request=${requestId || "-"} · 發生錯誤`;
            assistant.content.textContent += `\n\n[Error] ${msg}`;
            scrollMessagesToBottom();
            continue;
          }

          if (event === "done" && data && typeof data === "object") {
            assistant.metaEl.textContent = `request=${data.id || requestId || "-"} · 完成`;
            if (toolCallPayload) {
              toolPre.textContent = prettyJson(toolCallPayload);
              toolDetails.open = true;
            } else {
              toolPre.textContent = "(no tool_call received)";
            }
            scrollMessagesToBottom();
            return;
          }
        }
      }
    } catch (err) {
      const msg = err?.name === "AbortError" ? "已停止串流" : String(err);
      appendLog(`[${nowStr()}] ERROR ${msg}`);
      assistant.metaEl.textContent = `request=${requestId || "-"} · 結束`;
      assistant.content.textContent += `\n\n[${msg}]`;
    } finally {
      sendBtn.disabled = false;
      stopBtn.disabled = true;
      currentAbortController = null;
      scrollMessagesToBottom();
    }
  }

  clearLogBtn.addEventListener("click", () => {
    logLines.length = 0;
    renderLog();
  });

  toggleLogOrderBtn.addEventListener("click", () => {
    logOrder = logOrder === "desc" ? "asc" : "desc";
    toggleLogOrderBtn.textContent = logOrder === "desc" ? "最新在上" : "最新在下";
    renderLog();
  });

  stopBtn.addEventListener("click", () => {
    if (currentAbortController) currentAbortController.abort();
  });

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = (messageInput.value || "").trim();
    if (!message) return;
    messageInput.value = "";
    messageInput.focus();
    await chatOnce({ message });
  });
})();
