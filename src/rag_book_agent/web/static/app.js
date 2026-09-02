const fileInput = document.getElementById("file-input");
const chooseFiles = document.getElementById("choose-files");
const dropZone = document.getElementById("drop-zone");
const uploadResult = document.getElementById("upload-result");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const conversation = document.getElementById("conversation");
const sources = document.getElementById("sources");
const sourceTemplate = document.getElementById("source-template");
const documentList = document.getElementById("document-list");
const documentTemplate = document.getElementById("document-template");
const forceWeb = document.getElementById("force-web");
const agentMode = document.getElementById("agent-mode");
const langGraphForm = document.getElementById("langgraph-form");
const langGraphQuestion = document.getElementById("langgraph-question");
const langGraphSubmit = document.getElementById("langgraph-submit");
const langGraphResult = document.getElementById("langgraph-result");
const langGraphTrace = document.getElementById("langgraph-trace");
const langGraphAnswer = document.getElementById("langgraph-answer");
const langGraphDiagram = document.getElementById("langgraph-diagram");
const memoryKey = "rag-book-session-id";
const sessionId = localStorage.getItem(memoryKey) || crypto.randomUUID();
localStorage.setItem(memoryKey, sessionId);
let currentSessionId = sessionId;
const sessionList = document.getElementById("session-list");
let progressNode = null;

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function renderRetrieval(info) {
  const parts = [];
  if (info.sparse) parts.push("BM25");
  if (info.dense) parts.push("BGE 向量");
  if (info.rrf) parts.push("RRF");
  if (info.rerank) parts.push("重排");
  if (info.web_search) parts.push(info.web_search_mode === "forced" ? "Web Search · 强制" : "Web Search");
  if (info.agent) parts.push("Agent · " + (info.agent_roles || []).join(" / "));
  if (info.agent_engine === "langgraph") parts.push("LangGraph 状态图");
  let label = parts.length ? parts.join("  ·  ") : "未获取检索信息";
  if (info.web_search_status === "deepseek-unavailable-no-api-key") label += "  ·  DeepSeek 未配置 API Key";
  if (info.web_search_status === "deepseek-request-failed-local-fallback") label += "  ·  DeepSeek 失败，已回退本地";
  setText("retrieval-mode", label);
}

function switchPage(page) {
  const isGraph = page === "langgraph";
  document.getElementById("rag-page").hidden = isGraph;
  document.getElementById("langgraph-page").hidden = !isGraph;
  document.querySelectorAll(".app-nav-item").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === page);
  });
  if (isGraph) langGraphDiagram.src = `/api/langgraph/diagram?ts=${Date.now()}`;
}

function renderLangGraphTrace(trace) {
  if (!trace || !trace.length) {
    langGraphTrace.textContent = "本次没有可展示的节点轨迹。";
    return;
  }
  langGraphTrace.textContent = trace.map((item, index) => {
    const name = item.node || "unknown";
    const details = Object.entries(item).filter(([key]) => key !== "node").map(([key, value]) => `${key}=${value}`).join(" · ");
    return `${index + 1}. ${name}${details ? `  ${details}` : ""}`;
  }).join("\n");
}

async function loadStatus() {
  const response = await fetch("/api/status");
  const data = await response.json();
  setText("document-count", `${data.stats.documents} 份资料`);
  setText("chunk-count", data.stats.chunks);
  setText("question-count", data.stats.questions);

  const connection = document.getElementById("connection");
  connection.className = "connection";
  if (data.generation.configured && data.generation.key_available) {
    connection.textContent = `AI 已连接 · ${data.generation.model}`;
    connection.classList.add("connected");
  } else if (data.generation.configured) {
    connection.textContent = "缺少 DeepSeek API Key · 使用证据模式";
    connection.classList.add("warning");
  } else {
    connection.textContent = "证据模式 · 未配置 AI";
    connection.classList.add("warning");
  }
  await loadDocuments();
  await loadSessions();
}

async function loadSessions() {
  const response = await fetch("/api/sessions");
  const data = await response.json();
  sessionList.replaceChildren();
  if (!data.sessions.length) { sessionList.innerHTML = '<div class="documents-empty">暂无历史对话</div>'; return; }
  for (const item of data.sessions) {
    const button = document.createElement("button");
    button.className = "session-item" + (item.id === currentSessionId ? " active" : "");
    button.textContent = `${item.title} (${item.turn_count} 轮)`;
    button.addEventListener("click", () => switchSession(item.id));
    sessionList.append(button);
  }
}

async function switchSession(id) {
  currentSessionId = id;
  localStorage.setItem(memoryKey, id);
  conversation.replaceChildren();
  const response = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
  const data = await response.json();
  for (const turn of data.turns || []) {
    addMessage("你的问题", turn.question, "question-message");
    addMessage("DeepSeek 回答", turn.answer, "answer-message", turn.sources || []);
  }
  await loadSessions();
}

function newSession() {
  currentSessionId = crypto.randomUUID();
  localStorage.setItem(memoryKey, currentSessionId);
  conversation.innerHTML = '<div class="welcome-message"><p class="eyebrow">LOCAL KNOWLEDGE</p><h1>向你的资料提问</h1><p>回答只基于已导入内容，每项结论都保留可查看的证据来源。</p></div>';
  renderSources([]);
  loadSessions().catch(() => {});
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function patchDocument(id, payload) {
  const response = await fetch(`/api/documents/${id}`, {method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  if (!response.ok) throw new Error((await response.json()).detail || "更新失败");
}

async function loadDocuments() {
  const response = await fetch("/api/documents");
  const data = await response.json();
  documentList.replaceChildren();
  if (!data.documents.length) { documentList.innerHTML = '<div class="documents-empty">暂无导入文档</div>'; return; }
  for (const item of data.documents) {
    const node = documentTemplate.content.cloneNode(true);
    node.querySelector(".document-title").textContent = item.title;
    node.querySelector(".document-meta").textContent = `${item.file_type.toUpperCase()} · ${formatSize(item.size)} · ${item.chunk_count} 个片段`;
    const library = node.querySelector(".library-toggle"); library.checked = item.in_library; library.title = "勾选后纳入 RAG 知识库";
    const enabled = node.querySelector(".enabled-toggle"); enabled.checked = item.enabled;
    library.addEventListener("change", async () => { try { await patchDocument(item.id, {in_library: library.checked}); } catch (e) { library.checked = !library.checked; alert(e.message); } });
    enabled.addEventListener("change", async () => { try { await patchDocument(item.id, {enabled: enabled.checked}); } catch (e) { enabled.checked = !enabled.checked; alert(e.message); } });
    node.querySelector(".document-delete").addEventListener("click", async () => { if (!confirm(`确认删除“${item.title}”？`)) return; const result = await fetch(`/api/documents/${item.id}`, {method: "DELETE"}); if (result.ok) { await loadStatus(); } else alert("删除失败"); });
    const chunkButton = node.querySelector(".document-chunks");
    const chunkTree = node.querySelector(".chunk-tree");
    chunkButton.addEventListener("click", async () => {
      if (!chunkTree.hidden) { chunkTree.hidden = true; chunkButton.textContent = "查看父子分块"; return; }
      const response = await fetch(`/api/documents/${item.id}/chunks`);
      const data = await response.json();
      chunkTree.replaceChildren();
      for (const chunk of data.chunks || []) {
        const line = document.createElement("div");
        line.className = `chunk-node ${chunk.chunk_type}`;
        const relation = chunk.chunk_type === "parent" ? "Parent" : `Child · parent=${chunk.parent_id || "-"}`;
        line.textContent = `${relation} · #${chunk.id} · ${chunk.heading || "无标题"} · ${chunk.preview}`;
        chunkTree.append(line);
      }
      chunkTree.hidden = false;
      chunkButton.textContent = "收起父子分块";
    });
    documentList.append(node);
  }
}

function renderSources(items) {
  sources.replaceChildren();
  if (!items.length) {
    sources.textContent = "没有找到可引用的证据。";
    sources.className = "sources empty";
    return;
  }
  sources.className = "sources";
  for (const item of items) {
    const node = sourceTemplate.content.cloneNode(true);
    node.querySelector(".source-item").id = `source-S${items.indexOf(item) + 1}`;
    node.querySelector(".source-meta").textContent = `${item.title} · 第 ${item.page} 页 · ${item.score}`;
    node.querySelector(".source-heading").textContent = item.heading || "未命名章节";
    const textNode = node.querySelector(".source-text");
    const toggle = node.querySelector(".source-toggle");
    const previewLength = 180;
    const fullText = item.text || "";
    let expanded = fullText.length <= previewLength;
    textNode.textContent = expanded ? fullText : `${fullText.slice(0, previewLength)}...`;
    toggle.hidden = expanded;
    toggle.addEventListener("click", () => {
      expanded = !expanded;
      textNode.textContent = expanded ? fullText : `${fullText.slice(0, previewLength)}...`;
      toggle.textContent = expanded ? "收起" : "显示全部";
    });
    sources.append(node);
  }
}

function addMessage(label, text, className, messageSources = []) {
  const wrapper = document.createElement("article");
  wrapper.className = "message";
  const labelNode = document.createElement("div");
  labelNode.className = "message-label";
  labelNode.textContent = label;
  const body = document.createElement("div");
  body.className = className;
  body.innerHTML = renderMarkdown(text);
  wrapper.append(labelNode, body);
  wrapper._sources = messageSources;
  if (messageSources.length) {
    const evidenceButton = document.createElement("button");
    evidenceButton.className = "message-evidence";
    evidenceButton.textContent = `查看本轮引用（${messageSources.length}）`;
    evidenceButton.addEventListener("click", () => renderSources(messageSources));
    wrapper.append(evidenceButton);
  }
  conversation.append(wrapper);
  conversation.scrollTop = conversation.scrollHeight;
  return body;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
}

function renderMarkdown(value) {
  let html = escapeHtml(value || "");
  html = html.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
  html = html.replace(/^### (.*)$/gm, "<h4>$1</h4>").replace(/^## (.*)$/gm, "<h3>$1</h3>").replace(/^# (.*)$/gm, "<h2>$1</h2>");
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/^[-*] (.*)$/gm, "<li>$1</li>").replace(/(?:<li>.*<\/li>\n?)+/g, (block) => `<ul>${block}</ul>`);
  html = html.replace(/\[S(\d+)\]/g, '<a class="citation" href="#source-S$1" data-source="S$1">[S$1]</a>');
  return html.replace(/\n/g, "<br>");
}

document.addEventListener("click", (event) => {
  const link = event.target.closest("a[data-source]");
  if (!link) return;
  event.preventDefault();
  const message = link.closest(".message");
  if (message && message._sources && message._sources.length) renderSources(message._sources);
  const target = document.getElementById(`source-${link.dataset.source}`);
  if (!target) return;
  target.scrollIntoView({behavior: "smooth", block: "center"});
  target.classList.add("source-highlight");
  setTimeout(() => target.classList.remove("source-highlight"), 1400);
});

async function uploadFiles(files) {
  if (!files.length) return;
  const data = new FormData();
  for (const file of files) data.append("files", file);
  uploadResult.textContent = `正在导入 ${files.length} 个文件...`;
  try {
    const response = await fetch("/api/upload", { method: "POST", body: data });
    const result = await readJsonOrText(response);
    if (!response.ok) throw new Error(result.detail || result.message || String(result) || "导入失败");
    const info = result.import;
    uploadResult.textContent = `已导入 ${info.imported} 份资料，新增 ${info.chunks} 个片段，跳过 ${info.skipped} 份。`;
    await loadStatus();
  } catch (error) {
    uploadResult.textContent = `导入失败：${error.message}`;
  } finally {
    fileInput.value = "";
  }
}

async function readJsonOrText(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  return {message: (await response.text()).slice(0, 500)};
}

chooseFiles.addEventListener("click", () => fileInput.click());
document.getElementById("refresh-documents").addEventListener("click", loadDocuments);
document.getElementById("export-logs").addEventListener("click", () => {
  const link = document.createElement("a");
  link.href = "/api/logs/export";
  link.download = "rag-book-operations.txt";
  document.body.append(link);
  link.click();
  link.remove();
});
document.getElementById("new-session").addEventListener("click", newSession);
document.querySelectorAll(".app-nav-item").forEach((button) => button.addEventListener("click", () => switchPage(button.dataset.page)));
document.getElementById("refresh-langgraph").addEventListener("click", () => { langGraphDiagram.src = `/api/langgraph/diagram?ts=${Date.now()}`; });
fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  const welcome = conversation.querySelector(".welcome-message");
  if (welcome) welcome.remove();
  addMessage("你的问题", question, "question-message");
  questionInput.value = "";
  askButton.disabled = true;
  askButton.textContent = "思考中";
  try {
    const response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: currentSessionId, force_web: forceWeb ? forceWeb.checked : false, agent_mode: agentMode.checked }),
    });
    if (!response.ok) throw new Error("问答失败");
    const answerBody = addMessage(agentMode.checked ? "Agent 回答" : "DeepSeek 回答", "", "answer-message streaming");
    const reader = response.body.getReader();
    const decoder = new TextDecoder(); let buffer = ""; let mode = "api";
    while (true) {
      const {value, done} = await reader.read(); if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) { if (!line) continue; const event = JSON.parse(line);
        if (event.type === "progress") {
          if (!progressNode) progressNode = addMessage("Agent 工作进度", "", "progress-message");
          progressNode.textContent += (progressNode.textContent ? "\n" : "") + "● " + event.text;
          conversation.scrollTop = conversation.scrollHeight;
        }
        if (event.type === "meta") { mode = event.mode; setText("answer-mode", mode === "api" ? "AI 回答" : "证据模式"); renderRetrieval(event.retrieval || {}); }
        if (event.type === "delta") { answerBody.dataset.raw = (answerBody.dataset.raw || "") + event.text; answerBody.innerHTML = renderMarkdown(answerBody.dataset.raw); conversation.scrollTop = conversation.scrollHeight; }
        if (event.type === "done") {
          renderSources(event.sources || []);
          const evidenceButton = document.createElement("button");
          evidenceButton.className = "message-evidence";
          evidenceButton.textContent = `查看本轮引用（${(event.sources || []).length}）`;
          evidenceButton.addEventListener("click", () => renderSources(event.sources || []));
          answerBody.parentElement.append(evidenceButton);
        }
      }
    }
    answerBody.classList.remove("streaming");
    if (progressNode) { progressNode.textContent += "\n● 已完成，正在展示最终答案"; progressNode = null; }
  } catch (error) {
    addMessage("请求失败", error.message, "answer-message error-message");
  } finally {
    askButton.disabled = false;
    askButton.textContent = "发送";
    questionInput.focus();
  }
});

langGraphForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = langGraphQuestion.value.trim();
  if (!question) return;
  langGraphSubmit.disabled = true;
  langGraphSubmit.textContent = "运行中";
  langGraphResult.textContent = "LangGraph 正在依次执行计划、本地研究、网页研究、去重和综合节点...";
  langGraphTrace.textContent = "图正在运行，等待节点结果...";
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question, session_id: currentSessionId, agent_engine: "langgraph"}),
    });
    const data = await readJsonOrText(response);
    if (!response.ok) throw new Error(data.detail || data.message || "LangGraph 运行失败");
    langGraphResult.innerHTML = renderMarkdown(data.answer || "");
    langGraphAnswer._sources = data.sources || [];
    renderLangGraphTrace((data.retrieval || {}).agent_research || []);
    renderRetrieval(data.retrieval || {});
    setText("answer-mode", data.mode || "LangGraph");
    renderSources(data.sources || []);
  } catch (error) {
    langGraphResult.textContent = `运行失败：${error.message}`;
    langGraphTrace.textContent = "工作流未能完成，请查看服务端日志。";
  } finally {
    langGraphSubmit.disabled = false;
    langGraphSubmit.textContent = "运行图";
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askForm.requestSubmit();
  }
});

loadStatus().catch(() => {
  document.getElementById("connection").textContent = "无法连接本地服务";
});
