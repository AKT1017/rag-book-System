# LangGraph Agent 使用说明

当前版本在原有知识问答页面旁边提供独立的 `LangGraph Agent` 子页面。LangGraph 已合并并替代旧经典 Agent，成为唯一的研究 Agent 实现，可以在顶部导航直接切换。

## 已接入的最小流程

```text
START -> plan -> local_research -> web_research -> deduplicate -> synthesize -> END
```

- `plan`：将问题拆成最多三个子问题。
- `local_research`：复用现有 BM25、BGE、RRF、重排和父子分块 RAG。
- `web_research`：调用现有 fetch 搜索，失败时由 Playwright 兜底。
- `deduplicate`：按 `parent_id` 和网页 URL 去重。
- `synthesize`：把整理后的证据交给 DeepSeek 生成 Markdown 回答。

实现文件：`src/rag_book_agent/agent/langgraph_workflow.py`。

## 前端使用

1. 打开 `http://127.0.0.1:8008/`。
2. 点击顶部的 `LangGraph Agent`。
3. 页面会显示由 LangGraph 官方 API `get_graph().draw_mermaid_png()` 生成的流程图。
4. 输入问题并点击“运行图”。页面会展示回答、引用和本次节点轨迹。

流程图接口为：

```text
GET /api/langgraph/diagram
```

问答接口仍使用统一的 `/api/ask`，请求体增加：

```json
{"question": "客观评价麦卡杜", "agent_engine": "langgraph"}
```

## 依赖和安装

项目使用 LangGraph 1.2 系列，和当前 LangChain Core 版本兼容：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

首次安装若只想补齐依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install "langgraph>=1.2.11,<1.3"
```

官方 PNG 渲染需要 LangGraph 的 Mermaid 渲染依赖；如果本机没有可用的渲染服务，问答流程仍可运行，但 `/api/langgraph/diagram` 可能返回 503，此时可查看图的 Mermaid 文本或安装项目依赖后重试。

## 当前边界

这是受限的线性状态图，目前没有启用模型自主 Function Calling、多轮 ReAct 或检查点恢复。后续可在 `synthesize` 前增加 `evidence_judge` 条件节点，再将工具选择升级为受限的 LLM tool call 循环。
