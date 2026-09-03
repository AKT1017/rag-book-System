# LangGraph 研究 Agent

## 1. 定位

LangGraph Agent 已替代早期的固定多 Agent 页面。它与普通知识问答共享本地 RAG、DeepSeek 生成、会话和引用组件，但只有切换到“研究 Agent”子页面后才运行。

它适合比较、调研、架构分析和方案规划等需要多来源证据的问题。简单的单点书籍问答应使用普通 RAG，延迟更低。

## 2. 状态图

```text
START
  -> understand
  -> plan
  -> decide_action
       |-> local_search -> observe --|
       |-> web_search   -> observe --|-> decide_action
       |-> finalize -> evidence_review -> synthesize
  -> END
```

主要状态包括问题、会话、复杂度车道、子问题计划、本地结果、网页结果、下一行动、行动次数、观察结论、证据预算、运行轨迹和最终答案。

## 3. 节点职责

| 节点 | 作用 |
|---|---|
| `understand` | 按规则识别 fast/standard/deep 复杂度及原因 |
| `plan` | 复用 DeepSeek 将复杂问题拆为有限子问题，超时或格式错误时规则兜底 |
| `decide_action` | 根据本地证据、是否强制联网和剩余步数选择工具或结束 |
| `local_search` | 复用 BM25、BGE、RRF、子块重排和父块扩展 |
| `web_search` | DeepSeek 原生网页搜索优先，本地 Search-Read-Rank 降级 |
| `observe` | 汇总当前证据数量并返回决策节点 |
| `evidence_review` | 按 `parent_id`/URL 去重，执行两阶段阅读和上下文预算 |
| `synthesize` | 复用统一生成器输出直接、有结论的 Markdown 回答 |

## 4. 工具与约束

工具白名单定义在 `agent/tools.py`：

- `local_search`
- `web_search`
- `library_stats`
- `calculator`

当前决策循环主要使用前两项。Agent 不拥有 Shell、任意网络请求、任意代码执行或文件写入权限。`agent_max_react_steps` 默认限制为 3，网页与总流程分别受配置中的超时和上下文预算约束。

## 5. 原生 Web Search

当 `deepseek_web_search=true` 且 `.env` 存在 `RAG_BOOK_API_KEY` 时：

1. `web_search` 工具请求 `{api_base}/responses`。
2. 请求携带 `tools=[{"type":"web_search"}]` 和 `max_tool_calls`。
3. DeepSeek 服务端自主执行 `search` 与 `open_page`。
4. 应用解析 `web_search_call`、最终 `message`、`url_citation` 和 Markdown 链接。
5. 网页摘要与 URL 转换为证据，进入去重、预算和最终综合。
6. 原生请求失败或没有最终正文时，自动使用 DDGS/httpx/trafilatura/Playwright 通道。

不强制设置 `tool_choice`。实测强制工具时，服务端可能把输出预算全部用于连续工具调用而没有最终 message；自动工具选择配合 Agent 自身的显式 `web_search` 节点更稳定。

## 6. 前端可观测性

Agent 页面保留两个互补视图：

- 动态决策图：当前节点高亮，完成节点变色，展示工具循环位置。
- 文字时间线：展示车道、行动理由、结果数量、降级状态和证据预览。

最终答案下方继续使用统一的逐轮引用面板。本地来源显示标题、页码和片段；网页来源显示标题与可点击 URL。查询详情保存在 SQLite trace 和操作日志中。

## 7. 降级语义

- 规划 LLM 失败：规则拆分。
- 原生网页搜索失败：本地网页搜索。
- 全部网页搜索失败：继续使用本地 RAG。
- DeepSeek 最终生成失败：带引用的证据摘要。
- 输出只有工具旁白：拒绝展示旁白，降级为证据摘要。
- 达到行动上限：停止工具调用并整理已有证据。

这些状态会写入 `agent_web_status`、`agent_web_provider`、`agent_web_pipeline` 和 `agent_research`，不能把“请求过联网”伪装为“联网成功”。

## 8. 运行与验证

```powershell
.\.venv\Scripts\rag-book.exe web
.\.venv\Scripts\python.exe -m pytest tests/test_agent.py tests/test_deepseek_web.py tests/test_web_search.py
```

浏览器打开 `http://127.0.0.1:8008/`，切换到研究 Agent 页面即可运行。详细参数和故障排查见 `DEPLOYMENT.md`。
