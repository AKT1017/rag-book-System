# Agent 模块架构说明

更新时间：2026-09-01  
适用分支：`develop`

## 1. 先给结论

本项目的 Agent 是一个面向知识研究问题的轻量编排器，不是通用电脑操作 Agent。它已经接入 DeepSeek 作为最终答案生成模型，也把原有 RAG 能力包装成 Agent 工具，但当前工具调用主要由 Python 编排器控制，并非完全由 LLM 通过原生 Function Calling 自主决定。

因此，准确的定位是：

> **受限的多角色 RAG Research Agent + DeepSeek Synthesizer**

而不是 Claude Code 的通用软件工程 Agent。

## 2. 当前真实架构

```text
Web API / 前端 Agent 开关
             |
             v
    AgentOrchestrator.run()
             |
             +--> Planner：确定性拆分问题，最多 3 个子问题
             |
             +--> Local Researcher：逐个调用 local_search
             |       BM25/FTS5 -> BGE embedding -> RRF -> reranker
             |       child 命中后展开 parent，上层按 parent_id 去重
             |
             +--> Web Researcher：并行调用 web_search
             |       DeepSeek 原生搜索 -> 本地 Search-Read-Rank 兜底
             |
             +--> Synthesizer：把本地证据、网页证据、记忆和计划
                     交给 DeepSeek Responses API
                     输出 Markdown，并要求 [S1]/[W1] 引用
             |
             +--> 失败/超时/工具旁白：降级为证据摘要
```

主要代码位置：

| 组件 | 文件 | 职责 |
|---|---|---|
| 编排器 | `src/rag_book_agent/agent/orchestrator.py` | 计划、研究、去重、调用生成边界 |
| 工具注册 | `src/rag_book_agent/agent/tools.py` | 暴露 RAG、网页、统计和计算器 |
| Agent 提示词 | `src/rag_book_agent/agent/prompts.py` | 规划和综合约束 |
| 本地检索 | `src/rag_book_agent/retrieval/engine.py` | 稀疏、稠密、融合、重排、父块展开 |
| 生成器 | `src/rag_book_agent/generation/answerer.py` | DeepSeek 请求、引用提示和降级 |
| API | `src/rag_book_agent/web/app.py` | Agent 开关、流式进度、回答和证据返回 |

## 3. 一次请求具体怎么走

### 3.1 入口

前端在 `/api/ask` 或 `/api/ask/stream` 发送 `agent_mode: true`。不开启时完全走普通 `RagService.ask()`，Agent 模块不会介入。

### 3.2 Planner

当前 `plan()` 使用中文逗号、分号等标点切分问题，最多保留三个子问题。简单问题保持为一个子问题。`PLANNER_PROMPT` 已存在，但当前没有调用 LLM 生成计划，这是为了降低延迟、减少不稳定输出和 API 消耗。

### 3.3 Local Researcher

每个子问题调用既有的 `service.search()`，所以 Agent 不会另起一套“简化版 RAG”。实际链路仍然是：

1. Query 预处理与术语扩展。
2. SQLite FTS5/BM25 稀疏召回。
3. ChromaDB + `BAAI/bge-small-zh-v1.5` 稠密召回。
4. RRF 融合候选。
5. `BAAI/bge-reranker-base` 重排，模型不可用时规则回退。
6. 子块命中后展开父块作为上下文。
7. Agent 层按 `parent_id` 保留最佳结果，避免同一父块重复塞入提示词。

SQLite 检索保持单线程，避免连接跨线程；网页任务可以并行。默认最多三步，整体受 `agent_timeout_seconds` 限制。

### 3.4 Web Researcher

Agent 的 `web_search` 工具优先调用 DeepSeek Responses API 原生 `web_search`。服务端负责 search/open_page 的自动续推，应用限制 `max_tool_calls`，并解析最终 message、`url_citation` 和 Markdown 来源链接，将它们转成右侧网页证据。原生调用失败时，自动降级到本地 Search-Read-Rank 管线：`ddgs` 发现候选，`httpx + trafilatura` 读取正文，静态抓取失败时由 Playwright 兜底，最后复用 BGE reranker 排序。

```text
DuckDuckGo HTML fetch
        |
        +-- 失败 --> Playwright Chromium
```

结果只作为网页补充证据，按 URL 去重，使用 `[W1]`、`[W2]` 编号。网页内容为空时仍会保留标题和链接，不把搜索旁白当成答案。

### 3.5 Synthesizer

Agent 研究完成后调用 `RagService._ask()`，但传入预计算的本地和网页证据。生成器把这些证据、会话记忆和子问题计划放进 DeepSeek 请求，由 DeepSeek 直接综合最终答案。

提示词要求：

- 先给结论，再给论据和限制。
- 本地证据使用 `[Sx]`，网页证据使用 `[Wx]`。
- 不描述“让我搜索”“我打开页面”等内部工具过程。
- 证据不足时说明不确定性，不编造事实。

## 4. 工具与“RAG Function Call”的准确现状

当前工具注册表是 Python 可调用函数：

```python
{
    "local_search": self.local_search,
    "web_search": self.web_search,
    "library_stats": self.library_stats,
    "calculator": self.calculator,
}
```

这已经完成了“把 RAG 封装为 Agent 工具”的第一步：工具有清晰名称、输入和返回值，且复用了成熟 RAG 管线。

但它还不是完整的 LLM Function Calling，原因是：

- DeepSeek 请求当前主要发送证据和综合提示，没有把上述 Python 函数的 JSON Schema 传给模型。
- 工具选择由 `AgentOrchestrator` 固定执行，不是模型返回 `tool_call` 后由运行时执行。
- 当前没有标准的 `assistant tool_call -> tool result -> assistant` 多轮消息循环。
- `agent_max_react_steps` 约束的是研究子问题数量，不是完整的模型驱动 ReAct 循环次数。

所以用户看到的“Agent”确实会使用 RAG，但它属于编排式调用，不应宣传为已经实现 Claude Code 级别的自主工具 Agent。

## 5. 和 Claude Code 的差距

| 维度 | 本项目 Agent | Claude Code 类 Agent |
|---|---|---|
| 目标 | 基于书籍/资料回答研究问题 | 完成软件工程任务 |
| 核心循环 | 固定 Planner -> Research -> Synthesize | LLM 自主观察、决策、调用工具、验证、再决策 |
| 工具调用 | Python 编排器固定调用 | 模型原生工具调用/协议驱动 |
| 工具范围 | 本地 RAG、网页搜索、统计、计算器 | 文件、Shell、Git、测试、浏览器、MCP 等 |
| 上下文 | 文档父块、网页结果、会话记忆 | 代码库、命令输出、变更 diff、长期任务状态 |
| 任务验证 | 引用、模式、基础测试 | 测试、编译、静态检查、运行结果和人工确认 |
| 安全模型 | 输入资料视为不可信，限制步数和超时 | 权限确认、命令风险控制、工作区边界、沙箱等 |
| 多 Agent | planner/researcher/synthesizer 是逻辑角色 | 通常可动态委派子任务并汇总结果 |
| 运行成本 | 低，最多少量搜索和一次综合生成 | 可能多轮调用，延迟和 token 成本更高 |

最重要的差距不是“有没有三个角色”，而是**决策权是否真正交给模型，以及是否存在可验证的循环**。当前项目的角色是模块边界；Claude Code 类型系统的角色更接近运行时中的动态策略。

## 6. 当前优点与不足

### 优点

- 复用原有 BM25、BGE、RRF、重排和父子分块，不会因 Agent 模式丢失 RAG 能力。
- 本地检索保持可复现，网页失败不会阻断本地回答。
- 限制子问题数量和总耗时，适合本机运行。
- 每轮记录检索方式、网页提供方、子问题和研究数量。
- Agent 失败会退回带引用的证据摘要，而不是返回内部异常。

### 不足

- Planner 目前不是 LLM 规划器。
- 工具尚未通过标准 JSON Schema 注册给 DeepSeek。
- 没有真正的模型驱动多轮 ReAct/Function Calling 循环。
- 网页抓取结果目前部分只有标题和 URL，正文抽取仍需增强。
- 没有独立的 Agent 轨迹回放、工具级评测和成本统计。
- 没有 Claude Code 那样的文件编辑、Shell、Git、权限审批能力，这属于产品边界而不是缺陷。

## 7. 推荐的最小升级路线

如果要把它升级为更接近 Claude Code 的 Agent，建议按以下顺序：

1. 为四个工具增加 JSON Schema 和统一 `ToolResult` 格式。
2. 在 DeepSeek 请求中注册 `tools`，解析模型返回的 `tool_calls`。
3. 实现最多 3 轮的消息循环：模型决策 -> Python 执行工具 -> 返回工具结果 -> 模型继续。
4. 增加停止条件：已有充分证据、模型给出最终答案、超时、重复调用、预算耗尽。
5. 加入工具结果校验、引用校验和轨迹保存。
6. 为 Agent 增加独立评测集：最终答案正确性、工具选择准确率、证据覆盖率、平均步数、p50/p95 延迟和 token 成本。

这条路线不会破坏普通 RAG：Agent 仍然只是前端显式开启的独立入口，普通问答继续使用原有 `RagService.ask()`。

## 8. 面试时的客观表述

推荐这样介绍：

> 我实现的是一个本地优先的 RAG Research Agent。它把现有混合检索封装为工具，由编排器完成问题拆解和多路研究，再把父块去重后的本地证据和网页补充交给 DeepSeek 综合。当前是受限编排式 Agent，具备工具封装、引用、超时和降级，但还没有完全实现 LLM 自主 Function Calling 和多轮 ReAct；下一步会用标准工具 Schema 和有限消息循环补齐。

这样的表述比声称“已经做成 Claude Code”更准确，也更经得起技术追问。
