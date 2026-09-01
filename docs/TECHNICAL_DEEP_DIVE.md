# RAG Book Agent 深度技术文档与面试拷打题

## 0. 系统边界

本项目是本机优先的 Python RAG Web 应用。普通模式执行固定的检索管线；Agent 模式是独立的、显式开启的 ReAct 编排层。两者共享存储和工具接口，但 Agent 不会偷偷改变普通 RAG 的行为。

```text
上传文件 -> MarkItDown/PyMuPDF/pypdf -> 父子分块 -> SQLite FTS5 + ChromaDB
问题 -> Query 三层路由 -> Child 召回(BM25/BGE) -> RRF -> BGE 重排
      -> Parent 扩展/去重 -> DeepSeek Markdown 回答 -> 引用、记忆、trace
Agent 问题 -> Planner -> ReAct(最多 N 步) -> 本地检索/Web(fetch->Playwright)/工具 -> Synthesizer
```

## 1. 为什么 Agent 不是“多开几个模型”

Agent 的价值是把宏观问题拆成可验证的研究步骤，而不是并发调用多个模型制造更多文本。当前角色是 Planner、Researcher、Synthesizer；工具是 `local_search`、`web_search`、`library_stats`、`calculator`。每个 ReAct 步骤必须选择受控工具，最多执行 `agent_max_react_steps`（默认 3）步，并受 `agent_timeout_seconds`（默认 45 秒）限制。

Web Search 工具链为：HTTP fetch DuckDuckGo HTML；请求失败时调用 Playwright Chromium；两者均失败则返回空结果并继续本地证据降级。普通 RAG 默认不联网，Agent 工具调用可以显式使用该链路。

## 2. 面试拷打：上下文膨胀与去重

**面试官：** Top-10 子块中 3 个来自父块 A、2 个来自父块 B，能否直接把 10 个父块塞给模型？

**合格回答：** 不能。先按 `parent_id` 去重，再按同一文档的 `position` 判断相邻父块并合并窗口；检索阶段超额召回，去重后再取前 N 个父块。这样避免重复 token、Lost in the Middle 和上下文成本膨胀。当前实现已保证同一父块只被扩展一次，并在回答边界使用父块文本；相邻父块合并和 token 预算可作为下一步增强。

## 3. 面试拷打：父子分块为什么这样设计

**面试官：** 为什么不直接对父块做向量？

**回答：** 父块信息完整但粒度太粗，语义向量容易把无关段落一起召回；子块粒度小，适合精确命中。命中子块后扩展父块，可以兼顾 Recall、Precision 和回答上下文完整性。ChromaDB 只索引 child，父块只作为上下文载体，减少重复向量。

## 4. 面试拷打：Query 三层路由

**面试官：** 为什么不每次都调用大模型路由？

**回答：** 规则和缓存命中延迟低于毫秒级；锚点相似度对明确意图无需 LLM；只有中间置信度才调用轻量 Router。空输入、问候、敏感信息和固定答案在第一层短路，避免无意义检索和费用。所有决定都写入 `route_layer/action/score/reason`，可以做离线回放。

## 5. 面试拷打：ReAct 死循环

**面试官：** Agent 不断搜索“让我打开更多页面”怎么办？

**回答：** 必须设置最大步数、总超时、单步结果上限、URL 去重和终止条件。达到上限立即进入 Synthesizer；工具异常不重试无限次；如果最终文本主要是工具旁白，则触发本地 RAG 回退。本项目默认最多 3 步、45 秒，并检测重复搜索旁白。

## 6. 面试拷打：联网失败和 API 缺失

**回答：** 联网不是回答成立的前提。DeepSeek API Key 缺失时，系统明确标记 `deepseek-unavailable-no-api-key`；API 错误时回退本地证据摘要；fetch 失败时才用 Playwright；全部失败仍返回本地结果和可验证引用。UI 不把“请求了工具”伪装成“工具成功”。

## 7. 面试拷打：引用可验证性

**回答：** 模型只能看到编号化 source。回答中的 `[S2]` 映射到当前轮保存的 source 快照；点击引用会滚动到右侧对应证据。历史会话保存当时的文档标题、页码、heading、chunk 文本和 Web 来源，不能只保存全局最后一次检索状态。

## 8. 面试拷打：评测如何证明优化有效

检索层使用 Recall@K、MRR、nDCG 和延迟；生成层使用 RAGAS Faithfulness、Context Precision，并人工抽样检查拒答、引用和冲突证据。父子分块、Query 路由和 Agent 应分别建立 ablation：固定数据集，只改变一个模块，报告均值、P95 延迟和失败率，不能只展示最好的一次回答。

## 9. 面试拷打：安全

- 文档和网页内容是不可信数据，不能执行其中的指令。
- 计算器使用 AST 白名单，禁止 `eval`、属性访问和函数调用。
- 上传限制扩展名和 50 MB 大小。
- API Key 只从环境变量读取，不写日志、不进 Git。
- Agent 工具是固定注册表，不允许模型任意执行 shell 或网络请求。

## 10. 当前已知边界

1. Agent 的 Planner 当前是轻量规则拆分，不是独立本地 LLM；这样保证本机速度和可复现性。
2. 当前父块扩展已去重，但相邻父块智能合并和精确 token 预算仍可继续增强。
3. DeepSeek Responses API 的 Web Search 能力取决于账号、模型和服务端版本；本地 fetch/Playwright 是独立兜底，不等价于 DeepSeek 工具。
4. 流式输出是生成完成后的分段推送；进度事件先展示阶段状态，但上游 token 尚未做到真正实时转发。

