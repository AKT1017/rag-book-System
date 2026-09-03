# 当前实施状态

更新时间：2026-09-03
分支：`develop`

本文只描述当前代码已经实现和验证的能力。设计目标与后续设想见 `PROJECT_DESIGN.md` 和 `OPTIMIZATION_ROADMAP.md`。

## 已实现

| 能力 | 当前实现 | 状态 |
|---|---|---|
| Web 应用 | FastAPI + Uvicorn + 原生 HTML/CSS/JavaScript | 可运行 |
| 文件导入 | PDF、MD、TXT、DOCX、XLS/XLSX、PPTX、CSV | 可运行 |
| 异步导入 | 后台线程任务、进度、成功/警告/失败日志 | 可运行 |
| PDF | pymupdf4llm、RapidOCR、PyMuPDF、pypdf 回退 | 可运行 |
| 分块 | 跨页章节父块 + 重叠子块 + 页码映射 | 可运行 |
| 元数据/稀疏检索 | SQLite + FTS5/BM25 | 可运行 |
| 向量检索 | ChromaDB + BAAI/bge-small-zh-v1.5 | 可运行 |
| 融合/重排 | RRF + BAAI/bge-reranker-base | 可运行 |
| 生成 | DeepSeek Responses API | 可运行 |
| 原生联网 | DeepSeek `web_search`、工具调用和来源解析 | 可运行 |
| 本地联网降级 | DDGS + httpx + trafilatura + Playwright + BGE 重排 | 可运行 |
| Agent | LangGraph 有界观察-行动循环 | 可运行 |
| 会话记忆 | JSONL 近期对话 + Markdown 压缩摘要 | 可运行 |
| 引用 | 每轮本地/网页来源，右侧短预览与展开 | 可运行 |
| 文档管理 | 查看、启用/禁用、纳入检索、删除、查看父子块 | 可运行 |
| 日志 | 操作日志、查询结构化 trace、TXT 导出 | 可运行 |
| 评测 | Recall/Precision/Hit Rate/MRR/nDCG/延迟、四路线消融、可选 RAGAS | 可运行 |

## 已验证

- 当前内置黄金题：`demo` 10 题、`cmrc2018` 300 题，可按 5/10/20/50/100/300/全部抽样。
- DeepSeek 原生 `web_search` 实测返回 `web_search_call`，可解析最终消息和 8 个网页来源。
- Agent 实测完成本地检索、原生网页检索、证据预算和最终综合。
- 原生网页搜索失败时，本地 Search-Read-Rank 可独立返回正文级证据。
- 本地搜索会拒绝私网/本机地址，并对 URL、正文和单域名结果去重。

测试数量会随版本变化，应以本地 `python -m pytest` 和 CI 结果为准，而不是把本文数字作为长期承诺。

## 当前限制

### 文档解析

- RapidOCR 可处理扫描文字，但不是公式专用 OCR；复杂 LaTeX 公式恢复质量有限。
- pymupdf4llm 可保留常见表格结构，但复杂跨页表格、图表语义和手写内容仍可能丢失。
- OCR、结构化 Markdown 和分块质量目前主要通过规则指标和人工抽样评估。

### 检索与生成

- BGE-small 与 bge-reranker-base 适合单机 CPU，但不是最高精度的大模型组合。
- 查询改写和复杂度路由仍包含较多规则，尚未完成大规模阈值校准。
- DeepSeek 是远程 API；项目尚未内置 GGUF/llama.cpp 本地生成后端。
- 引用主要依靠提示约束和来源展示，尚缺完整的 claim-level citation validator。

### Agent 与 Web Search

- Agent 是受限研究 Agent，不是 Claude Code/Codex 类型的通用编码 Agent。
- DeepSeek 原生搜索的来源质量取决于服务端，应用只能解析、预算和展示返回结果。
- Playwright 需要单独下载 Chromium，且只在静态抓取失败时使用。
- 当前检查点主要是业务会话存储，尚未接入生产型 LangGraph 持久化 checkpointer。

### 产品与运维

- 当前为单机、单用户定位，没有登录、RBAC、多租户知识库隔离和配额。
- 没有 Docker/Kubernetes、数据库迁移系统、指标监控、告警和自动备份。
- 前端是原生 JavaScript，不是 React/Vue 工程；优点是部署轻，代价是大型功能继续增长时维护成本会上升。

## 建议优先级

1. 增加 citation precision/recall、拒答率、nDCG 和路线 ablation 的固定回归门禁。
2. 为 LangGraph 增加 SQLite checkpointer，实现进程重启后的任务恢复。
3. 增加公式、表格和 OCR 的专门黄金样本与导入质量评测。
4. 将 Query Router 阈值用真实问题集校准，而不是继续添加关键词。
5. 公网部署前补鉴权、限流、上传隔离、备份和监控。

## 结论

当前版本已经完成个人本机知识库所需的导入、索引、检索、重排、生成、引用、记忆、Agent、联网、日志和评测闭环。它适合演示、学习和个人资料研究；距离生产级多用户平台的主要差距在权限、高可用、自动质量门禁和复杂文档解析，而不是缺少基本 RAG 流程。
