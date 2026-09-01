# RAG Book Agent 当前技术栈与实施状态

更新时间：2026-09-01（Agent 检索闭环版本）
项目路径：`D:\work_pro\rag-book-agent`

本文以当前代码和实际运行结果为准，区分“已经可用的实现”和设计文档中的后续目标。

## 1. 当前定位

这是一个本机优先的 Python RAG 原型，已经形成“导入资料 -> 建立索引 -> 检索/重排 -> 生成回答 -> 查看引用 -> 反馈/评测”的可运行闭环。当前版本适合个人知识库和工程验证，不应描述为生产级大规模 RAG 平台。

## 2. 实际技术栈

| 层 | 当前实际选型 | 状态 |
|---|---|---|
| 运行时 | Python 3.10+，当前环境 Python 3.13 | 已运行 |
| Web 后端 | FastAPI + Uvicorn | 已运行 |
| 前端 | 原生 HTML/CSS/JavaScript | 已运行 |
| TUI | 已移除，保留 Web 前端 | 已完成 |
| PDF 解析 | MarkItDown -> PyMuPDF -> pypdf 回退，支持常见加密 PDF | 已运行 |
| Markdown/TXT | Python 标准库 UTF-8 读取 | 已运行 |
| 稀疏检索 | SQLite FTS5/BM25 | 已运行 |
| 稠密检索 | ChromaDB + BAAI/bge-small-zh-v1.5；模型不可用时 HashEmbedding 回退 | 已运行 |
| 融合 | Reciprocal Rank Fusion，默认 `rrf_k=60` | 已运行 |
| 重排 | BAAI/bge-reranker-base cross-encoder；模型不可用时规则重排回退 | 已运行 |
| 生成 | DeepSeek Responses API | 已运行 |
| Web 检索 | DeepSeek 服务端 `web_search`（可选）与本地 DuckDuckGo fetch -> Playwright 回退 | 可选 |
| Agent | Planner -> Researchers -> evidence deduplication -> DeepSeek synthesizer | 已运行 |
| 无 API 降级 | 本地证据摘要模式 | 已运行 |
| 存储 | SQLite + FTS5 | 已运行 |
| 评测 | Recall/MRR 确定性评测 + RAGAS 0.2.15 | 已运行 |
| 测试 | pytest + Ruff | 已运行 |

主应用依赖见 `pyproject.toml`，RAGAS 使用独立的 `.venv-ragas`，避免和 Textual/Rich 依赖冲突。

## 3. 从导入到回答的实际流程

### 3.1 文件导入

支持 `.pdf`、`.md`、`.markdown`、`.txt`。导入器会解析文件、计算 SHA-256，并按文件路径和哈希跳过未变化文件。上传接口限制 50 MB，并使用文件名 basename 防止路径穿越。

当前不足：扫描 PDF 没有 OCR；PDF 表格、版面坐标、图片文字和复杂脚注恢复能力有限。PyPA 范本的 RST 文档是通过复制为 Markdown 后导入的，原生 RST 仍未纳入支持列表。

### 3.2 分块

`ChapterChunker` 按 Markdown 标题和段落切分，默认父块 3600 字符、子块 900 字符，并保留标题、页码、位置、`parent_id` 和 `chunk_type`。子块参与稀疏/稠密检索，命中后展开父块作为生成上下文。PDF 当前每页视作一个逻辑页面。

当前不足：没有 tokenizer 级 token 预算、相邻父块动态合并、表格专用表示、原始/规范化/展示三套文本和 manifest 版本记录。旧文档必须重新导入才能获得父子结构。

### 3.3 索引与检索

SQLite FTS5 负责 BM25 风格稀疏召回；ChromaDB 使用本地 BGE-small-zh-v1.5 进行稠密检索；两路结果使用 RRF 融合，随后交由 BGE-reranker-base 重排。模型加载失败时才回退到 HashEmbedding 和规则重排。中文问题会做少量中英术语扩展，例如“发布”扩展为 `publish/upload/release`，“打包”扩展为 `build/package/packaging`。

当前不足：模型为轻量级中文 BGE，不是 BGE-M3 这类更大模型；术语扩展仍是人工规则，不是通用查询改写模型。查询路由是规则、锚点相似度和可选轻量 LLM 的三层架构，但尚未有系统化线上校准。

### 3.4 重排

候选结果优先由 BGE-reranker-base cross-encoder 重排；模型不可用时，按问题词覆盖、标题词覆盖和精确短语命中的规则重排作为降级方案。

当前不足：没有邻接父块合并、上下文 token 预算、上下文压缩和完整多路候选 ablation。

### 3.5 生成与引用

配置 `config.json` 指向 DeepSeek Responses API，密钥从 `.env` 的 `RAG_BOOK_API_KEY` 读取。提示词要求依据本地证据回答，并使用 `[S1]` 形式引用；Agent 综合会区分网页补充 `[W1]`。API 失败时回退为检索证据摘要。回答会记录 trace、来源 chunk 和耗时。

当前不足：DeepSeek 是远程 API，不是完全离线的本地开源模型；当前没有 Qwen GGUF/llama.cpp 本地生成后端。引用校验和失败后自动重生成尚未形成完整的强约束验证器，主要依赖提示词和来源展示。

### 3.6 文档管理

Web 左侧可以查看文档标题、类型、大小、页数、chunk 数和导入时间；可以勾选是否纳入 RAG 知识库、启用/禁用检索、删除文档。删除会清理 chunks 和 FTS 记录。检索只使用启用且纳入知识库的文档。

当前不足：目前是单一默认知识库，没有真正的 `Library` 实体、用户权限和跨库隔离；禁用是即时状态，不是设计稿中的 tombstone/purge 两阶段删除。

## 4. 评测与测试现状

### 已完成

- pytest：`10 passed`
- Ruff：检查通过
- 范本导入：92 个 PyPA 文档、598 个 chunks
- RAGAS：已使用 DeepSeek 评估 9 道可回答题
- 最近一次 RAGAS 报告：`data/reports/ragas-latest.json`
- RAGAS 结果约为：Faithfulness 平均 `0.844`，Context Precision 平均 `0.824`
- 确定性评测支持 Recall@K、MRR@K；不可回答题不计入这两个指标，单独标记为 `refusal_only`

### 当前评测限制

当前数据库混合了历史资料、范本和旧 golden questions，全库 Recall 会受到资料混杂影响。最近一次全库结果为 Recall@10 `0.70`、MRR@10 `0.6333`，这不能当作纯 PyPA 语料的最终基线。

还缺少 Recall@5/20、nDCG、citation precision/recall、答案相关性、拒答率自动统计、dense-only/BM25-only/hybrid/rerank ablation、延迟 p50/p95 和资源占用报告。RAGAS 是 LLM-as-a-judge，调用 DeepSeek 会产生 API 用量，不能替代人工抽样。

## 5. 完成度估计

| 能力域 | 完成度 | 说明 |
|---|---:|---|
| 本地导入与增量更新 | 80% | PDF/MD/TXT 可用，缺 OCR 和复杂版面 |
| 基础混合检索 | 75% | FTS5 + 轻量稠密 + RRF 可用，语义模型未接入 |
| 重排 | 55% | 有规则重排，缺 cross-encoder |
| 生成闭环 | 70% | DeepSeek 已接入，缺本地开源 LLM |
| 引用与安全 | 55% | 有引用提示、路径和密钥保护，校验器/权限不足 |
| Web/TUI 操作 | 75% | Web 上传、管理、问答可用，TUI 为基础版本 |
| 评测体系 | 60% | pytest、Recall/MRR、RAGAS 已有，指标和门禁不完整 |
| 生产运维 | 25% | 缺 manifest、备份、迁移、监控、回滚和多用户 |

综合判断：当前“可运行 MVP 闭环”约完成 65%~70%；距离设计稿定义的完整生产级闭环约完成 40%~50%。百分比是工程范围估计，不是模型质量分数。

## 6. 优先补齐顺序

1. 接入真正的开源本地 embedding（优先 BGE-small-zh-v1.5 或 BGE-M3 的 CPU 版本），保留 HashEmbedding 作为无模型降级。
2. 接入开源本地生成模型后端（低配 Qwen GGUF + llama.cpp），并保留 DeepSeek 作为可选远程 provider。
3. 增加 RapidOCR/PyMuPDF 版面解析和 RST 支持。
4. 实现 citation validator、拒答率、citation precision/recall 和检索 ablation。
5. 引入真正的 Library/权限模型、索引 manifest、软删除与可恢复备份。
6. 增加集成/E2E/安全测试和固定语料回归门禁。

## 7. 结论

当前版本已经可以在本机导入书籍和技术文档、管理文档、父子分块检索、调用 DeepSeek 回答、在 Agent 模式下进行受限多路研究，并进行 RAGAS 测评。它的主要短板不是“没有 RAG 或 Agent 流程”，而是 OCR、完全离线生成、父块上下文优化、引用自动校验、权限隔离和生产运维仍处于增强阶段。
