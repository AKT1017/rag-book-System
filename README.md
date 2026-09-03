# RAG Book Agent

本机优先的中文文档 RAG 与有界研究 Agent。项目使用 Python、FastAPI 和原生 Web 前端，可异步导入 PDF、Markdown、TXT、Word、Excel、PPT、CSV，完成父子分块、混合检索、BGE 重排、DeepSeek 生成、原生联网搜索、引用追踪、会话记忆、运行日志和评测闭环。

## 当前能力

- **文档处理**：`pymupdf4llm -> RapidOCR -> PyMuPDF -> pypdf` 回退链，兼容电子 PDF 与扫描 PDF
- **结构化分块**：章节感知父子分块；子块检索和重排，父块扩展为生成上下文
- **混合检索**：SQLite FTS5/BM25 + ChromaDB + `BAAI/bge-small-zh-v1.5` + RRF
- **精排**：复用本地 `BAAI/bge-reranker-base` Cross-Encoder
- **问答生成**：DeepSeek Responses API；不可用时返回可追溯的本地证据摘要
- **研究 Agent**：LangGraph 有界观察-行动循环，可调用本地 RAG 与 Web Search
- **联网研究**：DeepSeek 原生 `web_search` 优先，本地 Search-Read-Rank 管线自动降级
- **Web 工作台**：拖拽上传、后台进度、知识库管理、历史会话、逐轮引用、日志与评测页面
- **评测闭环**：Recall/MRR、CMRC2018 测试集以及可选 RAGAS

## 一分钟启动

要求 Python 3.10+，推荐 3.11/3.12。Windows PowerShell：

```powershell
git clone git@github.com:AKT1017/rag-book-System.git
cd rag-book-System
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\rag-book.exe web
```

打开 [http://127.0.0.1:8008/](http://127.0.0.1:8008/)。已有环境日常只需最后一行。

配置 DeepSeek 时，将 `.env.example` 复制为 `.env` 并填写：

```text
RAG_BOOK_API_KEY=sk-你的密钥
```

密钥、上传资料、数据库和模型均被 `.gitignore` 排除。完整安装、模型下载、配置与排错见 [部署手册](docs/DEPLOYMENT.md)。

## 工作原理

### 本地 RAG

```text
文件 -> 解析/OCR -> 章节父子分块 -> SQLite FTS5 + ChromaDB
问题 -> Query 处理 -> BM25 + BGE 稠密召回 -> RRF -> BGE 重排
     -> 子块命中后扩展父块 -> DeepSeek 综合 -> Markdown 答案与 [S1] 引用
```

### LangGraph Agent

```text
START -> 理解问题 -> 规划 -> 决定行动
                              |-> local_search --|
                              |-> web_search ----|-> 观察 -> 再决策
                              |-> finalize
             -> 证据去重与预算 -> 最终综合 -> END
```

`web_search` 首先调用 DeepSeek Responses API 原生工具，并解析服务端 `search/open_page`、最终消息和网页引用。失败时降级为：

```text
DDGS 候选发现 -> URL 去重/域名限额 -> httpx 并发读取
-> trafilatura 正文抽取 -> Playwright 兜底 -> BGE 重排
```

Agent 有最大行动步数、上下文预算和失败降级，不允许任意 Shell 或文件操作。

## 项目结构

```text
src/rag_book_agent/
├─ ingest/             # 多格式加载、PDF 解析与 RapidOCR
├─ retrieval/          # BM25、向量召回、RRF 和 BGE 重排
├─ generation/         # DeepSeek Responses API、原生搜索响应解析
├─ agent/              # LangGraph 状态图与白名单工具
├─ web/                # FastAPI API 与原生前端
├─ chunking.py         # 章节感知父子分块
├─ storage.py          # SQLite、FTS5、文档与评测数据
├─ vector_store.py     # ChromaDB 适配器
├─ web_search.py       # 本地 Search-Read-Rank 降级通道
├─ memory.py           # JSONL 近期记忆与 Markdown 摘要
└─ service.py          # 业务编排入口
tests/                 # 单元与工作流回归测试
scripts/               # 数据集、路线评测和 RAGAS 脚本
samples/               # 可公开提交的示例资料
docs/                  # 架构、部署、状态和深度技术说明
```

## 常用命令

```powershell
# 启动
.\.venv\Scripts\rag-book.exe web

# 导入一个文件或目录
.\.venv\Scripts\rag-book.exe ingest "D:\books"

# 环境检查与统计
.\.venv\Scripts\rag-book.exe doctor
.\.venv\Scripts\rag-book.exe stats

# 测试
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts

# 重建演示知识库和黄金评测集
.\.venv\Scripts\python.exe scripts\rebuild_demo_dataset.py
```

## 文档

- [架构说明](docs/ARCHITECTURE.md)：模块边界与完整数据链路
- [部署手册](docs/DEPLOYMENT.md)：安装、模型、DeepSeek、运行和排错
- [LangGraph Agent](docs/LANGGRAPH_AGENT.md)：状态、节点、工具和降级
- [实现状态](docs/IMPLEMENTATION_STATUS.md)：已实现能力与真实限制
- [深度技术文档](docs/TECHNICAL_DEEP_DIVE.md)：实现细节、边界情况和面试问答
- [项目设计](docs/PROJECT_DESIGN.md)：设计目标和长期约束
- [优化路线](docs/OPTIMIZATION_ROADMAP.md)：后续演进优先级
- [Agent 对比说明](docs/AGENT_ARCHITECTURE.md)：Agent 架构与通用编码 Agent 的差距

## 安全与边界

- `.env`、`data/`、`models/`、虚拟环境和缓存不会提交 Git。
- 本地网页抓取拒绝本机、内网和保留地址，降低 SSRF 风险。
- 当前定位是个人本机知识库与工程验证系统，不包含多租户、登录鉴权和生产级高可用。
- DeepSeek 是远程 API；Embedding、Reranker、OCR 和本地搜索组件均可本机部署。
