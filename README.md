# RAG Book Agent

一个本机优先的中文书籍 RAG Web 应用，支持 PDF、Markdown、TXT 导入、文档管理、ChromaDB 向量检索、BM25 稀疏检索、RRF 融合、BGE 重排、DeepSeek 生成、联网搜索、引用证据、会话记忆和 RAGAS 评测。

## 快速开始

```powershell
cd D:\work_pro\rag-book-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\rag-book.exe init
.\.venv\Scripts\rag-book.exe ingest samples
.\.venv\Scripts\rag-book.exe web
```

浏览器打开 `http://127.0.0.1:8008/`。详细安装、模型下载、DeepSeek 配置、API、评测和故障排查见 [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)。

## 主要能力

- PDF、Markdown、TXT 解析和章节感知切片
- SQLite 元数据/FTS5 + ChromaDB 持久化向量
- BM25 + BGE-small-zh-v1.5 + RRF + BGE-reranker-base
- 引用式 Markdown 回答、DeepSeek Responses API 和 `web_search`
- fetch 搜索与 Playwright 兜底
- JSONL 近期记忆与 Markdown 压缩摘要
- 路线 Recall/MRR 与 RAGAS Faithfulness/Context Precision 评测

## 文档导航

- [架构说明](docs/ARCHITECTURE.md)
- [部署与运行手册](docs/DEPLOYMENT.md)
- [项目设计](docs/PROJECT_DESIGN.md)
- [实现状态](docs/IMPLEMENTATION_STATUS.md)
- [优化路线](docs/OPTIMIZATION_ROADMAP.md)
- [深度技术文档与面试拷打题](docs/TECHNICAL_DEEP_DIVE.md)

## 配置 DeepSeek

在项目根目录创建 `.env`（不会提交 Git）：

```text
RAG_BOOK_API_KEY=sk-你的密钥
```

请确认 `config.json` 中的 `api_base`、`api_model` 和 `deepseek_web_search` 配置正确。未配置 API 时仍可使用本地证据模式。

## 测试和评测

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe scripts/load_reference_template.py
.\.venv\Scripts\rag-book.exe eval --top-k 10
```

RAGAS 需要单独安装 `requirements-ragas.txt`，完整命令见部署手册。

## GitHub 上传注意

仓库已排除 `.env`、`data/`、`models/`、虚拟环境和缓存。不要强制添加这些目录，尤其不要上传 API Key。
