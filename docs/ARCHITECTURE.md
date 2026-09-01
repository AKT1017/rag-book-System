# RAG Book Agent 架构说明

## 1. 当前定位

这是一个本机优先的中文书籍 RAG Web 应用。它使用 Python 编写，支持 PDF、Markdown、TXT 导入，使用 ChromaDB 保存向量，使用 SQLite 保存文档元数据和全文索引，浏览器前端负责上传、资料管理和问答。

项目不依赖 Docker 或独立数据库服务；模型文件和运行数据放在本机 `models/`、`data/` 中，均不提交到 Git。

## 2. 目录职责

```text
rag-book-agent/
├─ src/rag_book_agent/       # 应用源码
│  ├─ ingest/                # PDF/MD/TXT 加载与文本抽取
│  ├─ retrieval/             # BM25、BGE 向量、RRF、重排
│  ├─ generation/            # DeepSeek Responses API 与本地证据回退
│  ├─ web/                   # FastAPI 路由和静态前端
│  ├─ chunking.py            # 文本清洗、章节感知切片
│  ├─ storage.py             # SQLite 文档、chunk、评测数据
│  ├─ vector_store.py        # ChromaDB 封装
│  ├─ service.py             # 导入、检索、问答的业务门面
│  ├─ memory.py              # JSONL 近期记忆和 Markdown 压缩摘要
│  ├─ web_search.py          # fetch 搜索，Playwright 兜底
│  ├─ audit_log.py           # HTTP 操作审计日志
│  ├─ config.py              # config.json 与 .env 配置
│  └─ cli.py                 # 命令行入口
├─ src/rag_book_agent/web/static/ # HTML、CSS、JavaScript 前端
├─ scripts/                  # CMRC/RAGAS/路线评测和范本导入脚本
├─ samples/                  # 可导入的示例资料和问题集
├─ tests/                    # 单元、工作流、Web 接口测试
├─ docs/                     # 设计、状态、部署和评测文档
├─ config.json               # 本机默认配置（不含密钥）
└─ pyproject.toml            # 安装、依赖和命令入口
```

## 3. 一次问答的处理链

1. 前端向 `/api/ask/stream` 发送问题、会话 ID 和 `force_web`。
2. `QuestionProcessor` 做轻量问题清洗和术语扩展。
3. SQLite FTS5/BM25 与 ChromaDB 稠密向量分别召回。
4. 两路结果用 RRF 合并，再使用 `BAAI/bge-reranker-base` 对候选重排。
5. 根据配置调用 DeepSeek Responses API；未配置 API 时返回本地证据摘要。
6. 答案以 Markdown 形式返回，并通过 `[S1]` 引用对应本地证据。
7. `MemoryStore` 保存近期问答，并定期生成压缩摘要。

联网搜索有两种开关：`deepseek_web_search` 控制是否向 DeepSeek 发送内置工具，`web_search_enabled` 控制本地 DuckDuckGo 适配器。前端“强制联网搜索”只会强制 DeepSeek 工具调用，不代表服务端一定成功调用，真实调用结果应以 API 返回为准。

## 4. 数据与安全边界

- `data/rag_book.db`：SQLite 元数据、chunk、FTS5、评测题。
- `data/chroma/`：ChromaDB 持久化向量。
- `data/uploads/`：上传原文件。
- `data/memory/`：会话 JSONL 和压缩 Markdown。
- `data/logs/operations.log`：操作审计日志，可从前端导出 TXT。
- `models/`：本地 BGE 模型。
- `.env`：DeepSeek 密钥，仅本机保存。

这些路径已在 `.gitignore` 中排除。生产部署前还应增加登录、反向代理和上传鉴权。
