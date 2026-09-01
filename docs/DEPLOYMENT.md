# RAG Book Agent 部署与运行手册

本文针对 Windows 本机部署，也适用于 Linux/macOS，命令中的路径按实际系统调整。

## 1. 环境要求

- Python 3.10 或更高版本（建议 3.11/3.12）
- 4 GB 以上可用内存；首次加载 BGE 重排模型建议 8 GB
- 约 2 GB 磁盘空间：虚拟环境、ChromaDB 和两个模型
- DeepSeek API Key（仅在需要生成式答案或联网搜索时需要）

PDF 解析使用 PyMuPDF（成熟的 MuPDF Python 绑定）；如果安装失败或单个 PDF 不兼容，会自动回退到 pypdf。项目同时声明 `cryptography`，用于 pypdf 读取 AES 加密 PDF。扫描版 PDF 若没有文本层仍需要额外 OCR，当前不会把 OCR 运行时强制安装进主环境。

## 2. 安装项目

```powershell
cd D:\work_pro\rag-book-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

升级已有环境时执行同一条安装命令，以安装新增的 `PyMuPDF` 依赖。

检查安装：

```powershell
.\.venv\Scripts\rag-book.exe doctor
.\.venv\Scripts\python.exe -m pytest
```

## 3. 下载并缓存开源模型

默认模型为 `BAAI/bge-small-zh-v1.5`（中文稠密向量）和 `BAAI/bge-reranker-base`（Cross-Encoder 重排）。可以使用 Hugging Face 下载到项目目录：

```powershell
.\.venv\Scripts\python.exe -m pip install -U huggingface_hub
.\.venv\Scripts\huggingface-cli.exe download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5
.\.venv\Scripts\huggingface-cli.exe download BAAI/bge-reranker-base --local-dir models/bge-reranker-base
```

离线配置应保持：

```json
"embedding_model": "models/bge-small-zh-v1.5",
"embedding_local_files_only": true,
"reranker_model": "models/bge-reranker-base",
"reranker_local_files_only": true
```

如果暂时不下载模型，将自动使用确定性的轻量回退算法，功能可运行但语义召回质量会下降。

## 4. 配置 DeepSeek

推荐在项目根目录创建 `.env`（该文件不会提交 Git）：

```text
RAG_BOOK_API_KEY=sk-你的密钥
```

`config.json` 中配置 API：

```json
{
  "api_base": "https://api.deepseek.com",
  "api_model": "deepseek-chat",
  "api_key_env": "RAG_BOOK_API_KEY",
  "deepseek_web_search": true,
  "web_search_enabled": false
}
```

`deepseek_web_search=true` 表示允许请求中携带 DeepSeek `web_search` 工具；`web_search_enabled=true` 才会启用本地 DuckDuckGo fetch/Playwright 适配器。不要把 Key 写入 `config.json`。

## 5. 初始化和导入资料

```powershell
.\.venv\Scripts\rag-book.exe init
.\.venv\Scripts\rag-book.exe ingest "D:\books"
.\.venv\Scripts\rag-book.exe stats
```

Web 页面也支持拖拽上传。支持扩展名：`.pdf`、`.md`、`.markdown`、`.txt`。导入后可在页面中启用/禁用文档，以及选择是否纳入知识库。

PDF 导入会逐页保留页码，并按页面文本块的阅读顺序抽取内容；多栏文档通常比基础文本抽取更稳定。导入结果仍进入原有切分、ChromaDB、BM25、RRF 和重排管线，不需要额外迁移。

MarkItDown 作为首选转换层处理 PDF；不兼容时自动回退到 PyMuPDF，再回退到 pypdf。转换后的 Markdown 不会绕过现有索引流程。

## 6. 启动 Web 前端

```powershell
.\.venv\Scripts\rag-book.exe web
```

浏览器打开 `http://127.0.0.1:8008/`。若 8008 被占用，可直接运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn rag_book_agent.web.app:app --host 127.0.0.1 --port 8009
```

页面顶部会显示 AI 连接状态。提问时勾选“强制联网搜索”会把 `tool_choice` 发送给 DeepSeek；不勾选时由模型自行决定是否使用工具。

## 7. 常用 API

- `GET /api/status`：运行状态、模型和文档统计
- `GET /api/documents`：文档列表
- `POST /api/upload`：上传文件
- `PATCH /api/documents/{id}`：启用/禁用、加入/移出知识库
- `DELETE /api/documents/{id}`：删除文档
- `GET /api/logs/export`：导出全部操作日志为 TXT
- `GET /api/sessions`：历史会话列表
- `GET /api/sessions/{id}`：恢复某个会话的完整问答
- `POST /api/ask`：普通 JSON 问答
- `POST /api/ask/stream`：NDJSON 流式 Markdown 问答

流式接口请求示例：

```json
{"question":"什么是 RRF？","session_id":"demo","force_web":false}
```

## 8. 评测与测试

运行自动化测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

导入成熟范本并做路线评测：

```powershell
.\.venv\Scripts\python.exe scripts/load_reference_template.py
.\.venv\Scripts\rag-book.exe eval --top-k 10
```

RAGAS 使用独立环境，避免污染主服务：

```powershell
python -m venv .venv-ragas
.\.venv-ragas\Scripts\python.exe -m pip install -r requirements-ragas.txt
.\.venv-ragas\Scripts\python.exe -m pip install -e . --no-deps
.\.venv-ragas\Scripts\python.exe scripts/run_ragas_eval.py --limit 30
```

报告位于 `data/reports/`。Recall/MRR 衡量检索命中，RAGAS 的 Faithfulness/Context Precision 衡量答案与上下文质量，二者应结合人工抽样解读。

## 9. 故障排查

`AI 未连接`：检查 `.env` 是否存在、变量名是否为 `RAG_BOOK_API_KEY`，并运行 `rag-book doctor`。

模型加载失败：确认模型目录完整且配置路径相对于项目根目录；删除错误缓存后重新下载。

上传后搜不到：检查文档是否“启用”且“纳入知识库”，然后重新导入或重启服务。

DeepSeek 搜索参数报错：不同 API 版本的工具字段可能变化，先关闭 `deepseek_web_search` 使用本地证据模式，再依据服务端错误调整 `answerer.py` 的 Responses 请求格式。

## 10. 备份与升级

停止服务后备份 `data/` 和 `config.json`。升级代码后重新执行 `pip install -e .`；模型和数据库不需要重复下载。不要备份或上传 `.env`。
