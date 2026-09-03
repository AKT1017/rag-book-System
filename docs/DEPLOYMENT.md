# RAG Book Agent 部署与运行手册

本文针对 Windows 本机部署，也适用于 Linux/macOS，命令中的路径按实际系统调整。

## 1. 环境要求

- Python 3.10 或更高版本（建议 3.11/3.12）
- 4 GB 以上可用内存；首次加载 BGE 重排模型建议 8 GB
- 约 2 GB 磁盘空间：虚拟环境、ChromaDB 和两个模型
- DeepSeek API Key（仅在需要生成式答案或联网搜索时需要）

PDF 首选 pymupdf4llm 生成结构化 Markdown；低文本或扫描页使用 RapidOCR ONNX。解析异常时依次回退 PyMuPDF 和 pypdf。项目声明 `cryptography` 以支持 pypdf 读取 AES 加密 PDF。

## 2. 安装项目

```powershell
cd D:\work_pro\rag-book-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

升级已有环境时执行同一条安装命令，以同步 PDF、Web Search 和 Agent 依赖。Playwright 仅是动态网页兜底；首次需要该能力时安装 Chromium：

```powershell
.\.venv\Scripts\python.exe -m playwright install chromium
```

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

`deepseek_web_search=true` 表示 Agent 优先调用 DeepSeek Responses API 的原生 `web_search` 工具，并解析服务端搜索调用和网页引用；`web_search_enabled=true` 控制普通问答是否主动启用本地网页适配器。Agent 在 DeepSeek 原生搜索失败时仍会自动使用本地 `ddgs -> httpx/trafilatura -> Playwright` 降级链路。不要把 Key 写入 `config.json`。

## 5. 初始化和导入资料

```powershell
.\.venv\Scripts\rag-book.exe init
.\.venv\Scripts\rag-book.exe ingest "D:\books"
.\.venv\Scripts\rag-book.exe stats
```

Web 页面也支持拖拽上传。支持扩展名：`.pdf`、`.md`、`.markdown`、`.txt`、`.docx`、`.xlsx`、`.xls`、`.pptx`、`.csv`。PDF、Word、Excel 和 PPT 会进入后台异步任务，页面可查看处理进度；导入后可在页面中启用/禁用文档，以及选择是否纳入知识库。

PDF 导入会保留软页码标记，先连续处理全文，再进入章节感知父子分块；多栏文档由 pymupdf4llm 负责结构化 Markdown。扫描页使用 RapidOCR。导入结果仍进入 ChromaDB、BM25、RRF 和重排管线，不需要额外迁移。

pymupdf4llm 作为首选转换层处理电子 PDF，并输出适合 RAG 的 Markdown；扫描页自动使用 RapidOCR。解析异常时回退到 PyMuPDF，再回退到 pypdf。转换后的 Markdown 不会绕过现有索引流程。

## 6. 启动 Web 前端

```powershell
.\.venv\Scripts\rag-book.exe web
```

浏览器打开 `http://127.0.0.1:8008/`。若 8008 被占用，可直接运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn rag_book_agent.web.app:app --host 127.0.0.1 --port 8009
```

页面顶部会显示 AI 连接状态。普通知识问答以本地资料为主；研究 Agent 会按复杂度和证据充分度决定是否调用原生 Web Search。应用不会强制发送 `tool_choice`，而是在显式网页研究节点中允许 DeepSeek 服务端自主完成搜索和打开页面，避免连续工具调用耗尽输出预算。

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

DeepSeek 原生搜索失败：先查看日志中的 `agent_web_pipeline`。系统会自动降级到本地 Search-Read-Rank；若本地动态页面也需要兜底，确认已执行 `python -m playwright install chromium`。Responses 端点必须支持 `tools=[{"type":"web_search"}]`。

## 10. 备份与升级

停止服务后备份 `data/` 和 `config.json`。升级代码后重新执行 `pip install -e .`；模型和数据库不需要重复下载。不要备份或上传 `.env`。
