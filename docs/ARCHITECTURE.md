# RAG Book Agent 架构

## 1. 系统定位

这是一个单机部署、Web 操作的中文文档 RAG 与研究 Agent。系统不要求 Docker 或外部数据库服务，业务数据保存在 SQLite 和 ChromaDB，本地模型保存在 `models/`，DeepSeek 仅承担可选的远程生成与原生网页研究。

## 2. 模块边界

| 模块 | 路径 | 职责 |
|---|---|---|
| 文档加载 | `ingest/` | PDF/OCR、Markdown、TXT、Word、Excel、PPT、CSV 转换 |
| 分块 | `chunking.py` | 标题识别、跨页连续文本、父子分块、页码映射 |
| 元数据 | `storage.py` | SQLite 文档、块、FTS5、评测题、查询轨迹 |
| 向量库 | `vector_store.py` | ChromaDB 写入、删除和相似度查询 |
| 检索 | `retrieval/` | Query 处理、BM25、BGE embedding、RRF 和 reranker |
| 生成 | `generation/` | DeepSeek Responses 请求、原生搜索响应和引用解析 |
| Agent | `agent/` | LangGraph 状态图、决策循环和白名单工具 |
| Web Search | `web_search.py` | 原生搜索失败后的本地 Search-Read-Rank |
| 会话 | `memory.py` | JSONL 近期对话与 Markdown 滚动摘要 |
| Web | `web/` | FastAPI、异步导入任务和原生 HTML/CSS/JS 前端 |
| 编排 | `service.py` | 导入、问答、状态、日志与反馈的统一业务入口 |

## 3. 文档导入链路

```text
上传文件
  -> 后台导入任务
  -> 格式加载器
     PDF: pymupdf4llm -> RapidOCR -> PyMuPDF -> pypdf
     Office/CSV: 对应轻量 Python 库
  -> 标准化 Page 列表
  -> 带软页码标记的连续文档
  -> 章节父块（约 3600 字符）
  -> 重叠子块（约 900 字符，重叠 220）
  -> SQLite 文档/块/FTS5
  -> ChromaDB 子块向量
```

父块保存完整上下文，子块参与索引。旧数据不会自动获得新分块策略，需要重新导入。

## 4. 本地检索链路

```text
问题
 -> 规则与缓存
 -> Query 规范化、术语扩展与轻量路由
 -> FTS5/BM25 稀疏召回 ┐
 -> BGE-small 稠密召回 ├-> RRF -> 子块 BGE reranker
                       ┘          -> Top-K 子块
                                  -> parent_id 去重和父块扩展
                                  -> 上下文预算
                                  -> DeepSeek / 本地证据摘要
```

子块先重排再扩展父块，避免长父块被 Cross-Encoder 截断。多个子块指向同一父块时只保留一次。

## 5. LangGraph Agent

Agent 是有界的观察-行动循环，而不是固定线性流水线：

```text
START -> understand -> plan -> decide_action
                               | local_search
                               | web_search
                               | finalize
          tool -> observe -----|
          finalize -> evidence_review -> synthesize -> END
```

- 首次行动优先建立本地事实基础。
- 深度问题或本地证据不足时调用网页研究。
- `agent_max_react_steps` 限制工具行动次数。
- `evidence_review` 按父块/URL 去重并执行本地、网页字符预算。
- 外部搜索失败不会终止图，仍可依据本地证据回答。

## 6. Web Search 双通道

首选通道：

```text
DeepSeek Responses API
-> tools=[{"type":"web_search"}]
-> 服务端 search/open_page 自动续推
-> 解析 message、url_citation 和 Markdown 来源链接
-> 转成网页证据
```

本地降级通道：

```text
DDGS -> URL 规范化/域名多样性 -> httpx 并发读取
-> trafilatura 正文抽取 -> Playwright 动态页兜底
-> 正文去重和质量过滤 -> 复用 BGE reranker
```

`deepseek_web_search` 控制原生工具；`web_search_enabled` 控制普通问答是否主动运行本地搜索。Agent 即使本地开关关闭，也可在原生工具失败后调用本地降级通道。

## 7. 数据、安全与运行边界

- `data/rag_book.db`：SQLite、FTS5、评测和轨迹。
- `data/chroma/`：向量索引。
- `data/uploads/`：上传原文件。
- `data/memory/`：会话 JSONL 和摘要 Markdown。
- `data/logs/`：可导出的操作日志。
- `.env`：API Key，不提交 Git。
- `models/`：本地 BGE 模型，不提交 Git。

网页抓取拒绝非公网地址；Agent 工具是固定白名单，不具备 Shell、任意文件读写或任意代码执行能力。当前没有登录、多租户、权限隔离和生产级高可用，部署到公网前必须增加鉴权、反向代理、限流和审计保护。
