# 两个高收益准确性优化

当前版本已经清空旧文档，保留 Web 前端。下一步优先做下面两个改动，它们对答案准确性提升明显，且不需要重写现有服务接口。

## 向量存储选型

向量库采用 **ChromaDB**：嵌入式、无需 Docker 或常驻服务、支持本地持久化和元数据过滤，适合本机轻量部署。SQLite 继续保存文档元数据与 FTS5 稀疏索引。

## 1. 真实语义向量模型

**建议：** `BAAI/bge-small-zh-v1.5`（中文/中英混合，CPU 友好，MIT），保留当前 HashEmbedding 作为无模型降级。

**改动位置：** `retrieval/engine.py` 增加 EmbeddingProvider；索引阶段缓存向量，查询阶段只编码一次；配置增加模型路径和离线开关。

**收益：** 能识别“怎么打包”与“如何构建 Python distribution”这类不同表述，显著提高语义召回，通常比哈希特征更适合跨语言和同义问题。

**验收：** 在固定 golden set 上比较 BM25、HashEmbedding、BGE 和混合检索的 Recall@10/MRR@10，并记录 CPU 延迟与内存。

## 2. Cross-encoder 重排

**建议：** `BAAI/bge-reranker-base`（Apache-2.0）；低配机器可配置为禁用并回退规则重排。

**改动位置：** 将当前 `LightweightReranker` 抽象为 Reranker 接口，对融合后的 top-30 候选批量计算 `(question, chunk)` 相关性，再选 top-6 上下文。

**收益：** 重排模型直接判断问题和段落的匹配关系，能减少“同领域但答非所问”的片段进入上下文，通常比单纯扩大召回更稳定地提升答案忠实度和上下文精确度。

**验收：** 对比规则重排和 cross-encoder 的 MRR、context precision、citation precision；模型不可用时必须有 `degraded=true` 日志和可用回退。

## 为什么先做这两个

它们分别解决“找不到正确证据”和“找到了但排序不对”两个最常见的 RAG 质量问题。OCR、RAPTOR、GraphRAG、本地生成模型和权限体系都值得做，但需要更大改动或更多数据，放在这两个模块之后更合适。

## 当前边界

本项目当前仍可使用 DeepSeek API 生成；这两个优化只改变检索和重排，不会改变 Web API、引用格式和前端操作。模型权重必须由用户显式下载并固定版本，运行时支持离线模式。
