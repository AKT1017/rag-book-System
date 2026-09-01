# rag-book-System
这是一个面向中文书籍与专业资料的本地化 RAG 智能问答系统，采用 Python 开发，支持 PDF、Markdown、TXT 等格式文档导入，并通过 MarkItDown、PyMuPDF 等成熟工具完成内容解析。系统采用父子分块策略，以子分块进行检索、父分块提供上下文，结合 BM25 稀疏检索、BGE 向量检索、RRF 融合和 BGE 重排模型，提高知识召回与答案准确率。
