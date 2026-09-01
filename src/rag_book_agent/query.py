"""Three-layer, low-latency query router used before retrieval."""

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict


@dataclass
class RouteDecision:
    query: str
    action: str = "rag"
    score: float = 0.0
    layer: int = 2
    cached_answer: str = ""
    reason: str = ""


class QuestionProcessor:
    """Rules/cache -> anchor similarity -> optional LLM handoff."""

    terms = {"发布": "publish upload release distribution", "打包": "build package packaging distribution", "安装": "install package", "依赖": "dependencies requirements", "构建": "build package", "上传": "upload publish", "概念": "concepts terminology glossary", "教程": "tutorial guide", "向量数据库": "vector database vector store"}
    anchors = {"technical_docs": "技术 文档 报错 错误 配置 安装 部署 API 代码", "book_qa": "书籍 章节 内容 解释 什么 是 如何 原文", "evaluation": "评测 测试 指标 召回率 准确率 RAGAS", "web_research": "最新 新闻 实时 今天 互联网 搜索 资料"}
    greetings = re.compile(r"^(你好|您好|嗨|hello|hi|在吗|有人吗)[！!。.?？ ]*$", re.I)
    sensitive = re.compile(r"(密码|私钥|api[ _-]?key|身份证|银行卡|验证码)", re.I)

    def __init__(self, cache=None, llm_router=None, ttl_seconds: int = 900):
        self.cache = cache
        self.llm_router = llm_router
        self.ttl_seconds = ttl_seconds
        self._memory_cache: Dict[str, tuple] = {}
        self.last_decision = RouteDecision("")

    def process(self, question: str) -> str:
        return self.route(question).query

    def route(self, question: str) -> RouteDecision:
        normalized = re.sub(r"\s+", " ", question or "").strip()
        if not normalized:
            return self._remember(RouteDecision("", "fallback", 0.0, 1, reason="empty"))
        if self.greetings.match(normalized):
            return self._remember(RouteDecision(normalized, "greeting", 1.0, 1, "你好，我可以帮你检索和分析知识库内容。", "greeting"))
        if self.sensitive.search(normalized):
            return self._remember(RouteDecision(normalized, "safety", 1.0, 1, "出于安全原因，我不会处理密码、密钥或身份认证信息。", "sensitive"))
        cached = self._get_cache(normalized)
        if cached:
            return self._remember(RouteDecision(normalized, "cache", 1.0, 1, cached, "cache-hit"))
        query = self._expand(normalized)
        scores = {name: self._similarity(normalized, anchor) for name, anchor in self.anchors.items()}
        action, score = max(scores.items(), key=lambda item: item[1])
        if score <= 0.5:
            decision = RouteDecision(query, "fallback", score, 2, reason="low-confidence")
        elif score > 0.82:
            decision = RouteDecision(query, action, score, 2, reason="anchor-high-confidence")
        elif self.llm_router:
            decision = RouteDecision(query, self.llm_router(normalized) or action, score, 3, reason="llm-route")
        else:
            decision = RouteDecision(query, "rag", score, 2, reason="anchor-medium-confidence")
        return self._remember(decision)

    def _expand(self, question: str) -> str:
        additions = [target for source, target in self.terms.items() if source in question or source.lower() in question.lower()]
        return " ".join([question] + additions)

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        a, b = Counter(left.lower()), Counter(right.lower())
        dot = sum(value * b.get(key, 0) for key, value in a.items())
        na = math.sqrt(sum(value * value for value in a.values()))
        nb = math.sqrt(sum(value * value for value in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def _get_cache(self, key: str) -> str:
        item = self._memory_cache.get(key)
        if item and item[0] > time.time():
            return item[1]
        if self.cache:
            try:
                value = self.cache.get("rag:query:" + key)
                return value.decode("utf-8") if isinstance(value, bytes) else str(value or "")
            except Exception:
                pass
        return ""

    def cache_answer(self, question: str, answer: str) -> None:
        key = re.sub(r"\s+", " ", question or "").strip()
        self._memory_cache[key] = (time.time() + self.ttl_seconds, answer)
        if self.cache:
            try:
                self.cache.setex("rag:query:" + key, self.ttl_seconds, answer)
            except Exception:
                pass

    def _remember(self, decision: RouteDecision) -> RouteDecision:
        self.last_decision = decision
        return decision
