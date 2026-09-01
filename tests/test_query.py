from rag_book_agent.query import QuestionProcessor


def test_query_router_rules_and_cache():
    router = QuestionProcessor()
    assert router.route("你好").action == "greeting"
    router.cache_answer("固定问题", "固定答案")
    result = router.route("固定问题")
    assert result.action == "cache"
    assert result.cached_answer == "固定答案"


def test_query_router_has_anchor_and_fallback_layers():
    router = QuestionProcessor()
    assert router.route("怎么配置 API 和部署代码").layer == 2
    assert router.route("完全不相关的随机输入").action in {"fallback", "rag"}
