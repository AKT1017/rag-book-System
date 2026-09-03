from rag_book_agent.config import Settings
from rag_book_agent.web_search import WebSearch


def test_canonical_url_unwraps_duckduckgo_and_removes_tracking():
    url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc%3Futm_source%3Dx%26page%3D2"
    assert WebSearch._canonical_url(url) == "https://example.com/doc?page=2"


def test_domain_diversity_and_duplicate_urls():
    settings = Settings(web_search_per_domain_limit=1)
    search = WebSearch(settings)
    rows = [
        {"url": "https://a.example/one", "title": "one"},
        {"url": "https://a.example/two", "title": "two"},
        {"url": "https://b.example/one", "title": "three"},
        {"url": "https://b.example/one#fragment", "title": "duplicate"},
    ]
    selected = search._normalize_and_diversify(rows, 10)
    assert [item["title"] for item in selected] == ["one", "three"]


def test_rank_deduplicates_page_bodies():
    search = WebSearch(Settings())
    rows = [
        {"url": "https://a.example", "title": "RAG", "text": "RAG retrieval evidence " * 8},
        {"url": "https://b.example", "title": "copy", "text": "RAG retrieval evidence " * 8},
    ]
    ranked = search._rank("RAG retrieval", rows)
    assert len(ranked) == 1
    assert ranked[0]["score"] > 0


def test_unreadable_or_tiny_text_is_rejected():
    search = WebSearch(Settings())
    assert not search._is_usable_text("short", minimum=80)
    assert not search._is_usable_text("\ufffd" * 100, minimum=80)
