from rag_book_agent.config import Settings
from rag_book_agent.generation.answerer import AnswerGenerator


def test_responses_parser_extracts_native_calls_and_citations():
    data = {
        "status": "completed",
        "output": [
            {"type": "web_search_call", "status": "completed",
             "action": {"type": "search"}},
            {"type": "message", "content": [{
                "type": "output_text",
                "text": "结论 [官方文档](https://example.com/docs#ws_call_id=abc)",
                "annotations": [{"type": "url_citation", "url": "https://example.com/docs",
                                 "title": "官方文档"}],
            }]},
        ],
    }
    text, results, trace = AnswerGenerator._parse_responses_data(data)
    assert text.startswith("结论")
    assert results == [{
        "title": "官方文档", "url": "https://example.com/docs", "text": "",
        "provider": "deepseek-native", "fetch_status": "server-search",
    }]
    assert trace["tool_calls"] == 1
    assert trace["completed_calls"] == 1


def test_responses_parser_accepts_nested_url_citation():
    data = {"output": [{"type": "message", "content": [{
        "type": "output_text", "text": "answer",
        "annotations": [{"url_citation": {"url": "https://example.org", "title": "source"}}],
    }]}]}
    _, results, _ = AnswerGenerator._parse_responses_data(data)
    assert results[0]["url"] == "https://example.org"


def test_native_web_settings_have_bounded_tool_calls():
    settings = Settings()
    assert 1 <= settings.deepseek_web_max_tool_calls <= 10
