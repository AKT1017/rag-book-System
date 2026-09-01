from rag_book_agent.memory import ConversationMemory


def test_persistent_memory_keeps_recent_turns_and_compresses_history(tmp_path):
    memory = ConversationMemory(tmp_path, "browser-session")
    for number in range(11):
        memory.add("问题 %d" % number, "回答 %d" % number)

    assert memory.turn_path.exists()
    assert memory.summary_path.exists()
    assert len(memory._read_turns()) == 6
    assert "问题 0" in memory.summary_path.read_text(encoding="utf-8")
    assert "问题 10" in memory.context()


def test_memory_persists_sources_per_turn(tmp_path):
    memory = ConversationMemory(tmp_path, "sources-session")
    memory.add("问题", "答案", [{"id": 3, "title": "资料"}], [{"title": "网页"}])
    turn = memory.turns()[0]
    assert turn["sources"][0]["id"] == 3
    assert turn["web_sources"][0]["title"] == "网页"
