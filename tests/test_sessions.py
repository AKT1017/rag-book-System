from rag_book_agent.memory import ConversationMemory


def test_sessions_can_be_listed_and_restored(tmp_path):
    memory = ConversationMemory(tmp_path, "session-one")
    memory.add("第一个问题", "第一个答案")
    items = ConversationMemory.list_sessions(tmp_path)
    assert items[0]["id"] == "session-one"
    assert ConversationMemory(tmp_path, "session-one").turns()[0]["answer"] == "第一个答案"
