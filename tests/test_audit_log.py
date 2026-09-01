from rag_book_agent.audit_log import OperationLog


def test_operation_log_writes_utf8_text(tmp_path):
    log = OperationLog(tmp_path)
    log.write("上传文件", "notes.md")
    data = log.read_bytes().decode("utf-8")
    assert "上传文件" in data
    assert "notes.md" in data
