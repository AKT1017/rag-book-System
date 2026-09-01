"""Small append-only operation log for the local web application."""

from datetime import datetime
from pathlib import Path
from threading import Lock


class OperationLog:
    _lock = Lock()

    def __init__(self, project_dir: Path):
        self.path = project_dir / "data" / "logs" / "operations.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, operation: str, detail: str = "") -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = "%s | %-18s | %s\n" % (timestamp, operation, detail[:1000])
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def read_bytes(self) -> bytes:
        if not self.path.exists():
            return "暂无操作日志。\n".encode("utf-8")
        return self.path.read_bytes()
