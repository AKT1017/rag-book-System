"""Small persistent conversation memory: recent turns plus a rolling summary."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List


class ConversationMemory:
    def __init__(self, data_dir: Path, session_id: str):
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)[:64] or "default"
        self.directory = data_dir / "memory"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.turn_path = self.directory / (safe_id + ".jsonl")
        self.summary_path = self.directory / (safe_id + ".md")

    def context(self, recent_limit: int = 6) -> str:
        turns = self._read_turns()
        summary = (
            self.summary_path.read_text(encoding="utf-8").strip()
            if self.summary_path.exists()
            else ""
        )
        recent = turns[-recent_limit:]
        parts = []
        if summary:
            parts.append("历史摘要：\n" + summary)
        if recent:
            lines = ["近期对话："]
            for turn in recent:
                lines.append("用户：" + turn["question"])
                lines.append("助手：" + turn["answer"][:800])
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def add(self, question: str, answer: str, sources=None, web_sources=None) -> None:
        row = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "web_sources": web_sources or [],
        }
        with self.turn_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._compress_if_needed()

    def _read_turns(self) -> List[dict]:
        if not self.turn_path.exists():
            return []
        rows = []
        for line in self.turn_path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def turns(self) -> List[dict]:
        """Return persisted turns for restoring a browser conversation."""
        return self._read_turns()

    @staticmethod
    def list_sessions(data_dir: Path) -> List[dict]:
        directory = data_dir / "memory"
        if not directory.exists():
            return []
        sessions = []
        for path in directory.glob("*.jsonl"):
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if rows:
                first = rows[0]
                sessions.append({
                    "id": path.stem,
                    "title": first.get("question", "新对话")[:36],
                    "updated_at": rows[-1].get("at", ""),
                    "turn_count": len(rows),
                })
        return sorted(sessions, key=lambda item: item["updated_at"], reverse=True)

    def _compress_if_needed(self) -> None:
        turns = self._read_turns()
        if len(turns) <= 10:
            return
        archived = turns[:-6]
        current = (
            self.summary_path.read_text(encoding="utf-8").strip()
            if self.summary_path.exists()
            else ""
        )
        lines = [current] if current else []
        for turn in archived:
            answer = re.sub(r"\s+", " ", turn["answer"]).strip()[:320]
            lines.append("- 问：%s\n  答：%s" % (turn["question"], answer))
        summary = "\n".join(lines)[-5000:]
        self.summary_path.write_text(summary + "\n", encoding="utf-8")
        with self.turn_path.open("w", encoding="utf-8") as handle:
            for turn in turns[-6:]:
                handle.write(json.dumps(turn, ensure_ascii=False) + "\n")
