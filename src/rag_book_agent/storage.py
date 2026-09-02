import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rag_book_agent.models import Chunk, ParsedDocument


class Storage:
    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self.settings = None
        self.connection = sqlite3.connect(str(database_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                file_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                enabled INTEGER NOT NULL DEFAULT 1,
                in_library INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                heading TEXT NOT NULL DEFAULT '',
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                position INTEGER NOT NULL
                ,parent_id INTEGER
                ,chunk_type TEXT NOT NULL DEFAULT 'child'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                heading,
                tokenize='unicode61'
            );

            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                mode TEXT NOT NULL,
                source_ids TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                trace_id INTEGER NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
                rating TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS trace_details (
                trace_id INTEGER PRIMARY KEY REFERENCES traces(id) ON DELETE CASCADE,
                detail_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS golden_questions (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL UNIQUE,
                reference_answer TEXT NOT NULL DEFAULT '',
                expected_chunk_ids TEXT NOT NULL DEFAULT '[]'
            );
            """
        )
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(documents)")}
        if "enabled" not in columns:
            self.connection.execute(
                "ALTER TABLE documents ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1"
            )
        if "in_library" not in columns:
            self.connection.execute(
                "ALTER TABLE documents ADD COLUMN in_library INTEGER NOT NULL DEFAULT 1"
            )
        chunk_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(chunks)")}
        if "parent_id" not in chunk_columns:
            self.connection.execute("ALTER TABLE chunks ADD COLUMN parent_id INTEGER")
        if "chunk_type" not in chunk_columns:
            self.connection.execute("ALTER TABLE chunks ADD COLUMN chunk_type TEXT NOT NULL DEFAULT 'child'")
        self.connection.commit()

    def list_documents(self) -> List[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT d.*, COUNT(c.id) AS chunk_count "
                "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
                "GROUP BY d.id ORDER BY d.indexed_at DESC, d.id DESC"
            ).fetchall()
        )

    def update_document(
        self, document_id: int, enabled: Optional[bool] = None, in_library: Optional[bool] = None
    ) -> bool:
        changes = []
        values = []
        if enabled is not None:
            changes.append("enabled = ?")
            values.append(1 if enabled else 0)
        if in_library is not None:
            changes.append("in_library = ?")
            values.append(1 if in_library else 0)
        if not changes:
            return False
        values.append(document_id)
        cursor = self.connection.execute(
            "UPDATE documents SET " + ", ".join(changes) + " WHERE id = ?", values
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def delete_document(self, document_id: int) -> bool:
        self.connection.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id = ?)",
            (document_id,),
        )
        cursor = self.connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        self.connection.commit()
        return cursor.rowcount > 0

    def document_is_current(self, path: str, content_hash: str) -> bool:
        row = self.connection.execute(
            "SELECT content_hash FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return row is not None and row["content_hash"] == content_hash

    def replace_document(self, document: ParsedDocument, chunks: Iterable[Chunk]) -> int:
        cursor = self.connection.cursor()
        old_row = cursor.execute(
            "SELECT id FROM documents WHERE path = ?", (document.path,)
        ).fetchone()
        if old_row is not None:
            old_id = old_row["id"]
            cursor.execute(
                "DELETE FROM chunks_fts WHERE rowid IN "
                "(SELECT id FROM chunks WHERE document_id = ?)",
                (old_id,),
            )
            cursor.execute("DELETE FROM documents WHERE id = ?", (old_id,))

        cursor.execute(
            "INSERT INTO documents(path, title, file_type, content_hash, page_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                document.path,
                document.title,
                document.file_type,
                document.content_hash,
                len(document.pages),
            ),
        )
        document_id = int(cursor.lastrowid)

        pending = list(chunks)
        parent_ids = {}
        for chunk in pending:
            cursor.execute(
                "INSERT INTO chunks(document_id, text, heading, page_start, page_end, position, chunk_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document_id,
                    chunk.text,
                    chunk.heading,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.position,
                    chunk.chunk_type,
                ),
            )
            chunk_id = int(cursor.lastrowid)
            if chunk.chunk_type == "parent":
                parent_ids[chunk.position] = chunk_id
            else:
                parent_id = parent_ids.get(chunk.parent_id)
                if parent_id:
                    cursor.execute("UPDATE chunks SET parent_id = ? WHERE id = ?", (parent_id, chunk_id))
                cursor.execute("INSERT INTO chunks_fts(rowid, text, heading) VALUES (?, ?, ?)", (chunk_id, chunk.text, chunk.heading))

        self.connection.commit()
        return document_id

    def search_fts(
        self, query: str, limit: int, document_ids: Optional[List[int]] = None
    ) -> List[sqlite3.Row]:
        sql = """
            SELECT c.*, d.title AS document_title, d.path AS document_path,
                   bm25(chunks_fts, 1.0, 2.0) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ? AND d.enabled = 1 AND d.in_library = 1
            ORDER BY rank
            LIMIT ?
        """
        parameters = [query]
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            sql = sql.replace(
                "WHERE chunks_fts MATCH ?",
                "WHERE chunks_fts MATCH ? AND c.document_id IN (%s)" % placeholders,
            )
            parameters.extend(document_ids)
        parameters.append(limit)
        try:
            return list(self.connection.execute(sql, parameters).fetchall())
        except sqlite3.OperationalError:
            return []

    def list_chunks(
        self, document_ids: Optional[List[int]] = None, active_only: bool = False
    ) -> List[sqlite3.Row]:
        active_clause = "WHERE d.enabled = 1 AND d.in_library = 1" if active_only else "WHERE 1=1"
        sql = """
            SELECT c.*, d.title AS document_title, d.path AS document_path
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {active} AND c.chunk_type = 'child' ORDER BY c.id
        """.format(active=active_clause)
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            condition = "c.document_id IN (%s)" % placeholders
            if active_only:
                sql = sql.replace(
                    "WHERE d.enabled = 1 AND d.in_library = 1",
                    "WHERE d.enabled = 1 AND d.in_library = 1 AND " + condition,
                )
            else:
                sql = sql.replace("WHERE 1=1", "WHERE " + condition)
            return list(self.connection.execute(sql, document_ids).fetchall())
        return list(self.connection.execute(sql).fetchall())

    def get_chunk(self, chunk_id: int) -> Optional[sqlite3.Row]:
        sql = """
            SELECT c.*, d.title AS document_title, d.path AS document_path
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
        """
        return self.connection.execute(sql, (chunk_id,)).fetchone()

    def stats(self) -> Dict[str, int]:
        documents = self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        questions = self.connection.execute("SELECT COUNT(*) FROM golden_questions").fetchone()[0]
        return {"documents": documents, "chunks": chunks, "questions": questions}

    def add_trace(
        self,
        question: str,
        answer: str,
        mode: str,
        source_ids: List[int],
        elapsed_ms: int,
    ) -> int:
        cursor = self.connection.execute(
            "INSERT INTO traces(question, answer, mode, source_ids, elapsed_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (question, answer, mode, json.dumps(source_ids), elapsed_ms),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def add_feedback(self, trace_id: int, rating: str, note: str = "") -> None:
        self.connection.execute(
            "INSERT INTO feedback(trace_id, rating, note) VALUES (?, ?, ?)",
            (trace_id, rating, note),
        )
        self.connection.commit()

    def add_trace_detail(self, trace_id: int, detail: Dict) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO trace_details(trace_id, detail_json) VALUES (?, ?)",
            (trace_id, json.dumps(detail, ensure_ascii=False, default=str)),
        )
        self.connection.commit()

    def list_trace_details(self, limit: int = 80) -> List[Dict]:
        rows = self.connection.execute(
            "SELECT traces.id, traces.question, traces.mode, traces.elapsed_ms, traces.created_at, "
            "trace_details.detail_json FROM traces LEFT JOIN trace_details "
            "ON traces.id = trace_details.trace_id ORDER BY traces.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        output = []
        for row in rows:
            try:
                detail = json.loads(row["detail_json"] or "{}")
            except json.JSONDecodeError:
                detail = {}
            output.append({
                "id": row["id"], "question": row["question"], "mode": row["mode"],
                "elapsed_ms": row["elapsed_ms"], "created_at": row["created_at"], "detail": detail,
            })
        return output

    def add_golden_question(
        self, question: str, expected_chunk_ids: List[int], reference_answer: str = ""
    ) -> None:
        self.connection.execute(
            "INSERT INTO golden_questions(question, reference_answer, expected_chunk_ids) "
            "VALUES (?, ?, ?) ON CONFLICT(question) DO UPDATE SET "
            "reference_answer=excluded.reference_answer, "
            "expected_chunk_ids=excluded.expected_chunk_ids",
            (question, reference_answer, json.dumps(expected_chunk_ids)),
        )
        self.connection.commit()

    def list_golden_questions(self) -> List[sqlite3.Row]:
        return list(
            self.connection.execute("SELECT * FROM golden_questions ORDER BY id").fetchall()
        )
