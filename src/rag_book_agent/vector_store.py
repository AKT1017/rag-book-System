"""ChromaDB persistence with an optional small local sentence-transformer."""

from typing import Iterable, List

from rag_book_agent.config import Settings
from rag_book_agent.models import Chunk


class EmbeddingProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None

    def encode(self, texts: List[str], query: bool = False) -> List[List[float]]:
        if not self.settings.embedding_model:
            raise RuntimeError("No embedding model configured")
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.settings.resolve_model_path(self.settings.embedding_model),
                local_files_only=self.settings.embedding_local_files_only,
            )
        prefix = "为这个句子生成表示以用于检索相关文章：" if query else ""
        values = self.model.encode(
            [prefix + text for text in texts], normalize_embeddings=True, show_progress_bar=False
        )
        return values.tolist()


class ChromaVectorStore:
    def __init__(self, settings: Settings):
        import chromadb

        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )
        self.embedding = EmbeddingProvider(settings)

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        if not chunks:
            return
        embeddings = self.embedding.encode([chunk.text for chunk in chunks])
        self.collection.upsert(
            ids=[str(chunk.id) for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"document_id": int(chunk.document_id)} for chunk in chunks],
        )

    def search(self, question: str, limit: int) -> List[tuple]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=self.embedding.encode([question], query=True),
            n_results=min(limit, self.collection.count()),
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [
            (int(chunk_id), 1.0 - float(distance)) for chunk_id, distance in zip(ids, distances)
        ]

    def delete(self, chunk_ids: List[int]) -> None:
        if chunk_ids:
            self.collection.delete(ids=[str(chunk_id) for chunk_id in chunk_ids])
