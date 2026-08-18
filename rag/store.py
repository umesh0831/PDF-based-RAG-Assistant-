from dataclasses import dataclass, field
from typing import List

import chromadb
import ollama
from chromadb import EmbeddingFunction, Embeddings
from rank_bm25 import BM25Okapi

from models import Chunk, RetrievedContext

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
_RRF_K = 60


class OllamaEmbedder(EmbeddingFunction):
    """Embeds text locally using qwen3-embedding:0.6b served by Ollama.

    Chosen over cloud-hosted embeddings so the full pipeline stays local —
    a hard requirement for any enterprise document Q&A use case.
    """

    def __call__(self, input: List[str]) -> Embeddings:
        return [ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)["embedding"] for text in input]


@dataclass
class HybridStore:
    collection: chromadb.Collection
    bm25: BM25Okapi | None = None
    corpus: list[str] = field(default_factory=list)


def get_fresh_store(client: chromadb.Client, collection_name: str = "rag_docs") -> HybridStore:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        embedding_function=OllamaEmbedder(),
        metadata={"hnsw:space": "cosine"},
    )
    return HybridStore(collection=collection)


def add_chunks(store: HybridStore, chunks: List[Chunk]) -> None:
    store.collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )
    store.corpus = [c.text for c in chunks]
    tokenized = [doc.lower().split() for doc in store.corpus]
    store.bm25 = BM25Okapi(tokenized)


def _rrf_merge(dense: list[str], sparse: list[str], top_n: int) -> list[str]:
    scores: dict[str, float] = {}
    for rank, doc in enumerate(dense):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, doc in enumerate(sparse):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:top_n]


def retrieve(store: HybridStore, question: str, n_results: int = 5) -> RetrievedContext:
    """Hybrid retrieval: dense (ChromaDB cosine) + sparse (BM25), merged with RRF.

    Each retriever fetches more candidates than needed; RRF combines the ranked
    lists so chunks that rank highly in both get boosted, giving better coverage
    than either retriever alone.
    """
    n_candidates = min(n_results * 3, len(store.corpus))

    # Dense: embedding similarity via ChromaDB
    dense_docs = store.collection.query(query_texts=[question], n_results=n_candidates)["documents"][0]

    # Sparse: BM25 keyword scoring
    token_query = question.lower().split()
    bm25_scores = store.bm25.get_scores(token_query)
    sparse_docs = [
        store.corpus[i]
        for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_candidates]
    ]

    return RetrievedContext(question=question, chunks=_rrf_merge(dense_docs, sparse_docs, top_n=n_results))
