"""
Tests for hybrid search: BM25 indexing, RRF merging, and retrieve().
Ollama and ChromaDB are mocked — these tests run without any models pulled.
"""
import sys
import types
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

# ── Stub out heavy dependencies before importing store ────────────────────────

# ollama stub
ollama_stub = types.ModuleType("ollama")
ollama_stub.embeddings = MagicMock(return_value={"embedding": [0.1, 0.2, 0.3]})
sys.modules["ollama"] = ollama_stub

# chromadb stub
chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.Client = MagicMock
chromadb_stub.Collection = MagicMock
EmbFunc = type("EmbeddingFunction", (), {"__call__": lambda s, x: x})
Embeddings = list
chromadb_stub.EmbeddingFunction = EmbFunc
chromadb_stub.Embeddings = Embeddings
sys.modules["chromadb"] = chromadb_stub

sys.path.insert(0, "/tmp/rag-assistant")

from models import Chunk
from rag.store import HybridStore, _rrf_merge, add_chunks, retrieve


def _make_chunk(text: str, idx: int) -> Chunk:
    import uuid
    return Chunk(id=str(uuid.uuid4()), text=text, source="test.pdf", chunk_index=idx)


class TestRRFMerge(unittest.TestCase):
    def test_chunk_in_both_lists_ranks_above_chunk_in_one(self):
        # "shared" appears #1 in dense and #1 in sparse → highest combined score
        # "dense_only" appears #2 in dense, not in sparse
        # "sparse_only" appears #2 in sparse, not in dense
        dense  = ["shared", "dense_only"]
        sparse = ["shared", "sparse_only"]
        result = _rrf_merge(dense, sparse, top_n=3)
        self.assertEqual(result[0], "shared", "chunk in both lists should rank first")

    def test_top_n_is_respected(self):
        dense  = ["a", "b", "c", "d"]
        sparse = ["b", "a", "d", "c"]
        result = _rrf_merge(dense, sparse, top_n=2)
        self.assertEqual(len(result), 2)

    def test_rrf_score_is_additive(self):
        # rank 0 in dense: 1/(60+1) ≈ 0.01639
        # rank 0 in sparse: 1/(60+1) ≈ 0.01639
        # combined ≈ 0.03279 — higher than rank 0 in only one list
        dense  = ["x", "y"]
        sparse = ["x", "z"]
        result = _rrf_merge(dense, sparse, top_n=3)
        self.assertEqual(result[0], "x")

    def test_empty_inputs(self):
        self.assertEqual(_rrf_merge([], [], top_n=2), [])

    def test_top_n_larger_than_combined_unique_chunks(self):
        result = _rrf_merge(["a"], ["a"], top_n=5)
        self.assertEqual(result, ["a"])


class TestBM25Indexing(unittest.TestCase):
    def _make_store(self, texts):
        mock_collection = MagicMock()
        store = HybridStore(collection=mock_collection)
        chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
        add_chunks(store, chunks)
        return store, chunks

    def test_bm25_index_is_built_after_add_chunks(self):
        store, _ = self._make_store(["the cat sat on the mat", "dogs like to run"])
        self.assertIsNotNone(store.bm25)

    def test_corpus_is_stored(self):
        texts = ["hello world", "foo bar baz"]
        store, _ = self._make_store(texts)
        self.assertEqual(store.corpus, texts)

    def test_bm25_ranks_exact_keyword_match_first(self):
        texts = [
            "The CAGR of the sector was 12% over five years",
            "Revenue grew steadily due to market expansion",
            "Operational costs increased in the second quarter",
        ]
        store, _ = self._make_store(texts)
        scores = store.bm25.get_scores(["cagr"])
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        self.assertEqual(store.corpus[best_idx], texts[0])


class TestRetrieve(unittest.TestCase):
    def test_retrieve_returns_retrieved_context(self):
        from models import RetrievedContext

        texts = [
            "The CAGR of the sector was 12% over five years",
            "Revenue grew due to market expansion",
            "Costs increased in Q2",
        ]
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[texts[0], texts[1]]]}

        store = HybridStore(collection=mock_collection)
        chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
        add_chunks(store, chunks)

        result = retrieve(store, "what is the CAGR?", n_results=2)

        self.assertIsInstance(result, RetrievedContext)
        self.assertEqual(result.question, "what is the CAGR?")
        self.assertEqual(len(result.chunks), 2)

    def test_retrieve_boosts_exact_keyword_match(self):
        # Dense returns only one generic chunk; BM25 surfaces the CAGR chunk at
        # rank 0. With only one dense result there is no dual-list bonus to beat,
        # so the BM25-only keyword match wins the second slot in top-2.
        texts = [
            "The CAGR of the sector was 12% over five years",  # BM25 winner
            "Revenue grew due to market expansion",
            "Costs increased in Q2",
        ]
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[texts[1]]]}

        store = HybridStore(collection=mock_collection)
        chunks = [_make_chunk(t, i) for i, t in enumerate(texts)]
        add_chunks(store, chunks)

        result = retrieve(store, "CAGR growth", n_results=2)

        # texts[1] wins slot 1 (dense+sparse); CAGR chunk wins slot 2 (sparse rank 0)
        self.assertIn(texts[0], result.chunks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
