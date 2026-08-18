# PDF Q&A Assistant — Fully Local RAG

A lightweight, fully local Retrieval-Augmented Generation (RAG) system for PDF document Q&A. Built end-to-end with open-source tools — no API keys, no cloud calls, your documents never leave your machine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Ollama](https://img.shields.io/badge/Runtime-Ollama-black?style=flat-square)
![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Why fully local?

Most RAG demos call OpenAI, Gemini, or Anthropic for embeddings or generation. That's fine for tutorials — not fine for enterprise document Q&A where the documents themselves are sensitive (contracts, financial statements, medical records, logistics manifests). I built this with the constraint that **the entire pipeline runs on a single machine** with no outbound network calls once models are pulled. Data governance is a design driver here, not an afterthought.

---

## Architecture

```
                INDEXING PIPELINE                              QUERY PIPELINE
                (runs once per PDF)                            (runs per question)

                ┌────────────────┐                             ┌────────────────┐
                │  PDF Document  │                             │ User Question  │
                └───────┬────────┘                             └───────┬────────┘
                        │                                              │
                        ▼                                              │
            ┌───────────────────────┐                                  │
            │ 1 · MarkItDown Parser │                                  │
            │   PDF → Markdown      │                                  │
            │   preserves structure │                                  │
            └───────────┬───────────┘                                  │
                        ▼                                              │
            ┌───────────────────────┐                                  │
            │ 2 · Semantic Chunker  │                                  │
            │   splits on meaning,  │                                  │
            │   not fixed size      │                                  │
            └───────────┬───────────┘                                  │
                        ▼                                              │
            ┌───────────────────────┐                                  │
            │ 3 · Pydantic Models   │                                  │
            │   validate at every   │                                  │
            │   stage boundary      │                                  │
            └───────────┬───────────┘                                  │
                        ▼                                              ▼
            ┌──────────────────────────────────────────────────────────┐
            │  4 · qwen3-embedding:0.6b  (served via Ollama, local)    │
            │      same model embeds both chunks and queries           │
            └──────────────────────────┬───────────────────────────────┘
                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  5 · Hybrid Retrieval                                    │
            │      ChromaDB cosine (dense) + BM25 (sparse)             │
            │      merged with Reciprocal Rank Fusion · top-5          │
            └──────────────────────────┬───────────────────────────────┘
                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  6 · Cross-encoder Re-ranker                             │
            │      ms-marco-MiniLM-L-6-v2 (flashrank, local ONNX)      │
            │      jointly scores each (query, chunk) pair · top-2     │
            └──────────────────────────┬───────────────────────────────┘
                                       ▼
            ┌──────────────────────────────────────────────────────────┐
            │  7 · gpt-oss:20b  (served via Ollama, local)             │
            │      generates a grounded answer from retrieved context  │
            └──────────────────────────────────────────────────────────┘
```

---

## Design decisions and why

| Component | Choice | Reasoning |
|---|---|---|
| **PDF parser** | `MarkItDown` | Most projects use `PyPDF` / `pdfplumber` — these give raw text and lose document structure (headings, tables, lists). MarkItDown converts to Markdown, preserving structure that helps the LLM ground answers. |
| **Chunker** | Semantic chunking | Fixed-size chunking is easy but constantly cuts sentences mid-thought. Semantic chunking measures embedding similarity between adjacent sentences and breaks on topic shifts — chunks remain coherent. |
| **Validation** | Pydantic models at every stage boundary | In a multi-stage pipeline, silent type errors propagate downstream and corrupt retrieval. Pydantic fails loud at the boundary where the error actually originates. |
| **Embeddings** | `qwen3-embedding:0.6b` via Ollama | Top-tier on the MTEB leaderboard for its size class, and small enough (0.6B parameters) to run comfortably on consumer hardware. Same model embeds both indexed chunks and queries for vector-space consistency. |
| **Vector store** | ChromaDB (dense) + BM25 (sparse), merged with RRF | Dense retrieval misses exact keywords (acronyms, codes, names). BM25 catches them but has no semantic understanding. Reciprocal Rank Fusion combines both ranked lists — chunks that score highly in both get boosted. Top-5 candidates passed to re-ranker. |
| **Re-ranker** | `ms-marco-MiniLM-L-6-v2` via flashrank (local ONNX) | Bi-encoders embed query and chunks independently — fast, but imprecise. A cross-encoder scores each (query, chunk) pair jointly, catching relevance nuances the bi-encoder misses. Running it only on the top-5 RRF candidates keeps latency acceptable. Final top-2 sent to the LLM. |
| **LLM** | `gpt-oss:20b` via Ollama | Open-source, runs locally on Apple Silicon / consumer GPUs, surprisingly competitive with closed-source models for grounded Q&A where retrieval already constrains the answer. |
| **UI** | Streamlit | Fastest way to ship a chat-style interface in Python with file upload, session state, and source-context expanders out of the box. |

---

## What I learned

- **In RAG, retrieval quality dominates generation quality.** You can swap one LLM for another and barely move the needle on answer quality. Tune chunking and the system improves dramatically. Most iteration time went into chunking, not prompting.
- **Same embedding model for chunks and queries is non-negotiable.** Cross-model similarity is meaningless — embeddings only compare within the same vector space.
- **Fail loud at boundaries.** Adding Pydantic validation between parser → chunker → store caught bugs that would otherwise have silently produced wrong answers.

---

## Setup

**Prerequisites**

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

**Install models**

```bash
ollama pull qwen3-embedding:0.6b
ollama pull gpt-oss:20b
```

> `gpt-oss:20b` is ~13 GB and needs ~16 GB RAM. On lower-memory machines, swap to a smaller model (e.g. `llama3.1:8b`) by editing `LLM_MODEL` in `rag/llm.py`.

**Install Python dependencies**

```bash
pip install -r requirements.txt
```

**Run**

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a PDF, and ask questions.

---

## Project structure

```
.
├── app.py                  # Streamlit UI and orchestration
├── models.py               # Pydantic models — Chunk, RetrievedContext, QueryResult
├── config/
│   └── prompts.yaml        # Versioned prompt config (system prompt + user template)
├── rag/
│   ├── parser.py           # MarkItDown PDF → Markdown
│   ├── chunker.py          # Semantic chunking on embedding-similarity breaks
│   ├── store.py            # HybridStore: ChromaDB (dense) + BM25 (sparse) + RRF merge
│   ├── reranker.py         # Cross-encoder re-ranking via flashrank (ms-marco-MiniLM)
│   └── llm.py              # gpt-oss:20b grounded generation
├── Dockerfile              # Container image for self-hosted deployment
├── render.yaml             # Render.com service configuration
└── requirements.txt
```

---

## Limitations

- Scanned / image-only PDFs are not supported (no OCR upstream of the parser)
- In-memory ChromaDB — indexed documents reset on app restart

---

## License

MIT
