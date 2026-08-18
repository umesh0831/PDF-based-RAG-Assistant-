import re
import uuid

from models import Chunk

DEFAULT_TARGET_TOKENS = 350
DEFAULT_SIM_THRESHOLD = 0.62
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _embed_sentence(text: str) -> list[float]:
    import ollama
    return ollama.embeddings(model="qwen3-embedding:0.6b", prompt=text)["embedding"]


def chunk_text(
    text: str,
    source: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    similarity_threshold: float = DEFAULT_SIM_THRESHOLD,
    min_chunk_tokens: int = 40,
) -> list[Chunk]:
    """Semantic chunking — splits on meaning boundaries, not fixed token counts.

    For each adjacent pair of sentences we compute embedding similarity. When
    similarity drops below the threshold (topic shift) and the current buffer
    is large enough, we close the chunk. This avoids cutting sentences mid-thought
    and produces chunks that are semantically coherent for retrieval.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = [sentences[0]]
    buffer_tokens = _approx_tokens(sentences[0])
    prev_embedding = _embed_sentence(sentences[0])
    idx = 0

    for sentence in sentences[1:]:
        sent_embedding = _embed_sentence(sentence)
        similarity = _cosine(prev_embedding, sent_embedding)

        should_break = (
            similarity < similarity_threshold and buffer_tokens >= min_chunk_tokens
        ) or buffer_tokens >= target_tokens

        if should_break:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=" ".join(buffer),
                    source=source,
                    chunk_index=idx,
                )
            )
            idx += 1
            buffer = [sentence]
            buffer_tokens = _approx_tokens(sentence)
        else:
            buffer.append(sentence)
            buffer_tokens += _approx_tokens(sentence)

        prev_embedding = sent_embedding

    if buffer and buffer_tokens >= min_chunk_tokens:
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=" ".join(buffer),
                source=source,
                chunk_index=idx,
            )
        )

    return chunks
