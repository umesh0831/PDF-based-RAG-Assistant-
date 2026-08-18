from flashrank import Ranker, RerankRequest

from models import RetrievedContext

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-6-v2", cache_dir="/tmp/flashrank")
    return _ranker


def rerank(context: RetrievedContext, top_n: int = 2) -> RetrievedContext:
    """Cross-encoder re-ranking of hybrid retrieval candidates.

    The bi-encoder used at retrieval time (qwen3-embedding) embeds query and
    chunks independently, which is fast but imprecise. A cross-encoder scores
    each (query, chunk) pair jointly — much more accurate, but too slow to run
    on the full corpus. Running it only on the small candidate set from hybrid
    retrieval gives the best of both: broad recall from retrieval, precise
    ranking from the cross-encoder.
    """
    passages = [{"id": i, "text": chunk} for i, chunk in enumerate(context.chunks)]
    results = _get_ranker().rerank(RerankRequest(query=context.question, passages=passages))
    return RetrievedContext(
        question=context.question,
        chunks=[r["text"] for r in results[:top_n]],
    )
