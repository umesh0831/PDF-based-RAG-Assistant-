from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A semantically coherent chunk of source text.

    Validated at the parser→chunker→store boundary so malformed data
    fails loud at the boundary rather than silently downstream.
    """

    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)


class RetrievedContext(BaseModel):
    """Top-k chunks returned by the vector store for a given question."""

    question: str = Field(..., min_length=1)
    chunks: list[str]


class QueryResult(BaseModel):
    """End-to-end result of a single Q&A round."""

    question: str
    context_chunks: list[str]
    answer: str
