from pathlib import Path

import ollama
import yaml

from models import QueryResult, RetrievedContext

LLM_MODEL = "gpt-oss:20b"

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "prompts.yaml"

def _load_prompts() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["prompts"]

_prompts = _load_prompts()


def generate_answer(
    context: RetrievedContext,
    model: str = LLM_MODEL,
) -> QueryResult:
    """Generate a grounded answer using gpt-oss:20b served locally via Ollama.

    Open-source, runs fully local, no API keys, no data leaves the machine.
    """
    joined_context = "\n\n---\n\n".join(context.chunks)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _prompts["system"]},
            {"role": "user", "content": _prompts["user_template"].format(
                context=joined_context,
                question=context.question,
            )},
        ],
    )
    return QueryResult(
        question=context.question,
        context_chunks=context.chunks,
        answer=response["message"]["content"],
    )
