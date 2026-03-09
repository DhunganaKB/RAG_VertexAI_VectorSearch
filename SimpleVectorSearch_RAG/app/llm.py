from __future__ import annotations

from dataclasses import dataclass

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel


@dataclass(frozen=True)
class LlmConfig:
    project_id: str
    location: str
    model: str
    max_output_tokens: int
    temperature: float


def init_vertex(project_id: str, location: str) -> None:
    vertexai.init(project=project_id, location=location)


def generate_grounded_answer(
    model: GenerativeModel,
    question: str,
    context: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    prompt = f"""You are a helpful assistant.
Answer the QUESTION using only the CONTEXT.
If the context does not contain the answer, say you don't know.

When you use a fact from a source, cite it with bracket numbers like [1] or [2].

CONTEXT:
{context}

QUESTION:
{question}
"""

    resp = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return (resp.text or "").strip()

