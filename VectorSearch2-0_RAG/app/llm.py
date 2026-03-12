"""
LLM — Gemini via Vertex AI
===========================
Generates a grounded answer from retrieved context chunks.
"""
from __future__ import annotations

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel


def init_vertex(project_id: str, location: str) -> None:
    vertexai.init(project=project_id, location=location)


def generate_answer(
    model: GenerativeModel,
    question: str,
    context: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    prompt = f"""You are a helpful assistant.
Answer the QUESTION using ONLY the information in CONTEXT below.
If the context does not contain the answer, say "I don't have enough information to answer that."
Cite sources with bracket numbers like [1] or [2] when you use facts from them.

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
