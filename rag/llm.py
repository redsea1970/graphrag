"""LLM layer for RAG demo.

Demo uses MockLLM: deterministic, offline, no API keys.
Real providers (Ollama / OpenAI) are stubs with clear TODO,
so the portfolio card can honestly say "plug-and-play ready".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Fact:
    """One retrieved fact from the knowledge graph."""
    subject: str
    relation: str
    object: str

    def as_text(self) -> str:
        return f"{self.subject} --{self.relation}--> {self.object}"


class LLMClient(Protocol):
    """Minimal interface any LLM provider must implement."""

    def answer(self, question: str, facts: Sequence[Fact]) -> str:
        ...


class MockLLM:
    """Deterministic offline LLM for demo & tests.

    Builds an answer by listing retrieved facts in natural language.
    No network, no keys, reproducible output.
    """

    def answer(self, question: str, facts: Sequence[Fact]) -> str:
        if not facts:
            return (
                "В графе не найдено фактов по этому вопросу. "
                "Попробуйте упомянуть известную сущность (персона, организация, локация)."
            )
        lines = [f"По вопросу «{question}» в графе найдено {len(facts)} факт(ов):"]
        for f in facts:
            lines.append(f"- {f.subject} — {f.relation} — {f.object}")
        lines.append("(Ответ сгенерирован MockLLM — детерминированно, без внешней LLM.)")
        return "\n".join(lines)


class OllamaLLM:
    """Stub for local Ollama provider. Not used in demo."""

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host

    def answer(self, question: str, facts: Sequence[Fact]) -> str:  # pragma: no cover
        raise NotImplementedError(
            "OllamaLLM is a stub. Wire it up via ollama-python if needed: "
            "pip install ollama; then POST /api/generate."
        )


class OpenAILLM:
    """Stub for OpenAI-compatible provider. Not used in demo."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def answer(self, question: str, facts: Sequence[Fact]) -> str:  # pragma: no cover
        raise NotImplementedError(
            "OpenAILLM is a stub. Wire it up via openai-python if needed: "
            "pip install openai; set OPENAI_API_KEY in .env."
        )


def get_llm(provider: str = "mock") -> LLMClient:
    """Factory. Demo default = mock."""
    provider = provider.lower()
    if provider == "mock":
        return MockLLM()
    if provider == "ollama":
        return OllamaLLM()
    if provider == "openai":
        return OpenAILLM()
    raise ValueError(f"Unknown LLM provider: {provider!r}")
