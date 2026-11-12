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
    """Local Ollama provider via HTTP /api/generate.

    Env:
        OLLAMA_HOST  default http://localhost:11434
        OLLAMA_MODEL default qwen2.5:7b
    Falls back to MockLLM answer if Ollama unreachable.
    """

    PROMPT_TMPL = (
        "Ты — ассистент по графу знаний. Ответь на вопрос, "
        "используя ТОЛЬКО перечисленные факты. Если фактов недостаточно — так и скажи.\n\n"
        "Факты:\n{facts}\n\nВопрос: {question}\nОтвет:"
    )

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        import os

        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def _facts_block(self, facts: Sequence[Fact]) -> str:
        if not facts:
            return "(фактов нет)"
        return "\n".join(f"- {f.as_text()}" for f in facts)

    def answer(self, question: str, facts: Sequence[Fact]) -> str:
        import httpx

        prompt = self.PROMPT_TMPL.format(facts=self._facts_block(facts), question=question)
        try:
            r = httpx.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            text = (data.get("response") or "").strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            return (
                f"[OllamaLLM недоступна: {exc!r}] "
                + MockLLM().answer(question, facts)
            )
        return MockLLM().answer(question, facts)


class OpenAILLM:
    """OpenAI-compatible provider (lazy import, optional dep).

    Env:
        OPENAI_API_KEY required
        OPENAI_MODEL   default gpt-4o-mini
        OPENAI_BASE_URL optional (for compatible endpoints)
    Falls back to MockLLM on errors / missing key.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        import os

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

    def answer(self, question: str, facts: Sequence[Fact]) -> str:
        if not self.api_key:
            return "[OpenAILLM: нет OPENAI_API_KEY] " + MockLLM().answer(question, facts)
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except Exception:
            return (
                "[OpenAILLM: пакет openai не установлен — pip install openai] "
                + MockLLM().answer(question, facts)
            )

        facts_block = "\n".join(f"- {f.as_text()}" for f in facts) or "(фактов нет)"
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Отвечай только по данным фактам, кратко."},
                    {"role": "user", "content": f"Факты:\n{facts_block}\n\nВопрос: {question}"},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip() or MockLLM().answer(question, facts)
        except Exception as exc:  # noqa: BLE001
            return f"[OpenAILLM ошибка: {exc!r}] " + MockLLM().answer(question, facts)


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
