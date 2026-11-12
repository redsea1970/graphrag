"""End-to-end RAG pipeline: question -> entities -> facts -> answer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rag.llm import Fact, LLMClient, get_llm
from rag.retriever import extract_question_entities


@dataclass
class QAResult:
    question: str
    entities: list[str]
    facts: list[Fact]
    answer: str


class RAGPipeline:
    """Question -> extract entities -> retrieve facts -> LLM answer."""

    def __init__(self, retriever, llm: LLMClient | None = None) -> None:
        self.retriever = retriever
        self.llm = llm or get_llm("mock")

    def ask(self, question: str, max_hops: int = 1) -> QAResult:
        entities = extract_question_entities(question)
        facts = self.retriever.retrieve(entities, max_hops=max_hops)
        answer = self.llm.answer(question, facts)
        return QAResult(
            question=question,
            entities=entities,
            facts=facts,
            answer=answer,
        )
