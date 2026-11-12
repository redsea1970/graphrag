"""Tests for RAG layer. Offline: uses output/graph.json, MockLLM."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.llm import Fact, MockLLM, get_llm
from rag.qa import RAGPipeline
from rag.retriever import GraphJsonRetriever

GRAPH = Path("output/graph.json")


# ---------- LLM ----------

def test_get_llm_mock() -> None:
    assert isinstance(get_llm("mock"), MockLLM)


def test_get_llm_unknown() -> None:
    with pytest.raises(ValueError):
        get_llm("nope")


def test_mock_llm_no_facts() -> None:
    out = MockLLM().answer("x", [])
    assert "не найдено" in out


def test_mock_llm_with_facts() -> None:
    out = MockLLM().answer("q", [Fact("A", "REL", "B")])
    assert "A — REL — B" in out
    assert "найдено 1" in out


def test_fact_as_text() -> None:
    assert Fact("A", "R", "B").as_text() == "A --R--> B"


# ---------- Retriever ----------

@pytest.fixture(scope="module")
def retriever() -> GraphJsonRetriever:
    if not GRAPH.exists():
        pytest.skip("output/graph.json missing; run graph_builder.py")
    return GraphJsonRetriever(GRAPH)


def test_retriever_empty_entities(retriever: GraphJsonRetriever) -> None:
    assert retriever.retrieve([]) == []


def test_retriever_known_entity(retriever: GraphJsonRetriever) -> None:
    facts = retriever.retrieve(["Сбер"])
    assert any(f.object == "Сбер" or f.subject == "Сбер" for f in facts)


def test_retriever_unknown_entity(retriever: GraphJsonRetriever) -> None:
    assert retriever.retrieve(["НетТакойСущностиXYZ"]) == []


def test_retriever_no_duplicates(retriever: GraphJsonRetriever) -> None:
    facts = retriever.retrieve(["Сбер"])
    keys = [(f.subject, f.relation, f.object) for f in facts]
    assert len(keys) == len(set(keys))


# ---------- Pipeline ----------

def test_pipeline_end_to_end(retriever: GraphJsonRetriever) -> None:
    pipe = RAGPipeline(retriever, MockLLM())
    res = pipe.ask("Где работает Иван Петров?")
    assert res.question == "Где работает Иван Петров?"
    assert isinstance(res.entities, list)
    assert isinstance(res.facts, list)
    assert isinstance(res.answer, str) and res.answer


def test_pipeline_no_entities(retriever: GraphJsonRetriever) -> None:
    pipe = RAGPipeline(retriever, MockLLM())
    res = pipe.ask("qwerty no entities here 123")
    # no crash; empty facts is acceptable for unknown text
    assert res.entities == [] or isinstance(res.entities, list)
    assert res.answer
