"""CLI demo for the GraphRAG pipeline.

Usage:
    python rag_demo.py "Где работает Иван Петров?"
    python rag_demo.py                 # interactive loop

Backend: Neo4j if reachable, else output/graph.json (offline fallback).
LLM: MockLLM (deterministic demo, no keys).
"""
from __future__ import annotations

import sys

from rag.llm import get_llm
from rag.qa import RAGPipeline
from rag.retriever import GraphJsonRetriever, Neo4jRetriever


def build_retriever():
    try:
        r = Neo4jRetriever()
        # cheap connectivity check
        with r.driver.session() as s:
            s.run("RETURN 1").consume()
        print("[backend] Neo4j (bolt://localhost:7687)")
        return r
    except Exception as e:
        print(f"[backend] Neo4j unavailable ({e.__class__.__name__}); using output/graph.json")
        return GraphJsonRetriever()


def print_result(res) -> None:
    print()
    print(f"Q: {res.question}")
    print(f"Entities: {res.entities or '[]'}")
    print(f"Facts: {len(res.facts)}")
    for f in res.facts:
        print(f"  - {f.as_text()}")
    print("Answer:")
    print(res.answer)
    print("-" * 60)


def main(argv: list[str]) -> int:
    retriever = build_retriever()
    pipe = RAGPipeline(retriever, get_llm("mock"))

    if len(argv) > 1:
        q = " ".join(argv[1:]).strip()
        print_result(pipe.ask(q))
    else:
        print("Interactive RAG demo. Empty line to exit.")
        while True:
            try:
                q = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            print_result(pipe.ask(q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
