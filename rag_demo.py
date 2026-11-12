"""CLI demo for the GraphRAG pipeline.

Usage:
    python rag_demo.py "Где работает Иван Петров?"
    python rag_demo.py                 # interactive loop
    python rag_demo.py --llm ollama "Вопрос"
    python rag_demo.py --llm openai "Вопрос"

Backend: Neo4j if reachable, else output/graph.json (offline fallback).
LLM: --llm mock|ollama|openai (default mock; ollama/openai auto-fallback to mock).
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
    import argparse

    ap = argparse.ArgumentParser(description="GraphRAG demo CLI")
    ap.add_argument("question", nargs="*", help="вопрос (иначе интерактив)")
    ap.add_argument(
        "--llm",
        default="mock",
        choices=["mock", "ollama", "openai"],
        help="LLM provider (default: mock)",
    )
    args = ap.parse_args(argv[1:])

    retriever = build_retriever()
    llm = get_llm(args.llm)
    print(f"[llm] {llm.__class__.__name__} (provider={args.llm})")
    pipe = RAGPipeline(retriever, llm)

    if args.question:
        q = " ".join(args.question).strip()
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
