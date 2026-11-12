"""Retrievers: fetch facts from the knowledge graph for a question.

graph.json format (actual):
  nodes: {id, text, type, mentions}
  edges: {source, target, rel, weight}   # source/target are TEXT names

Two backends:
- GraphJsonRetriever: offline, parses output/graph.json directly (tests, fallback).
- Neo4jRetriever:    live Neo4j (docker compose, neo4j/graphrag123).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from rag.llm import Fact


def _load_graph_dict(graph_path: Path) -> dict:
    if not graph_path.exists():
        raise FileNotFoundError(
            f"{graph_path} not found. Run: python graph_builder.py"
        )
    return json.loads(graph_path.read_text(encoding="utf-8"))


class GraphJsonRetriever:
    """Offline retriever over output/graph.json."""

    def __init__(self, graph_path: str | Path = "output/graph.json") -> None:
        self.graph_path = Path(graph_path)
        self.data = _load_graph_dict(self.graph_path)

    def retrieve(self, entities: Sequence[str], max_hops: int = 1) -> list[Fact]:
        if not entities:
            return []
        names = {e.strip().lower() for e in entities if e.strip()}
        edges = self.data.get("edges", [])

        # Direct hits: entity appears as source or target.
        selected: set[str] = set()
        for e in edges:
            s, t = str(e["source"]), str(e["target"])
            if s.lower() in names or t.lower() in names:
                selected.add(s)
                selected.add(t)

        # Hop expansion (names are unique keys here).
        for _ in range(max(0, max_hops - 1)):
            grown = set(selected)
            for e in edges:
                s, t = str(e["source"]), str(e["target"])
                if s in selected or t in selected:
                    grown.add(s)
                    grown.add(t)
            selected = grown

        out: list[Fact] = []
        seen: set[tuple[str, str, str]] = set()
        for e in edges:
            s, t = str(e["source"]), str(e["target"])
            if s not in selected and t not in selected:
                continue
            rel = str(e.get("rel", "RELATED_TO"))
            key = (s, rel, t)
            if key in seen:
                continue
            seen.add(key)
            out.append(Fact(s, rel, t))
        return out


class Neo4jRetriever:
    """Live Neo4j retriever. Requires docker compose up -d."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "graphrag123",
    ) -> None:
        from neo4j import GraphDatabase  # local import: only when used

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def retrieve(self, entities: Sequence[str], max_hops: int = 1) -> list[Fact]:
        if not entities:
            return []
        names = [e.strip().lower() for e in entities if e.strip()]
        cypher_1 = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE toLower(a.name) IN $names OR toLower(b.name) IN $names
        RETURN a.name AS s, type(r) AS rel, b.name AS o
        LIMIT 200
        """
        cypher_2 = """
        MATCH (a:Entity)-[r1]->(m:Entity)-[r2]->(b:Entity)
        WHERE toLower(a.name) IN $names
        RETURN a.name AS s, type(r1) AS rel, m.name AS o
        UNION
        MATCH (a:Entity)-[r1]->(m:Entity)-[r2]->(b:Entity)
        WHERE toLower(a.name) IN $names
        RETURN m.name AS s, type(r2) AS rel, b.name AS o
        LIMIT 300
        """
        cypher = cypher_1 if max_hops <= 1 else cypher_2
        out: list[Fact] = []
        seen: set[tuple[str, str, str]] = set()
        with self.driver.session() as sess:
            for rec in sess.run(cypher, names=names):
                key = (rec["s"], rec["rel"], rec["o"])
                if key in seen:
                    continue
                seen.add(key)
                out.append(Fact(rec["s"], rec["rel"], rec["o"]))
        return out


def extract_question_entities(question: str) -> list[str]:
    """Reuse existing Natasha pipeline to pull entity names from the question."""
    from nlp.extract import extract_entities

    names: list[str] = []
    for e in extract_entities(question):
        txt = getattr(e, "text", None)
        if txt and txt not in names:
            names.append(txt)
    return names
