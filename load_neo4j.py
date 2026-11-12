"""Загрузка output/graph.json в Neo4j.

Связи типизируются по имени отношения (WORKS_AT / LOCATED_IN / RELATED_TO),
а не хранятся как :REL с полем. Whitelist защищает от Cypher-инъекций
при динамическом типе связи.
"""
import json
import os
from pathlib import Path

from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "graphrag123")

ALLOWED_REL_TYPES = {"WORKS_AT", "LOCATED_IN", "RELATED_TO"}


def _safe_rel_type(rel: str) -> str:
    """Return an allowed relationship type; fall back to RELATED_TO."""
    rel = (rel or "").strip().upper()
    return rel if rel in ALLOWED_REL_TYPES else "RELATED_TO"


def load(graph_path: str = "output/graph.json") -> None:
    data = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    with driver.session() as s:
        s.run("MATCH (n:Entity) DETACH DELETE n")
        for node in data.get("nodes", []):
            s.run(
                "MERGE (e:Entity {id: $id}) "
                "SET e.name = $name, e.type = $type, e.mentions = $mentions",
                id=node["id"], name=node["text"], type=node["type"],
                mentions=node.get("mentions", 1),
            )
        for edge in data.get("edges", []):
            rel = _safe_rel_type(edge.get("rel", ""))
            # rel is whitelisted above -> safe to inline as type
            cypher = (
                "MATCH (a:Entity {name: $src}) MATCH (b:Entity {name: $dst}) "
                f"MERGE (a)-[r:`{rel}`]->(b) "
                "SET r.weight = $weight"
            )
            s.run(cypher, src=edge["source"], dst=edge["target"],
                  weight=edge.get("weight", 1))
        cnt = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        rc = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        types = [rec["t"] for rec in s.run(
            "CALL db.relationshipTypes() YIELD relationshipType AS t RETURN t"
        )]
    driver.close()
    print(f"Neo4j: {cnt} узлов, {rc} связей")
    print(f"Relationship types: {types}")


if __name__ == "__main__":
    load()
