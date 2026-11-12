"""Загрузка output/graph.json в Neo4j (MERGE узлов и связей)."""
import json
import os
from pathlib import Path

from neo4j import GraphDatabase

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "graphrag123")


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
            s.run(
                "MATCH (a:Entity {name: $src}) MATCH (b:Entity {name: $dst}) "
                "MERGE (a)-[r:REL {type: $rel}]->(b) "
                "SET r.weight = $weight",
                src=edge["source"], dst=edge["target"],
                rel=edge["rel"], weight=edge.get("weight", 1),
            )
        cnt = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        rc = s.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
    driver.close()
    print(f"Neo4j: {cnt} узлов, {rc} связей")


if __name__ == "__main__":
    load()
