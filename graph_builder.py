"""Сборка графа из Extraction (networkx) + сохранение в JSON."""
import json
from pathlib import Path
from typing import Dict, List
import networkx as nx
from nlp.extract import extract, Extraction


def build_graph(ext: Extraction) -> nx.DiGraph:
    g = nx.DiGraph()
    # узлы: уникальные по (text, type)
    for e in ext.entities:
        key = (e.text, e.type)
        if key not in g:
            g.add_node(key, type=e.type, mentions=1)
        else:
            g.nodes[key]["mentions"] += 1
    # рёбра: по связям (если оба узла уже есть)
    for r in ext.relations:
        head_key = _find(g, r.head)
        tail_key = _find(g, r.tail)
        if head_key and tail_key:
            if g.has_edge(head_key, tail_key):
                g[head_key][tail_key]["weight"] += 1
            else:
                g.add_edge(head_key, tail_key, rel=r.rel, weight=1)
    return g


def _find(g: nx.DiGraph, text: str):
    for (t, _) in g.nodes:
        if t == text:
            return (t, _)
    return None


def graph_to_dict(g: nx.DiGraph) -> Dict:
    return {
        "nodes": [
            {"id": f"n{i}", "text": n[0], "type": n[1], **data}
            for i, (n, data) in enumerate(g.nodes(data=True))
        ],
        "edges": [
            {"source": u[0], "target": v[0], "rel": data.get("rel"), "weight": data.get("weight", 1)}
            for u, v, data in g.edges(data=True)
        ],
    }


def save_graph(g: nx.DiGraph, path: str = "output/graph.json") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(graph_to_dict(g), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def process_text(text: str, out_path: str = "output/graph.json") -> Path:
    ext = extract(text)
    g = build_graph(ext)
    return save_graph(g, out_path)
