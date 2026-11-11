"""Извлечение сущностей и связей из русского текста (Natasha)."""
from dataclasses import dataclass, field, asdict
from typing import List, Dict
from natasha import (
    Segmenter, MorphVocab,
    NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger,
    Doc,
)

_segmenter = Segmenter()
_morph_vocab = MorphVocab()
_emb = NewsEmbedding()
_morph_tagger = NewsMorphTagger(_emb)
_syntax_parser = NewsSyntaxParser(_emb)
_ner_tagger = NewsNERTagger(_emb)

# Словарь оверрайдов: Natasha иногда метит организации как LOC (Сбер, Газпром).
ORG_OVERRIDES = {
    "сбер": "ORG", "сбербанк": "ORG", "газпром": "ORG", "роснефть": "ORG",
    "лукойл": "ORG", "яндекс": "ORG", "mail.ru": "ORG", "мтс": "ORG",
    "билайн": "ORG", "мегафон": "ORG", "тинькофф": "ORG", "втб": "ORG",
    "росбанк": "ORG", "аэрофлот": "ORG", "ржд": "ORG", "роснано": "ORG",
    "ростелеком": "ORG", "росатом": "ORG", "альфа-банк": "ORG",
    "альфабанк": "ORG", "ozon": "ORG", "озон": "ORG", "wildberries": "ORG",
}



@dataclass
class Entity:
    id: str
    text: str
    type: str          # PER, ORG, LOC
    start: int
    stop: int


@dataclass
class Relation:
    head: str
    rel: str
    tail: str
    sent: str


@dataclass
class Extraction:
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "entities": [asdict(e) for e in self.entities],
            "relations": [asdict(r) for r in self.relations],
        }


def _make_doc(text: str) -> Doc:
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)
    doc.parse_syntax(_syntax_parser)
    doc.tag_ner(_ner_tagger)
    for span in doc.spans:
        span.normalize(_morph_vocab)
    return doc


def extract_entities(text: str) -> List[Entity]:
    doc = _make_doc(text)
    out: List[Entity] = []
    for i, span in enumerate(doc.spans):
        norm = span.normal or span.text
        etype = ORG_OVERRIDES.get(norm.lower(), span.type)
        out.append(Entity(
            id=f"e{i}",
            text=norm,
            type=etype,
            start=span.start,
            stop=span.stop,
        ))
    return out


# Простые связи: сущность в пределах одного предложения.
# Тип связи выводим по маркерам в предложении.
_WORK_MARKERS = ("работает", "работал", "работала", "трудится", "служит")
_LOC_MARKERS = ("находится", "расположен", "расположена", "живёт", "живет", "базируется")


def _guess_rel(sent_text: str) -> str:
    low = sent_text.lower()
    if any(m in low for m in _WORK_MARKERS):
        return "WORKS_AT"
    if any(m in low for m in _LOC_MARKERS):
        return "LOCATED_IN"
    return "RELATED_TO"


def extract_relations(text: str) -> List[Relation]:
    doc = _make_doc(text)
    rels: List[Relation] = []
    sent_bounds = [(s.start, s.stop) for s in doc.sents]
    for (s_start, s_stop) in sent_bounds:
        sent_text = text[s_start:s_stop]
        sent_spans = [sp for sp in doc.spans if s_start <= sp.start < s_stop]
        if len(sent_spans) < 2:
            continue
        rel = _guess_rel(sent_text)
        head = sent_spans[0].normal or sent_spans[0].text
        for tail_span in sent_spans[1:]:
            tail = tail_span.normal or tail_span.text
            rels.append(Relation(head=head, rel=rel, tail=tail, sent=sent_text.strip()))
    return rels


def extract(text: str) -> Extraction:
    return Extraction(entities=extract_entities(text), relations=extract_relations(text))
