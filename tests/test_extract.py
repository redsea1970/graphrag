from nlp.extract import extract


def test_entities_and_relations():
    text = 'Иван Петров работает в Сбере. Сбер находится в Москве.'
    ext = extract(text)
    names = {e.text for e in ext.entities}
    assert 'Иван Петров' in names
    assert 'Москва' in names
    rels = {(r.head, r.rel, r.tail) for r in ext.relations}
    assert ('Иван Петров', 'WORKS_AT', 'Сбер') in rels
    assert ('Сбер', 'LOCATED_IN', 'Москва') in rels
