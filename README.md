# GraphRAG

Извлечение сущностей и связей из русского текста (Natasha), построение графа знаний (NetworkX), загрузка в Neo4j.

## Стек
Python 3.13, spaCy, Natasha, NetworkX, Neo4j (Docker), LangChain.

## Установка
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

## NLP-слой
python -c "from graph_builder import process_text; process_text('Иван Петров работает в Сбере. Газпром находится в Москве.', 'output/graph.json')"
Результат — output/graph.json (узлы PER/ORG/LOC, связи WORKS_AT/LOCATED_IN/RELATED_TO).
Организации-исключения (Сбер, Газпром и др.) переклассифицируются в ORG через ORG_OVERRIDES.

## Neo4j
docker compose up -d
- Browser: http://localhost:7474 (neo4j / graphrag123)
- Загрузка графа:
python load_neo4j.py
Скрипт очищает :Entity и заливает узлы/связи из output/graph.json.

## Тесты
python -m pytest -q
