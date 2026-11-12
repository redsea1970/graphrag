# GraphRAG

GraphRAG: извлечение сущностей и связей из русского текста (Natasha), построение графа знаний (NetworkX), загрузка в Neo4j, retrieval по графу и вопрос-ответ (RAG-демо).

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

## RAG (demo)
Retrieval-augmented generation поверх графа знаний.

Пайплайн: вопрос -> извлечение сущностей (Natasha) -> поиск фактов в графе (Neo4j, fallback: output/graph.json) -> генерация ответа (MockLLM).

Запуск:
python rag_demo.py "Где работает Иван Петров?"
python rag_demo.py            # интерактивный режим

Бэкенды ретривера:
- Neo4jRetriever    — живой Neo4j (docker compose up -d), связи типизированы (WORKS_AT/LOCATED_IN/RELATED_TO);
- GraphJsonRetriever — офлайн, читает output/graph.json (используется в тестах и как fallback).

LLM-слой: интерфейс LLMClient + MockLLM (детерминированный, без ключей и сети).
Заглушки OllamaLLM / OpenAILLM показывают точку расширения для реальных провайдеров.

## Структура
- nlp/extract.py         — извлечение сущностей и связей (Natasha)
- graph_builder.py       — построение графа, сохранение output/graph.json
- load_neo4j.py          — загрузка графа в Neo4j (типизированные связи)
- rag/llm.py             — LLMClient, MockLLM, заглушки провайдеров
- rag/retriever.py       — Neo4jRetriever, GraphJsonRetriever
- rag/qa.py              — RAGPipeline (вопрос -> факты -> ответ)
- rag_demo.py            — CLI-демо
- tests/                 — pytest (NLP + RAG)

## Тесты
python -m pytest -q
