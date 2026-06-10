# Evaluation dataset fixtures

Шаблон labeled dataset для `rague/evaluation/`.

## Файлы

| Файл | Назначение |
|------|------------|
| `basic_cases_example.json` | Минимальный пример формата — можно коммитить и делиться |
| `basic_cases.json` | Полный локальный fixture для unit-тестов (не коммитить, если содержит внутренние вопросы) |
| `data/evaluation/basic_cases.json` | Corpus-bound dataset для live eval против Qdrant (gitignored) |

## Быстрый старт

1. Скопируйте пример:

```bash
cp tests/fixtures/evaluation/basic_cases_example.json data/evaluation/my_cases.json
```

2. Замените placeholder-значения на реальные labels вашего корпуса.
3. Запустите eval с явным путём к dataset:

```bash
python3 -m rague.evaluation retrieval --dataset data/evaluation/my_cases.json
python3 -m rague.evaluation agent --dataset data/evaluation/my_cases.json --limit 5
python3 -m rague.evaluation agent-trace --dataset data/evaluation/my_cases.json --limit 3
```

Если `data/evaluation/basic_cases.json` существует, CLI использует его по умолчанию; иначе — `tests/fixtures/evaluation/basic_cases.json`.

## Формат case

Dataset — JSON-массив объектов. Каждый объект описывает один labeled вопрос.

### Обязательные поля

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | string | Уникальный идентификатор кейса |
| `question` | string | Вопрос пользователя |
| `relevant_docs` | string[] | ID релевантных документов из вашего корпуса |
| `should_retrieve` | boolean | Ожидается ли вызов retrieval |
| `should_cite` | boolean | Должен ли ответ содержать citations |

### Опциональные поля

| Поле | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `expected_answer_contains` | string[] \| null | — | Подстроки, которые должны встретиться в ответе (case-insensitive). `null` — проверка не выполняется |
| `relevant_id_field` | string | `"chunk_id"` | Поле metadata для сопоставления retrieval: `chunk_id`, `document_id`, `page_id` |
| `query_type` | string | `"fact_lookup"` | Метка типа вопроса для отчётов: `fact_lookup`, `code_lookup`, `long_context`, `greeting` и т.д. |
| `notes` | string | — | Произвольные заметки для человека; eval не использует |

### Пример

См. `basic_cases_example.json`:

```json
[
  {
    "id": "example-fact-lookup",
    "question": "Example question about a documented process or fact",
    "expected_answer_contains": ["keyword-a", "keyword-b"],
    "relevant_docs": ["page-001"],
    "relevant_id_field": "page_id",
    "query_type": "fact_lookup",
    "should_retrieve": true,
    "should_cite": true,
    "notes": "Copy this file, replace placeholders with your corpus labels, and point --dataset to your copy."
  }
]
```

## Типичные паттерны кейсов

**Retrieval + generation с citations** — основной сценарий:

```json
{
  "id": "my-code-lookup",
  "question": "How do I run the backup script?",
  "expected_answer_contains": ["backup.sh", "cron"],
  "relevant_docs": ["12345"],
  "relevant_id_field": "page_id",
  "query_type": "code_lookup",
  "should_retrieve": true,
  "should_cite": true
}
```

**Multi-page** — несколько релевантных страниц:

```json
{
  "id": "multi-topic",
  "question": "Where are setup and troubleshooting described?",
  "expected_answer_contains": ["install", "error"],
  "relevant_docs": ["111", "222"],
  "relevant_id_field": "page_id",
  "query_type": "long_context",
  "should_retrieve": true,
  "should_cite": true
}
```

**Greeting / no-retrieval** — вопрос без поиска по корпусу:

```json
{
  "id": "greeting",
  "question": "Hello!",
  "expected_answer_contains": null,
  "relevant_docs": [],
  "query_type": "greeting",
  "should_retrieve": false,
  "should_cite": false
}
```

## Выбор `relevant_id_field`

Значение должно совпадать с metadata полем в Qdrant:

- `page_id` — labels на уровне Confluence-страниц (типично для RAGue)
- `chunk_id` — labels на уровне отдельных чанков
- `document_id` — labels на уровне document_id в metadata

Retrieval metrics сравнивают retrieved IDs с `relevant_docs` через выбранное поле.

## Загрузка в коде

```python
from rague.evaluation.dataset import load_evaluation_cases

cases = load_evaluation_cases("data/evaluation/my_cases.json")
```

Loader: `rague/evaluation/dataset.py`. При ошибке валидации будет `ValueError` с указанием индекса и поля.

## Что не коммитить

Держите corpus-bound datasets с реальными вопросами и page ID локально в `data/evaluation/` — эта директория в `.gitignore`. В репозиторий добавляйте только шаблон `basic_cases_example.json` или обезличенные копии.
