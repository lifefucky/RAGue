Метрики оценки качества reranking
Для оценки эффективности rerankers используются специальные метрики:

MRR (Mean Reciprocal Rank): Средняя величина обратного ранга первого релевантного документа. Чем выше MRR, тем лучше.
NDCG (Normalized Discounted Cumulative Gain): Учитывает позицию релевантных документов в выдаче. Более важна позиция документа в топе списка.
MAP (Mean Average Precision): Средняя точность по всем позициям релевантных документов.
Recall@K: Доля релевантных документов, попавших в топ-K результатов.

Уровни тестирования RAG-систем
Smoke-тесты — базовая проверка работоспособности системы. Проверяем, что агент отвечает и не падает на простых сценариях.
Unit-тесты — проверка отдельных компонентов: ретривера, грейдера, генератора ответов.
Метрики качества — количественная оценка: precision, recall, faithfulness, citation rate.

Smoke-тесты: базовая проверка
Smoke-тесты позволяют убедиться, что агент работает на самых простых сценариях:

from langchain_core.messages import HumanMessage

def smoke_test_agent(app):
    test_cases = [
        {"name": "Простой вопрос без поиска", "query": "Привет!", "should_retrieve": False},
        {"name": "Вопрос, требующий поиска", "query": "Что такое LangGraph?", "should_retrieve": True},
        {"name": "Длинный вопрос", "query": "Объясни архитектуру LangGraph и её компоненты", "should_retrieve": True}
    ]

Создание тестового датасета
Для полноценного тестирования необходимо иметь размеченный набор вопросов и ожидаемых результатов:

test_dataset = [
    {
        "question": "Что такое LangGraph?",
        "expected_answer_contains": ["граф", "workflow", "агент"],
        "relevant_docs": ["doc2"],
        "should_cite": True
    },
    {
        "question": "Для чего используется RAG?",
        "expected_answer_contains": ["retrieval", "generation", "знания"],
        "relevant_docs": ["doc3"],
        "should_cite": True
    },
    {
        "question": "Привет, как дела?",
        "expected_answer_contains": None,
        "relevant_docs": [],
        "should_cite": False
    },
]

Метрики качества retrieval
Основные метрики для оценки поиска документов:

Precision@k
Показывает долю релевантных документов среди первых k найденных:

def calculate_precision_at_k(retrieved_docs: list, relevant_docs: list, k: int) -> float:
    retrieved_top_k = retrieved_docs[:k]
    relevant_found = sum(1 for doc in relevant_docs if doc in retrieved_top_k)
    return relevant_found / k if k > 0 else 0.0

                  
Recall@k
Показывает какой процент всех релевантных документов был найден:

def calculate_recall_at_k(retrieved_docs: list, relevant_docs: list, k: int) -> float:
    retrieved_top_k = retrieved_docs[:k]
    relevant_found = sum(1 for doc in relevant_docs if doc in retrieved_top_k)
    return relevant_found / len(relevant_docs) if relevant_docs else 0.0

                  
MRR (Mean Reciprocal Rank)
Насколько высоко стоит первый релевантный документ:

def calculate_mrr(test_results: list) -> float:
    reciprocal_ranks = []
    for r in test_results:
        retrieved = r.get("retrieved_docs", [])
        relevant = set(r.get("relevant_docs", []))
        rank = next((i+1 for i, doc in enumerate(retrieved) if doc in relevant), None)
        reciprocal_ranks.append(1.0/rank if rank else 0.0)
    return sum(reciprocal_ranks)/len(test_results) if test_results else 0.0

                  
Метрики качества generation
Основные показатели качества ответов:

Faithfulness — факты проверяемы и подтверждены источниками
Answer Relevance — отвечает ли ответ на заданный вопрос
Correctness — соответствует ли ответ эталонным данным
Citation Rate — доля утверждений с корректными ссылками на источники

Использование RAGAS
RAGAS — стандарт индустрии для оценки RAG-систем с проверенными метриками.

Типичные хорошие значения:

Faithfulness ≥ 0.85
Answer Relevance ≥ 0.90
Precision@k ≥ 0.70
Citation Rate ≥ 0.60 (если требуются обязательные ссылки)
