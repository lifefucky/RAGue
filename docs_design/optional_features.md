Ранняя остановка (Early Stopping)
Ранняя остановка — это техника, позволяющая прекратить обработку запроса досрочно, если уже достигнут достаточно хороший результат. Зачем продолжать вычисления, если мы уже нашли высокорелевантные документы?

Основная идея: устанавливаем порог качества (threshold), и если результат превышает его, останавливаем дальнейшую обработку. Это особенно эффективно в многоэтапных pipeline, где каждый следующий этап улучшает результат лишь незначительно.

Пример применения в RAG:

Если векторный поиск нашёл документы с очень высоким score (например, >0.95), можно пропустить BM25-поиск
Если первые 5 документов после reranking имеют score >0.9, нет смысла обрабатывать оставшиеся 95
Если LLM уверенно ответила на вопрос (высокая confidence), можно не делать fallback-запросы

Метод stream() у агентов (Стриминг выполнения) (https://stepik.org/lesson/2245613/step/4?unit=2279153)

Supervisor вместо вызова tools.

Оценка релевантности документа по 100 бальной шкале (https://stepik.org/lesson/2245629/step/4?unit=2279169)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5")

def critic_node(state: SupervisorStateWithQA):
    """Критик оценивает качество последнего результата."""
    
    # Берем последнее сообщение воркера
    last_message = state["messages"][-1].content
    
    # Критик оценивает результат
    critique_prompt = f"""Оцени качество выполнения задачи по шкале 0-100.
    
Результат работы: {last_message}

Верни только число от 0 до 100.
Если результат полный и правильный - 80+
Если есть недочеты - 50-79
Если результат неверен - меньше 50"""

from typing import Literal

MAX_RETRIES = 2

def route_after_critic(state: SupervisorStateWithQA) -> Literal["continue", "retry", "escalate"]:
    """Маршрутизация на основе оценки критика."""
    
    score = state.get("quality_score", 0)
    retry_count = state.get("retry_count", 0)
    
    # Результат хороший - продолжаем
    if score >= 80:
        return "continue"
    
    # Результат плохой, но попытки еще есть - повторяем
    if score < 80 and retry_count < MAX_RETRIES:
        return "retry"
    
    # Исчерпали попытки - эскалация
    return "escalate"



Параллельное выполнение через Send API
LangGraph поддерживает паттерн map-reduce через Send. Мы можем создать узел, который генерирует несколько параллельных задач воркеров
Чтобы получить разные мнения, каждый воркер работает с немного разной температурой
После того как все воркеры завершили работу, агрегатор собирает результаты и выбирает финальный ответ, например: 
from collections import Counter

def aggregator_majority(state: ConsensusState):
    """Агрегирует результаты через простое большинство."""
    
    results = state["worker_results"]
    answers = [r["answer"] for r in results]
    
    # Подсчитываем частоту каждого ответа
    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    
    final_answer = most_common[0]
    confidence = most_common[1] / len(answers)
    
    # Добавляем информацию о консенсусе
    consensus_message = {
        "role": "assistant",
        "content": f"Консенсус ({confidence*100:.0f}%): {final_answer}"
    }
    
    return {
        "messages": state["messages"] + [consensus_message],
        "final_answer": final_answer
    }
Агрегатор выбирает ответ, который встречается чаще всего.


Альтернатива: LLM-committee
Для задач где нет однозначного ответа, используем LLM для синтеза:

def aggregator_llm_synthesis(state: ConsensusState):
    """Агрегирует результаты через LLM-синтез."""
    
    results = state["worker_results"]
    task = state["messages"][0].content
    
    # Формируем промпт с разными мнениями
    opinions = "\n\n".join([
        f"Мнение {r['worker_id']+1}: {r['answer']}"
        for r in results
    ])
    
    synthesis_prompt = f"""Задача: {task}

Получены следующие мнения от экспертов:
{opinions}

Проанализируй эти мнения и дай итоговый ответ.
Если мнения расходятся, объясни почему и дай взвешенное решение."""

    synthesizer = ChatOpenAI(model="gpt-5", temperature=0)
    response = synthesizer.invoke(synthesis_prompt)
    
    synthesis_message = {
        "role": "assistant",
        "content": response.content
    }
    
    return {
        "messages": state["messages"] + [synthesis_message],
        "final_answer": response.content
    }

                  
LLM-committee работает лучше для открытых вопросов, где важен анализ разных перспектив.

Когда использовать консенсус
Используйте когда:

Задача критична и ошибка дорого стоит (медицина, финансы)
Модель систематически ошибается на этом типе задач
Нужна высокая уверенность в результате
Открытые вопросы где полезны разные перспективы
Не используйте когда:

Задача простая и модель редко ошибается
Бюджет ограничен (N воркеров = N× стоимость)
Скорость важнее точности

