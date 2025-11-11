import logging
import re

from config import PipelineConfig
from schemas import RetrieverOut, ScoredQuestion
from utils import find_top_match, retry_call

logger = logging.getLogger(__name__)


def retriever(user_query: str, config: PipelineConfig) -> RetrieverOut:
    logger.info(f"Starting retriever for query: {user_query[:100]}...")
    
    if not config.all_QS_clean_list:
        logger.warning("Empty question list in config")
        return RetrieverOut(results=[])

    questions_block = "\n".join(
        f"{i+1}. {q}" for i, q in enumerate(config.all_QS_info_dict)
    )

    rc = config.retriever_config

    prompt = f"""
# ТЕХНИЧЕСКОЕ ЗАДАНИЕ: Отбор вопросов из результатов опросов населения России на различные темы для наиболее точного ответа на вопрос пользователя

**ЦЕЛЬ:** Найти минимальный набор вопросов для решения аналитической задачи пользователя через фильтрацию респондентов и анализ

**ЗАПРОС:** {user_query}

**ДОСТУПНЫЕ ВОПРОСЫ:**
{questions_block}

## СПЕЦИАЛИЗИРОВАННЫЕ КРИТЕРИИ ОТБОРА:

### ВЫСОКИЙ ПРИОРИТЕТ (релевантность: 90-100):
- Прямая тематическая связь с запросом
- Ключевые вопросы для понимания объекта анализа
- Вопросы, формирующие основу фильтрации

### СРЕДНИЙ ПРИОРИТЕТ (релевантность: 60-89):
- Вспомогательные вопросы по теме
- Факторы влияния на основной показатель
- Вопросы для кросс-анализа

### СЕГМЕНТАЦИОННЫЕ ВОПРОСЫ (релевантность: 40-59):
- Демографические характеристики (возраст, доход, образование)
- Географические признаки (регион, город, тип населенного пункта)
- Социально-экономические факторы

### КОНТЕКСТУАЛЬНЫЕ (релевантность: 25-39):
- Поведенческие паттерны
- Предпочтения и привычки
- Условия принятия решений

### НИЗКИЙ ПРИОРИТЕТ (релевантность: 0-24):
- Не релевантные для конкретной задачи
- Слишком общие или неспецифичные вопросы

## СТРАТЕГИЯ ОТБОРА:

1. **Определи тип анализа:**
   - Простое распределение → основной вопрос + сегментация
   - Сравнительный анализ → несколько связанных вопросов
   - Причинно-следственный → основной + факторы влияния
   - Сегментация → демография + поведенческие вопросы

2. **Построй минимальную цепочку:**
   - 1-3 основных вопроса по теме
   - 1-3 вопроса для сегментации (обязательно для аналитики)
   - При необходимости: 1-3 контекстных вопроса

3. **Обеспечь логическую связность:**
   - Вопросы должны дополнять друг друга
   - Избегай дублирования функций
   - Приоритет практической применимости

## ФОРМАТ ОТВЕТА:

**Анализ задачи:** [Краткое описание аналитической цели и требуемого типа исследования]

**Выбранные вопросы:**

1. [<релевантность>/100] "<ТОЧНАЯ КОПИЯ ВОПРОСА ИЗ СПИСКА>"
   Роль: [Основной/Сегментационный/Контекстный]
   Обоснование: [Зачем этот вопрос для решения задачи]

[Продолжить для всех выбранных вопросов, максимум 10]

## КРИТИЧЕСКИЕ ТРЕБОВАНИЯ:
- Копируй вопросы ТОЧНО как в списке (без изменений)
- Включай минимум 1 демографический и 1 географический вопрос для любого анализа
- Не выбирай больше 10 вопросов (приоритет качества над количеством)
- Каждый вопрос должен иметь чёткую роль в решении задачи
"""

    def _call():
        logger.debug(f"Calling LLM with model: {rc.model}")
        logger.debug(f"Prompt length: {len(prompt)}")
        
        resp = config.client.chat.completions.create(
            model=rc.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=rc.temperature,
        )
        assert resp.choices[0].message.content, "Empty retriever response"
        return resp.choices[0].message.content.strip()

    response_text = retry_call(_call, retries=rc.retries, base_delay=rc.base_delay)
    logger.debug(f"LLM response length: {len(response_text)}")

    # TODO: использовать structured output вместо парсинга
    retriever_struct_out = _parse_retriever_response(response_text)
    
    logger.info(f"Found {len(retriever_struct_out.results)} candidate questions")

    # Нормализуем названия вопросов через fuzzy matching
    for qs in retriever_struct_out.results:
        original = qs.question
        qs.question = find_top_match(qs.question, config.all_QS_clean_list)
        if qs.question != original:
            logger.debug(f"FUZZY-norm question: '{original}' -> '{qs.question}'")

    logger.info(f"Retriever completed with {len(retriever_struct_out.results)} questions")
    return retriever_struct_out


def _parse_retriever_response(text: str) -> RetrieverOut:
    logger.debug("Parsing retriever response")
    
    pattern = r'''
    [\s\*]*
    (?P<number>\d+)\.?[\s\*]*
    \[(?P<score>\d+)/100\][\s\*]*
    "(?P<question>[^"]+)"
    [\n\s\*]*Роль:[\n\s\*]*
    (?P<role>[^\n]+?)\s*
    [\n\s\*]*Обоснование:[\n\s\*]*
    (?P<reason>[^\n]+)
    '''

    matches = re.finditer(pattern, text, re.VERBOSE | re.IGNORECASE)
    
    scored_questions = []
    for match in matches:
        try:
            scored_questions.append(
                ScoredQuestion(
                    question=match.group("question").strip(),
                    reason=f"[{match.group('role').strip()}] {match.group('reason').strip()}",
                    relevance=float(match.group("score"))
                )
            )
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse question: {e}")
            continue
    
    if not scored_questions:
        logger.error(f"Failed to parse any questions from response. Response: {text}")
        raise ValueError(
            "Не удалось распарсить ни одного вопроса из ответа LLM. "
            "Проверьте формат ответа модели."
        )
    
    # Сортируем по убыванию релевантности
    scored_questions.sort(key=lambda x: x.relevance, reverse=True)
    
    logger.debug(f"Successfully parsed {len(scored_questions)} questions")
    return RetrieverOut(results=scored_questions)
