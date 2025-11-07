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
Запрос пользователя: {user_query}

Список доступных вопросов:
{questions_block}

---
Инструкция:

1. Проанализируй запрос
2. Подбери релевантные вопросы из списка с оценкой релевантности (0-100)

Формат вывода:

**Рассуждение:**
[Краткий анализ запроса: 2-5 предложений о том, что нужно пользователю]

**Рекомендованные вопросы:**

1. **[90/100]** "[Исходная формулировка вопроса]"
   • **Обоснование:** [Почему вопрос напрямую отвечает на запрос]

2. **[80/100]** "[Исходная формулировка вопроса]"
   • **Обоснование:** [Как вопрос связан с ключевой темой]

3. **[60/100]** "[Исходная формулировка вопроса]"
   • **Обоснование:** [Какой контекст или смежную тему раскрывает]

4. **[50/100]** "[Исходная формулировка вопроса]"
   • **Обоснование:** [Чем может быть полезен для понимания]
---
"""

    def _call():
        logger.debug(f"Calling LLM with model: {rc.model}")
        resp = config.client.chat.completions.create(
            model=rc.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=rc.temperature,
        )
        assert resp.choices[0].message.content, "Empty retriever response"
        return resp.choices[0].message.content.strip()

    response_text = retry_call(_call, retries=rc.retries, base_delay=rc.base_delay)
    logger.debug(f"LLM response length: {len(response_text)}")
    
    retriever_struct_out = _parse_retriever_response(response_text)
    
    logger.info(f"Found {len(retriever_struct_out.results)} candidate questions")

    # Нормализуем названия вопросов через fuzzy matching
    for qs in retriever_struct_out.results:
        original = qs.question
        qs.question = find_top_match(qs.question, config.all_QS_clean_list)
        if qs.question != original:
            logger.debug(f"Normalized question: '{original}' -> '{qs.question}'")

    logger.info(f"Retriever completed with {len(retriever_struct_out.results)} questions")
    return retriever_struct_out


def _parse_retriever_response(text: str) -> RetrieverOut:
    logger.debug("Parsing retriever response")
    
    pattern = r'''
        \*{0,2}\[(?P<score>\d+)/100\]\*{0,2}\s*
        ["«]?(?P<question>[^"\n«»]+?)["»]?\s*
        [•\-*]?\s*\*{0,2}Обоснование\*{0,2}:?\s*
        (?P<reason>[^\n]+)
    '''

    matches = re.finditer(pattern, text, re.VERBOSE | re.IGNORECASE)
    
    scored_questions = []
    for match in matches:
        scored_questions.append(
            ScoredQuestion(
                question=match.group("question"),
                reason=match.group("reason").strip(),
                relevance=float(match.group("score"))
            )
        )
    
    if not scored_questions:
        logger.error(f"Failed to parse any questions from response. Response preview: {text[:500]}")
        raise ValueError(
            "Не удалось распарсить ни одного вопроса из ответа LLM. "
            "Проверьте формат ответа модели."
        )
    
    logger.debug(f"Successfully parsed {len(scored_questions)} questions")
    return RetrieverOut(results=scored_questions)