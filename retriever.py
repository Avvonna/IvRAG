import logging
import re
from string import Template

from config import PipelineConfig
from schemas import RetrieverOut, ScoredQuestion
from utils import find_top_match, retry_call, split_dict_into_chunks

logger = logging.getLogger(__name__)


def retriever(
    user_query: str,
    config: PipelineConfig,
    n_blocks: int = 2
) -> RetrieverOut:
    logger.info(f"Starting retriever for query: {user_query[:100]}...")
    
    if not config.all_QS_clean_list:
        logger.warning("Empty question list in config")
        return RetrieverOut(results=[])

    # questions_block = "\n".join(
    #     f"{i+1}. {q}" for i, q in enumerate(config.all_QS_info_dict)
    # )
    questions_blocks = config.catalog.as_value_catalog()
    questions_blocks = split_dict_into_chunks(questions_blocks, n_blocks)

    rc = config.retriever_config

    prompt_template = Template(f"""

**ЦЕЛЬ:** Найти минимальный набор вопросов для решения аналитической задачи пользователя.
**ЗАПРОС:** {user_query}

**ДОСТУПНЫЕ ВОПРОСЫ:**
$q_block

## ФОРМАТ ОТВЕТА:

**Анализ задачи:** [Краткое описание аналитической цели и требуемого хода исследования]

**Выбранные вопросы:**

1. "<ТОЧНАЯ КОПИЯ ВОПРОСА ИЗ СПИСКА>"
   Обоснование: [Зачем этот вопрос для решения задачи]
=

""")
    def _blocks_iter():
        logger.debug(f"Calling LLM with model: {rc.model}")
        res = []

        for i, block in enumerate(questions_blocks, start=1):
            logger.debug(f"{i}/{len(questions_blocks)} part of questions...")
            prompt = prompt_template.substitute({"q_block": block})
            response_text = retry_call(lambda: _call(prompt), retries=rc.retries, base_delay=rc.base_delay)
            logger.debug(f"LLM response length: {len(response_text)}")

            # retriever_struct_out = _parse_retriever_response(response_text)
            res += [response_text]
        
        # logger.info(f"Found {len(res.results)} candidate questions")
        return res
    
    def _call(prompt):
        logger.debug(f"Prompt length: {len(prompt)}")
        
        resp = config.client.chat.completions.create(
            model=rc.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=rc.temperature,
        )
        assert resp.choices[0].message.content, "Empty retriever response"
        return resp.choices[0].message.content.strip()
    
    res = retry_call(_blocks_iter, retries=rc.retries, base_delay=rc.base_delay)
    res = "\n".join(res)
    res = _parse_retriever_response(res)

    # Нормализуем названия вопросов через fuzzy matching
    for qs in res.results:
        original = qs.question
        qs.question = find_top_match(qs.question, config.all_QS_clean_list)
        if qs.question != original:
            logger.debug(f"FUZZY-norm question: '{original}' -> '{qs.question}'")

    logger.info(f"Retriever completed with {len(res.results)} questions")

    return res


def _parse_retriever_response(text: str) -> RetrieverOut:
    logger.debug("Parsing retriever response")
    
    pattern = r'''
    [\s\*]*
    (?P<number>\d+)\.?[\s\*]*
    "(?P<question>[^"]+)"
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
                    reason=match.group('reason').strip(),
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
    
    logger.debug(f"Successfully parsed {len(scored_questions)} questions")
    return RetrieverOut(results=scored_questions)
