from __future__ import annotations

import logging
import re
from string import Template

from openai.types.chat import ChatCompletion

from .config import PipelineConfig
from .schemas import RetrieverOut, ScoredQuestion
from .utils import find_top_match, retry_call, split_dict_into_chunks

logger = logging.getLogger(__name__)


def retriever(
    user_query: str,
    config: PipelineConfig,
) -> RetrieverOut:
    logger.info(f"Starting retriever for query: {user_query}")

    rc = config.retriever_config

    if not config.all_QS_info_dict:
        logger.warning("No questions in config. Returning empty results.")
        return RetrieverOut(results=[])

    questions_blocks = split_dict_into_chunks(config.catalog.as_value_catalog(), rc.n_questions_splits)
    prompt_template = _make_retriever_prompt(user_query)

    retriever_out = RetrieverOut(results=[])
    reasons = []

    params = {
        "model": rc.model,
        "temperature": rc.temperature
    }

    if rc.max_tokens:
        params["max_tokens"] = rc.max_tokens
    if rc.reasoning_effort:
        params["reasoning_effort"] = rc.reasoning_effort

    extra_body = {}
    if rc.provider_sort:
        extra_body["provider"] = {"sort": rc.provider_sort}
    if extra_body:
        params["extra_body"] = extra_body

    def _call(prompt):
        logger.debug(f"Calling LLM with model: {rc.model}")
        logger.debug(f"Prompt length: {len(prompt)}")
        
        resp: ChatCompletion = config.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        return resp.choices[0].message

    for i, block in enumerate(questions_blocks, start=1):
        logger.debug(f"{i}/{len(questions_blocks)} part of questions...")
        prompt = prompt_template.substitute({"q_block": block})
        msg = retry_call(lambda: _call(prompt), retries=rc.retries, base_delay=rc.base_delay)

        if rc.reasoning_effort:
            try:
                if msg.reasoning_details and len(msg.reasoning_details) > 0: # type: ignore
                    reasons += ["#" * 20, f"Часть {i}", "#" * 20, msg.reasoning_details[0]["text"], "\n"] # type: ignore
                else:
                    logger.warning(f"No reasoning details found for block {i}")
            except (AttributeError, IndexError, KeyError) as e:
                logger.warning(f"Error extracting reasoning for block {i}: {e}")

        retriever_out.results += _parse_retriever_response(msg.content).results # type: ignore

    if reasons:
        retriever_out.reasoning = "\n".join(reasons)

    norm_questions = []
    for qs in retriever_out.results:
        original_question = qs.question
        matched_question = find_top_match(original_question, config.all_QS_info_dict.keys())
        qs.question = matched_question

        if qs.question != original_question:
            logger.debug(f"FUZZY-norm question: '{original_question}' -> '{qs.question}'")
        norm_questions.append(qs)

    retriever_out.results = norm_questions

    logger.info(f"Retriever completed with {len(retriever_out.results)} questions")

    config.update_context(retriever_out)

    return retriever_out

def _make_retriever_prompt(user_query: str) -> Template:
    return Template(f"""
**ЦЕЛЬ:** Найти набор вопросов необходимый для решения аналитической задачи пользователя.

ВАЖНО:
- При поиске смотри также на варианты ответов
- Не объединяй вопросы: каждый вопрос пиши отдельным пунктом
- Соблюдай формат ответа

**ЗАПРОС:** {user_query}

**ДОСТУПНЫЕ ВОПРОСЫ:**
$q_block

## ФОРМАТ ОТВЕТА:

**Анализ задачи:** [Краткое описание аналитической цели и требуемого хода исследования]

**Выбранные вопросы:**

1. **<ТОЧНАЯ КОПИЯ ВОПРОСА ИЗ СПИСКА>**
   Обоснование: [Зачем этот вопрос для решения задачи]
2. **<ДРУГОЙ ТОЧНЫЙ ВОПРОС ИЗ СПИСКА>**
   Обоснование: [Зачем этот вопрос для решения задачи]
...
""")

def _parse_retriever_response(analysis: str) -> RetrieverOut:
    logger.debug("Parsing retriever response")

    pattern = re.compile(
        r"""(?ms)
        \d+\.\s*
        (?:
            "(?P<q1>[^"\n]+)"
            |
            \*\*(?P<q2>.*?)\*\*
        )
        .*?
        Обоснование\s*:\s*
        (?P<reason>.+?)(?=\n|\Z)
        """, re.VERBOSE
    )

    scored_questions = []

    for i, match in enumerate(pattern.finditer(analysis), start=1):
        try:
            question_text = match.group("q1") or match.group("q2")
            reason_text = match.group('reason').strip()

            if not question_text:
                logger.warning(f"Parsed empty question for match {i}. Skipping.")
                continue
            if not reason_text:
                logger.warning(f"Parsed empty reason for question '{question_text}'. Skipping.")
                continue

            scored_questions.append(
                ScoredQuestion(
                    question=question_text,
                    reason=reason_text,
                )
            )
            logger.debug(f"Parsed question: '{question_text}'")
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to parse question from match {i}: {e}. Raw match: {match.group(0)}")
            continue

    if not scored_questions:
        logger.error(f"Failed to parse any questions from response. Full analysis: \n{analysis}")
        raise ValueError(
            "Не удалось распарсить ни одного вопроса из ответа LLM. "
            "Проверьте формат ответа модели."
        )

    logger.debug(f"Successfully parsed {len(scored_questions)} questions")
    return RetrieverOut(results=scored_questions)
