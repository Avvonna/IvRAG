import json
import logging

from capability_spec import CapabilitySpec
from catalog import QuestionCatalog
from config import PipelineConfig
from schemas import PlannerOut, RetrieverOut
from utils import retry_call

logger = logging.getLogger(__name__)


def planner(
    user_query: str,
    retriever_out: RetrieverOut,
    config: PipelineConfig
) -> PlannerOut:
    logger.info(f"Starting planner for query: {user_query[:100]}...")
    
    plc = config.planner_config

    # Фильтруем вопросы
    chosen_clean_questions_list = retriever_out.clean_list()
    chosen_questions_catalog = QuestionCatalog(
        questions=[
            q for q in config.catalog.questions 
            if q.id in chosen_clean_questions_list
        ]
    )
    
    logger.debug(f"Working with {len(chosen_questions_catalog.questions)} questions")
    
    # Подмешиваем их в контекст
    context = {
        "allowed_questions": chosen_questions_catalog.as_value_catalog(),
        "dataset_schema": config.df_schema,
    }

    prompt = _make_planner_prompt(user_query, context, plc.capability_spec)
    logger.debug(f"Generated prompt of length: {len(prompt)}")

    def _call():
        logger.debug(f"Calling LLM with model: {plc.model}")
        resp = config.client.chat.completions.create(
            model=plc.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=plc.temperature,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        assert content, "Empty planner response"
        return content.strip()

    response_text = retry_call(_call, retries=plc.retries, base_delay=plc.base_delay)
    logger.debug(f"LLM response length: {len(response_text)}")
    
    try:
        plan = PlannerOut.model_validate_json(response_text)
        logger.info(f"Successfully created plan with {len(plan.steps)} steps")
        return plan
    except Exception as e:
        logger.error(f"Failed to parse planner response: {e}")
        logger.debug(f"Response text: {response_text[:1000]}")
        raise ValueError(f"Не удалось распарсить план: {e}")


def _make_planner_prompt(
    user_query: str,
    context: dict,
    capability_spec: CapabilitySpec
) -> str:
    context_json = json.dumps(context or {}, ensure_ascii=False)

    prompt = f"""
Роль: Ты — ПЛАНИРОВЩИК (tool-agnostic). Составь абстрактный план как DAG шагов для решения задачи.

Цель:
- user_query: {user_query}
- context: {context_json}

{capability_spec.to_prompt_context("detailed")}

ВАЖНО:
- Используй ТОЛЬКО операции из CapabilitySpec.
- Используй ТОЛЬКО вопросы из allowed_questions. НЕ ПРИДУМЫВАЙ новые.
- Оперируй ТОЛЬКО колонками из dataset_schema. Если операция требует поля, которого нет, не используй её.
- Если для решения требуется вопрос, которого нет в allowed_questions, верни план из одного шага CHECK_DATA_AVAILABILITY
  с понятным сообщением об отсутствии вопроса и не пытайся подменять его.

Правила построения плана:
1) До 10 шагов (если не указано иное в лимитах). id шагов: s1, s2, …
2) У каждого шага: {{ "id", "goal", "operation", "inputs", "outputs", "constraints", "depends_on"(опц.) }}.
3) Все inputs должны быть доступны из user_query/контекста или outputs предыдущих шагов.

Ответ ТОЛЬКО ЧИСТЫМ JSON (без Markdown, без комментариев).

Формат ответа (СТРОГО):
{{
  "analysis": "1–2 предложения о стратегии",
  "steps": [
    {{"id":"s1","goal":"...","operation":"LOAD_WAVE_DATA","inputs":[],"outputs":["dataset"],"constraints":{{"wave_id":"<wave_id>"}}}}
  ]
}}
""".strip()
    return prompt