import json
import logging

from capability_spec import CapabilitySpec
from catalog import QuestionCatalog
from config import PipelineConfig
from schemas import PlannerOut, RetrieverOut
from utils import retry_call, remove_defs_and_refs

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
        logger.debug(f"Prompt: {prompt}")
        resp = config.client.responses.parse(
            model=plc.model,
            input=[{"role": "user", "content": prompt}],
            temperature=plc.temperature,
            text_format=    # TODO: ???
        )
        plan = resp.output_parsed
        logger.debug(f"PLAN: {plan}")
        assert plan, "Empty planner response"

        return plan
    
    return retry_call(_call, retries=plc.retries, base_delay=plc.base_delay)

def _make_planner_prompt(
    user_query: str,
    context: dict,
    capability_spec: CapabilitySpec
) -> str:
    context_json = json.dumps(context or {}, ensure_ascii=False)

    prompt = f"""
Роль: Ты — ПЛАНИРОВЩИК. Составь абстрактный план как DAG шагов для решения задачи.

Цель:
- user_query: {user_query}
- context: {context_json}

{capability_spec.to_prompt_context("detailed")}

ВАЖНО:
- Используй ТОЛЬКО операции из CapabilitySpec в СТРОГОМ СООТВЕТСТВИИ с описанием.
- Используй ТОЛЬКО вопросы и ответы на них из allowed_questions. НЕ ПРИДУМЫВАЙ новые, НЕ МЕНЯЙ ФОРМУЛИРОВКИ.
- Используй ТОЛЬКО необходимые тебе вопросы из доступных

Правила построения плана:
1) До 5 шагов (если не указано иное в лимитах). id шагов: s1, s2, …
2) У каждого шага: {{ "id", "goal", "operation", "inputs", "outputs", "constraints", "depends_on"(опц.) }}.
3) Все inputs должны быть доступны из user_query/контекста или outputs предыдущих шагов.
""".strip()
    return prompt