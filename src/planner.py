import logging

from .config import PipelineConfig
from .schemas import PlannerOut
from .utils import retry_call

logger = logging.getLogger(__name__)


def planner(
    user_query: str,
    config: PipelineConfig
) -> PlannerOut:
    logger.info(f"Starting planner for query: {user_query}")

    pc = config.planner_config
    
    prompt = _make_planner_prompt(user_query, config)
    logger.debug(f"Generated prompt of length: {len(prompt)}")

    params = {
        "model": pc.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": pc.temperature,
        "response_format": PlannerOut,
    }

    if pc.max_tokens:
        params["max_tokens"] = pc.max_tokens
    if pc.reasoning_effort:
        params["reasoning_effort"] = pc.reasoning_effort

    extra_body = {}
    if pc.provider_sort:
        extra_body["provider"] = {"sort": pc.provider_sort}
    if extra_body:
        params["extra_body"] = extra_body

    def _call() -> PlannerOut:
        """Вызов LLM с structured output"""
        logger.debug(f"Calling LLM with model: {pc.model}")

        resp = config.client.chat.completions.parse(**params)
        plan = resp.choices[0].message.parsed

        if not plan or not plan.steps:
            raise ValueError("LLM returned empty plan or no steps")

        # logger.debug(f"Plan:\n{plan}")

        return plan

    
    plan = retry_call(_call, retries=config.planner_config.retries, base_delay=config.planner_config.base_delay)
    
    logger.debug(f"Planner returned:\n{plan}")
    return plan


def _make_planner_prompt(
    user_query: str,
    config: PipelineConfig
) -> str:

    operations_spec = config.planner_config.capability_spec\
        .to_prompt_context("detailed", include_examples=True)

    prompt = f"""
# РОЛЬ
Ты — ПЛАНИРОВЩИК для системы анализа данных. Твоя цель — преобразовать запрос пользователя в последовательность исполняемых шагов (план).

# СПЕЦИФИКАЦИЯ ОПЕРАЦИЙ
{operations_spec}

# КАТАЛОГ ДАННЫХ (Доступные вопросы и ответы)
{config.relevant_as_value_catalog()}

# ЗАПРОС ПОЛЬЗОВАТЕЛЯ
"{user_query}"

# ИНСТРУКЦИЯ ПО ПОТОКУ ДАННЫХ (DATA FLOW)
1. Если Шаг A производит данные (output), придумай ему уникальное имя переменной.
2. Если Шаг B должен использовать эти данные, передай это имя в его inputs.
3. Соблюдай префиксы: `df_` для таблиц, `res_` для результатов вычислений.

# ФОРМАТ ШАГОВ
1. **id**: "s1", "s2"...
2. **depends_on**: ID предыдущих шагов.
3. **inputs**: Аргументы операции.
4. **outputs**: Список имен переменных, которые создаст этот шаг.

# ФИНАЛЬНЫЙ ВОЗВРАТ (Export Variables)
В поле `export_variables` перечисли список имен переменных (из `outputs` твоих шагов), которые являются ответом на вопрос пользователя и должны попасть в итоговый отчет (Excel).
Не включай туда промежуточные переменные (например, промежуточные результаты фильтрации), если они не нужны пользователю.

# ПРИМЕР ПЛАНА
User: "Посчитай NPS для женщин"
Plan:
steps:
  - id: "s1", op: "LOAD_DATA", outputs: ["df_raw"]
  - id: "s2", op: "FILTER", inputs: {{dataset: "df_raw", ...}}, outputs: ["df_women"]
  - id: "s3", op: "CALC_NPS", inputs: {{dataset: "df_women"}}, outputs: ["res_nps_women"]
export_variables: ["res_nps_women"]

# ТВОЯ ЗАДАЧА
Составь план и список переменных для экспорта.
""".strip()

    return prompt