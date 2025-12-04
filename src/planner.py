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
    
    logger.debug(f"Planner returned: {plan}")
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

# ИНСТРУКЦИЯ ПО ПОТОКУ ДАННЫХ (DATA FLOW) — ЭТО КРИТИЧНО!
Ты должен явно передавать данные между шагами. Система НЕ передает контекст магическим образом.
1. Если Шаг A производит данные (output), придумай ему уникальное имя (переменную).
2. Если Шаг B должен использовать эти данные, передай это имя в его inputs.

# ПРАВИЛА ИМЕНОВАНИЯ ПЕРЕМЕННЫХ
Чтобы система не перепутала переменную с обычным текстом, следуй этим префиксам в `outputs`:
- Для таблиц данных (DataFrames): используй префикс `df_` (например: `df_filtered`, `df_women`).
- Для числовых метрик/результатов: используй префикс `res_` (например: `res_nps`, `res_chart`).

# ФОРМАТ ШАГОВ
1. **id**: Строго "s1", "s2", "s3"...
2. **depends_on**: Список ID шагов, которые должны выполниться ДО текущего.
3. **inputs**: Словарь аргументов. Если значение — это результат предыдущего шага, пиши имя переменной (например, "df_filtered"). Если это текстовая константа (например, "Москва"), пиши как есть.
4. **give_to_user**: Ставь `true` ТОЛЬКО для тех шагов, результат которых является ОТВЕТОМ на вопрос пользователя. Промежуточные шаги (фильтрация, загрузка) — `false`.

# ПРИМЕР ПЛАНА (Thinking Process)
User: "Посчитай NPS для женщин"
Plan:
- s1 [LOAD_DATA]: Загружаем данные. Output: ["df_raw"]
- s2 [FILTER]: Берем "df_raw", оставляем gender='Female'. Output: ["df_women"]. Depends on: ["s1"]
- s3 [CALC_NPS]: Берем "df_women". Считаем метрику. Output: ["res_nps_women"]. Give to user: True. Depends on: ["s2"]

# ТВОЯ ЗАДАЧА
Составь план для текущего запроса.
- Если запрос невыполним с данными операциями -> верни пустой список шагов или только LOAD_DATA.
- Используй ТОЛЬКО доступные операции.
- В inputs используй точные значения из Каталога Данных (если применимо).

""".strip()

    return prompt
