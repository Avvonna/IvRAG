import logging

from openai.types.shared_params import Reasoning

from config import PipelineConfig
from schemas import PlannerOut
from utils import retry_call

logger = logging.getLogger(__name__)


def planner(
    user_query: str,
    config: PipelineConfig
) -> PlannerOut:
    logger.info(f"Starting planner for query: {user_query}")

    pc = config.planner_config
    
    prompt = _make_planner_prompt(user_query, config)
    logger.debug(f"Generated prompt of length: {len(prompt)}")

    def _call() -> PlannerOut:
        """Вызов LLM с structured output"""
        logger.debug(f"Calling LLM with model: {pc.model}")
        
        try:           
            resp = config.client.responses.parse(
                model=pc.model,
                input=[{"role": "user", "content": prompt}],
                temperature=pc.temperature,
                reasoning=Reasoning(effort=pc.reasoning_effort),
                extra_body={"provider": {"sort": "latency"}},
                text_format=PlannerOut
            )
            plan = resp.output_parsed
            
            if not plan:
                logger.error("LLM вернул пустой ответ")
                return PlannerOut()
            
            if not plan.steps:
                logger.error("LLM вернул план без шагов")
                return PlannerOut()
            
            logger.debug(f"Plan received: {len(plan.steps)} steps")
            logger.debug(f"Plan analysis: {plan.analysis}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return PlannerOut()
    
    plan = retry_call(_call, retries=config.planner_config.retries, base_delay=config.planner_config.base_delay)
    
    logger.debug(f"Planner returned: {plan}")
    logger.info(f"Planner completed: {len(plan.steps)} steps generated")
    return plan


def _make_planner_prompt(
    user_query: str,
    config: PipelineConfig
) -> str:

    operations_spec = config.planner_config.capability_spec\
        .to_prompt_context("detailed", include_examples=True)

    prompt = f"""
# РОЛЬ
Ты — ПЛАНИРОВЩИК операций для системы анализа данных опросов.

# ЗАПРОС ПОЛЬЗОВАТЕЛЯ
{user_query}

# ДОСТУПНЫЕ ОПЕРАЦИИ
{operations_spec}

# ЗАДАЧА
Определи, возможно ли ответить на запрос пользователя с использованием доступных операций.
Если нет - верни план только с LOAD_DATA
Если да - составь оптимальную последовательность команд

# ДОСТУПНЫЕ ВОПРОСЫ (для каждого указаны варианты ответов и волны, когда его задавали)
{config.relevant_as_value_catalog()}

# КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ К ПЛАНУ
Обязательные правила:
1. Использовать ТОЛЬКО операции из спецификации
2. Использовать ТОЛЬКО доступные вопросы и ответы
3. Соблюдать типы данных — inputs/outputs должны соответствовать спецификации
4. Определить зависимости — если шаг использует output предыдущего, укажи в depends_on
5. Названия выводов операций делай осмысленными - результаты будут извлекаться из итогового контекста
6. Устанавливай `give_to_user: true` ТОЛЬКО для финальных шагов, которые содержат ответ на вопрос пользователя

ВАЖНО: step IDs должны быть СТРОГО в формате:
- Первый шаг: "s1" 
- Второй шаг: "s2"
- Третий шаг: "s3"
- И так далее...

""".strip()

    return prompt
