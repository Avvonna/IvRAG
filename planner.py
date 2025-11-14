import logging

from openai.types.shared import Reasoning

from config import PipelineConfig
from schemas import DreamerOut, PlannerOut
from utils import retry_call

logger = logging.getLogger(__name__)


def planner(
    user_query: str,
    dreamer_out: DreamerOut,
    config: PipelineConfig
) -> PlannerOut:
    logger.info(f"Starting planner for query: {user_query[:100]}...")

    pc = config.planner_config
    
    prompt = _make_planner_prompt(user_query, dreamer_out, config)
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
                raise ValueError("LLM вернул пустой ответ")
            
            if not plan.steps:
                raise ValueError("LLM вернул план без шагов")
            
            logger.debug(f"Plan received: {len(plan.steps)} steps")
            logger.debug(f"Plan analysis: {plan.analysis}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
    
    plan = retry_call(_call, retries=config.planner_config.retries, base_delay=config.planner_config.base_delay)
    
    logger.info(f"Planner completed: {len(plan.steps)} steps generated")
    logger.debug(f"Planner returned: {plan}")
    return plan


def _make_planner_prompt(
    user_query: str,
    dreamer_out: DreamerOut,
    config: PipelineConfig
) -> str:

    operations_spec = config.planner_config.capability_spec\
        .to_prompt_context("detailed", include_examples=True)

    prompt = f"""
# РОЛЬ
Ты — ПЛАНИРОВЩИК операций для системы анализа данных опросов.

# ЗАДАЧА
Составь оптимальную последовательность команд для выполнения шагов плана.

# ЗАПРОС ПОЛЬЗОВАТЕЛЯ
{user_query}

# ПЛАН АНАЛИЗА
{dreamer_out.analysis}

# ДОСТУПНЫЕ ВОПРОСЫ (для каждого указаны варианты ответов и волны, когда его задавали)
{config.relevant_as_value_catalog()}

# ДОСТУПНЫЕ ОПЕРАЦИИ
{operations_spec}

# КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ К ПЛАНУ
Обязательные правила:
1. Использовать ТОЛЬКО операции из спецификации — не придумывай новые операции
2. Использовать ТОЛЬКО доступные вопросы
3. Использовать ТОЛЬКО доступные ответы для каждого вопроса
4. Соблюдать типы данных — inputs/outputs должны соответствовать спецификации
5. Определить зависимости — если шаг использует output предыдущего, укажи в depends_on
6. Названия выводов операций делай осмысленными - результаты будут извлекаться из итогового контекста

ФОРМАТ ОТВЕТА
Верни план в формате JSON согласно схеме PlannerOut.
Если анализ нельзя выполнить с помощью данного тебе функционала - верни пустой план.

ВАЖНО: step IDs должны быть СТРОГО в формате:
- Первый шаг: "s1" 
- Второй шаг: "s2"
- Третий шаг: "s3"
- И так далее...

""".strip()

    return prompt
