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
    """
    Составляет план выполнения для решения пользовательского запроса
    
    Args:
        user_query: Запрос пользователя
        retriever_out: Результаты работы retriever с релевантными вопросами
        config: Конфигурация pipeline
        
    Returns:
        План выполнения в виде последовательности шагов
    """
    logger.info(f"Starting planner for query: {user_query[:100]}...")
    
    plc = config.planner_config

    # Фильтруем каталог вопросов по результатам retriever
    chosen_clean_questions_list = retriever_out.clean_list()
    chosen_questions_catalog = QuestionCatalog(
        questions=[
            q for q in config.catalog.questions 
            if q.id in chosen_clean_questions_list
        ]
    )
    
    logger.debug(f"Working with {len(chosen_questions_catalog.questions)} questions")
    
    # Формируем контекст для LLM
    context = {
        "allowed_questions": chosen_questions_catalog.as_value_catalog(),
        "dataset_schema": config.df_schema,
    }

    prompt = _make_planner_prompt(user_query, context, plc.capability_spec)
    logger.debug(f"Generated prompt of length: {len(prompt)}")

    def _call() -> PlannerOut:
        """Вызов LLM с structured output"""
        logger.debug(f"Calling LLM with model: {plc.model}")
        
        try:
            # Используем structured output через beta API
            resp = config.client.beta.chat.completions.parse(
                model=plc.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=plc.temperature,
                response_format=PlannerOut
            )
            
            # Извлечение распарсенного ответа
            plan = resp.choices[0].message.parsed
            
            if not plan:
                raise ValueError("LLM вернул пустой ответ")
            
            if not plan.steps:
                raise ValueError("LLM вернул план без шагов")
            
            logger.debug(f"Plan received: {len(plan.steps)} steps")
            logger.debug(f"Plan analysis: {plan.analysis}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
    
    plan = retry_call(_call, retries=plc.retries, base_delay=plc.base_delay)
    
    logger.info(f"Planner completed: {len(plan.steps)} steps generated")
    logger.debug(f"Planner returned: {plan}")
    return plan


def _make_planner_prompt(user_query: str, context: dict, capability_spec: CapabilitySpec) -> str:
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    operations_spec = capability_spec.to_prompt_context("detailed", include_examples=True)

    prompt = f"""
# РОЛЬ
Ты — ПЛАНИРОВЩИК операций для системы анализа данных опросов.

# ЗАДАЧА
Составь оптимальный план выполнения для решения запроса пользователя.

# ЗАПРОС ПОЛЬЗОВАТЕЛЯ
{user_query}

# ДОСТУПНЫЙ КОНТЕКСТ
{context_json}

# ДОСТУПНЫЕ ОПЕРАЦИИ
{operations_spec}

# КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ К ПЛАНУ
Обязательные правила:
1. Использовать ТОЛЬКО операции из спецификации — не придумывай новые операции
2. Использовать ТОЛЬКО вопросы из allowed_questions — не меняй формулировки
3. ИСПОЛЬЗОВАТЬ ТОЛЬКО ДОСТУПНЫЕ ANSWER_VALUES — каждый answer_value ДОЛЖЕН существовать в списке answers для указанного question
4. ОБЯЗАТЕЛЬНАЯ ПРОВЕРКА — перед указанием answer_values проверь, что они есть в context_json
5. Соблюдать типы данных — inputs/outputs должны соответствовать спецификации
6. Определить зависимости — если шаг использует output предыдущего, укажи в depends_on

ФОРМАТ ОТВЕТА
Верни план в формате JSON согласно схеме PlannerOut.
""".strip()

    return prompt
