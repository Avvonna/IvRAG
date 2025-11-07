import logging
from typing import Any

from grounder import GroundedStep, GrounderOut
from operations import GroundingError

logger = logging.getLogger(__name__)


def executor(grounded_plan: GrounderOut, runtime_ctx: dict[str, Any]) -> dict[str, Any]:
    logger.info(f"Starting executor with {len(grounded_plan.steps)} steps")
    
    for i, st in enumerate(grounded_plan.steps):
        logger.info(f"Executing step {i+1}/{len(grounded_plan.steps)}: {st.id} ({st.op_type.value})")
        
        # Проверка зависимостей
        for dep in st.depends_on or []:
            if dep not in [x.id for x in grounded_plan.steps]:
                raise GroundingError(f"Неизвестная зависимость: {dep}")
        
        # Материализация входов
        kwargs = _materialize_inputs(st, runtime_ctx)
        logger.debug(f"Step {st.id} inputs: {list(kwargs.keys())}")
        
        # Выполнение операции
        try:
            result = st.impl(**kwargs)
        except Exception as e:
            logger.error(f"Step {st.id} failed: {e}")
            raise GroundingError(f"Шаг {st.id} завершился с ошибкой: {e}")
        
        if not isinstance(result, dict):
            raise GroundingError(f"Шаг {st.id} вернул не словарь: {type(result)}")
        
        logger.debug(f"Step {st.id} produced: {list(result.keys())}")
        
        # Обновление контекста
        runtime_ctx.update(result)
        
        # Если outputs указаны явно и результат один, копируем его под все имена
        if st.outputs and len(result) == 1:
            only_val = list(result.values())[0]
            for out_name in st.outputs:
                runtime_ctx.setdefault(out_name, only_val)
                logger.debug(f"Set output alias: {out_name}")
    
    logger.info("Executor completed successfully")
    return runtime_ctx


def _materialize_inputs(step: GroundedStep, ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Материализует входы шага из контекста
    
    Args:
        step: Шаг для выполнения
        ctx: Текущий контекст выполнения
        
    Returns:
        dict с готовыми аргументами для операции
    """
    logger = logging.getLogger(__name__)
    kwargs: dict[str, Any] = {}

    # Разрешаем ссылки на переменные в контексте
    for param_name, ref in (step.inputs or {}).items():
        # Если ref - это строка и она есть в контексте, берём значение
        if isinstance(ref, str) and ref in ctx:
            kwargs[param_name] = ctx[ref]
            logger.debug(f"Resolved {param_name}='{ref}' from context")
        else:
            # Иначе используем как есть
            kwargs[param_name] = ref
            logger.debug(f"Using literal {param_name}={ref}")

    # Добавляем constraints как параметры
    for key, value in (step.constraints or {}).items():
        # Constraints тоже могут содержать ссылки на контекст
        if isinstance(value, str) and value in ctx:
            kwargs[key] = ctx[value]
            logger.debug(f"Resolved constraint {key}='{value}' from context")
        else:
            kwargs[key] = value

    # df_full нужен почти всегда
    if "df_full" in ctx and "df_full" not in kwargs:
        kwargs["df_full"] = ctx["df_full"]

    return kwargs