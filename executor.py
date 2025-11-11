import logging
from collections import defaultdict, deque
from typing import Any

from grounder import GroundedStep, GrounderOut
from operations import GroundingError

logger = logging.getLogger(__name__)


def executor(grounded_plan: GrounderOut, runtime_ctx: dict[str, Any]) -> dict[str, Any]:
    """ Выполняет план операций с учётом зависимостей и топологической сортировки """
    logger.info(f"Starting executor with {len(grounded_plan.steps)} steps")
    
    # Валидация зависимостей
    _validate_dependencies(grounded_plan.steps)
    
    # Топологическая сортировка шагов
    sorted_steps = _topological_sort(grounded_plan.steps)
    logger.info(f"Steps execution order: {[s.id for s in sorted_steps]}")
    
    # Выполнение шагов
    for i, step in enumerate(sorted_steps):
        logger.info(f"Executing step {i+1}/{len(sorted_steps)}: {step.id} ({step.op_type.value})")
        
        try:
            # Материализация входов из контекста
            kwargs = _materialize_inputs(step, runtime_ctx)
            logger.debug(f"Step {step.id} inputs: {list(kwargs.keys())}")
            
            # Выполнение операции
            result = step.impl(**kwargs)
            
            if not isinstance(result, dict):
                raise GroundingError(
                    f"Шаг {step.id} вернул не словарь: {type(result).__name__}. "
                    f"Ожидается dict[str, Any]"
                )
            
            logger.debug(f"Step {step.id} produced: {list(result.keys())}")
            
            # Обновление контекста
            runtime_ctx.update(result)
            
            # Если outputs указаны явно, создаём алиасы
            if step.outputs and len(result) == 1:
                only_val = list(result.values())[0]
                for out_name in step.outputs:
                    if out_name not in runtime_ctx:
                        runtime_ctx[out_name] = only_val
                        logger.debug(f"Created output alias: {out_name}")
            
            logger.info(f"Step {step.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}", exc_info=True)
            raise GroundingError(
                f"Ошибка выполнения шага {step.id} ({step.op_type.value}): {e}"
            ) from e
    
    logger.info("Executor completed successfully")
    return runtime_ctx


def _validate_dependencies(steps: list[GroundedStep]) -> None:
    """ Проверяет корректность зависимостей между шагами """
    logger.debug("Validating step dependencies")
    
    step_ids = {s.id for s in steps}
    
    # Проверка существования зависимостей
    for step in steps:
        for dep in step.depends_on or []:
            if dep not in step_ids:
                raise GroundingError(
                    f"Шаг '{step.id}' зависит от несуществующего шага '{dep}'"
                )
    
    # Проверка на циклы (используем топологическую сортировку)
    try:
        _topological_sort(steps)
        logger.debug("Dependencies validation passed")
    except GroundingError as e:
        raise GroundingError(f"Обнаружены циклические зависимости: {e}")


def _topological_sort(steps: list[GroundedStep]) -> list[GroundedStep]:
    """
    Сортирует шаги в порядке выполнения с учётом зависимостей (топологическая сортировка)
    """
    logger.debug("Performing topological sort")
    
    # Построение графа зависимостей
    step_map = {s.id: s for s in steps}
    in_degree = defaultdict(int)  # количество входящих рёбер
    adj_list = defaultdict(list)   # список смежности
    
    # Инициализация
    for step in steps:
        if step.id not in in_degree:
            in_degree[step.id] = 0
        
        for dep in step.depends_on or []:
            adj_list[dep].append(step.id)
            in_degree[step.id] += 1
    
    # Очередь шагов без зависимостей
    queue = deque([sid for sid in step_map.keys() if in_degree[sid] == 0])
    sorted_steps = []
    
    # Алгоритм Кана
    while queue:
        current_id = queue.popleft()
        sorted_steps.append(step_map[current_id])
        
        for next_id in adj_list[current_id]:
            in_degree[next_id] -= 1
            if in_degree[next_id] == 0:
                queue.append(next_id)
    
    if len(sorted_steps) != len(steps):
        remaining = set(step_map.keys()) - {s.id for s in sorted_steps}
        raise GroundingError(
            f"Обнаружены циклические зависимости. "
            f"Невозможно обработать шаги: {remaining}"
        )
    
    logger.debug(f"Topological sort completed: {[s.id for s in sorted_steps]}")
    return sorted_steps


def _materialize_inputs(step: GroundedStep, ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Материализует входные параметры шага из контекста выполнения
    
    Правила материализации:
    1. Если значение - строка и она есть в контексте → берём значение из контекста
    2. Иначе используем значение как есть (литерал)
    3. Constraints добавляются как параметры
    4. dataset добавляется автоматически, если есть в контексте
    
    Args:
        step: Шаг для выполнения
        ctx: Текущий контекст выполнения
        
    Returns:
        Словарь с готовыми аргументами для вызова операции
    """
    kwargs: dict[str, Any] = {}
    
    # Материализация явных входов
    for param_name, ref in (step.inputs or {}).items():
        if isinstance(ref, str) and ref in ctx:
            # Разрешаем ссылку на переменную в контексте
            kwargs[param_name] = ctx[ref]
            logger.debug(f"  {param_name} = ctx['{ref}'] ({type(ctx[ref]).__name__})")
        else:
            # Используем как литеральное значение
            kwargs[param_name] = ref
            logger.debug(f"  {param_name} = {ref!r} (literal)")
    
    # Материализация constraints
    for key, value in (step.constraints or {}).items():
        if key in kwargs:
            logger.warning(f"Constraint '{key}' conflicts with input, skipping")
            continue
            
        if isinstance(value, str) and value in ctx:
            kwargs[key] = ctx[value]
            logger.debug(f"  {key} = ctx['{value}'] (constraint)")
        else:
            kwargs[key] = value
            logger.debug(f"  {key} = {value!r} (constraint literal)")
    
    # Автоматическое добавление dataset если нужно
    if "dataset" in ctx and "dataset" not in kwargs:
        kwargs["dataset"] = ctx["dataset"]
        logger.debug("  dataset = ctx['dataset'] (auto)")
    
    return kwargs
