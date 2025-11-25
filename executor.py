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

    user_deliverables = {}
    
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
            
            logger.debug(f"Step {step.id} produced keys: {list(result.keys())}")
            for key, value in result.items():
                logger.debug(f"  {key}: {_safe_repr(value)}")
            
            # Обновление контекста
            runtime_ctx.update(result)
            
            # Если outputs указаны явно, создаём алиасы
            if step.outputs and len(result) == 1:
                only_val = list(result.values())[0]
                for out_name in step.outputs:
                    if out_name not in runtime_ctx:
                        runtime_ctx[out_name] = only_val
                        logger.debug(f"Created output alias: {out_name}")

            if step.give_to_user:
                logger.info(f"Step {step.id} marked for user delivery. Outputs: {step.outputs}")
                
                # Если outputs явно названы
                for out_name in step.outputs:
                    if out_name in runtime_ctx:
                        user_deliverables[out_name] = runtime_ctx[out_name]

                # Если outputs не были названы
                if not step.outputs:
                    for k, v in result.items():
                        user_deliverables[k] = v

            logger.info(f"Step {step.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}", exc_info=True)
            raise GroundingError(
                f"Ошибка выполнения шага {step.id} ({step.op_type.value}): {e}"
            ) from e
    
    logger.info(f"Executor completed. Returning {len(user_deliverables)} user deliverables.")
    return user_deliverables


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
    in_degree = defaultdict(int)    # количество входящих рёбер
    adj_list = defaultdict(list)    # список смежности
    
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

def _safe_repr(value: Any, max_len: int = 100) -> str:
    """Безопасное представление значения для логов"""
    try:
        if hasattr(value, 'shape'):
            # Для pandas/numpy объектов
            return f"{type(value).__name__}(shape={getattr(value, 'shape', 'unknown')})"
        elif hasattr(value, '__len__') and len(value) > 10:
            # Для больших коллекций
            return f"{type(value).__name__}(len={len(value)})"
        else:
            repr_str = repr(value)
            if len(repr_str) > max_len:
                return repr_str[:max_len] + "..."
            return repr_str
    except Exception:
        return f"{type(value).__name__}(<unrepresentable>)"

def _materialize_inputs(step: GroundedStep, ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Материализует входные параметры шага из контекста выполнения
    
    Правила материализации:
    1. Если значение - строка и она есть в контексте → берём значение из контекста
    2. Если значение - список строк → рекурсивно материализуем каждый элемент
    3. Иначе используем значение как есть (литерал)
    4. Constraints добавляются как параметры
    5. dataset добавляется автоматически, если есть в контексте
    
    Args:
        step: Шаг для выполнения
        ctx: Текущий контекст выполнения
        
    Returns:
        Словарь с готовыми аргументами для вызова операции
    """
    kwargs: dict[str, Any] = {}
    
    def _resolve_value(value: Any) -> Any:
        """Рекурсивно разрешает значение из контекста"""
        if isinstance(value, str) and value in ctx:
            # Разрешаем ссылку на переменную в контексте
            resolved = ctx[value]
            logger.debug(f"    resolved '{value}' -> {type(resolved).__name__}")
            return resolved
        elif isinstance(value, list):
            # Рекурсивно разрешаем каждый элемент списка
            return [_resolve_value(item) for item in value]
        else:
            # Используем как литеральное значение
            return value
    
    # Материализация явных входов
    for param_name, ref in (step.inputs or {}).items():
        resolved_value = _resolve_value(ref)
        kwargs[param_name] = resolved_value
        logger.debug(f"  {param_name} = {_safe_repr(resolved_value)}")
    
    # Материализация constraints
    for key, value in (step.constraints or {}).items():
        if key in kwargs:
            logger.warning(f"Constraint '{key}' conflicts with input, skipping")
            continue
            
        resolved_value = _resolve_value(value)
        kwargs[key] = resolved_value
        logger.debug(f"  {key} = {_safe_repr(resolved_value)} (constraint)")
    
    # Автоматическое добавление dataset если нужно
    if "dataset" in ctx and "dataset" not in kwargs:
        kwargs["dataset"] = ctx["dataset"]
        logger.debug("  dataset = ctx['dataset'] (auto)")
    
    return kwargs
