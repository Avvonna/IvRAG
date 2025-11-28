import logging
from collections import defaultdict, deque
from typing import Any

from .grounder import GroundedStep, GrounderOut
from .operations import GroundingError

logger = logging.getLogger(__name__)


def executor(grounded_plan: GrounderOut, runtime_ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """ Выполняет план операций с учётом зависимостей и топологической сортировки """
    logger.info(f"Starting executor with {len(grounded_plan.steps)} steps")
    
    # Валидация зависимостей
    _validate_dependencies(grounded_plan.steps)
    
    # Топологическая сортировка шагов
    sorted_steps = _topological_sort(grounded_plan.steps)
    logger.info(f"Steps execution order: {[s.id for s in sorted_steps]}")

    user_deliverables = {}
    
    # Структуры для отслеживания истории данных
    var_ancestors: dict[str, set[str]] = defaultdict(set)
    step_descriptions: dict[str, str] = {}

    # Выполнение шагов
    for i, step in enumerate(sorted_steps):
        logger.info(f"Executing step {i+1}/{len(sorted_steps)}: {step.id} ({step.op_type.value})")

        # Сохраняем описание для истории
        step_descriptions[step.id] = f"[{step.op_type.value}] {step.goal}"

        current_ancestors = set()
        for inp_val in (step.inputs or {}).values():
            if isinstance(inp_val, str) and inp_val in var_ancestors:
                current_ancestors.update(var_ancestors[inp_val])
            elif isinstance(inp_val, list):
                for item in inp_val:
                    if isinstance(item, str) and item in var_ancestors:
                        current_ancestors.update(var_ancestors[item])

        current_ancestors.add(step.id)
        
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
            
            # Сохраняем реальные результаты в контекст
            runtime_ctx.update(result)

            # По умолчанию используем ключи, которые вернула функция
            output_keys = list(result.keys())

            # Если outputs указаны явно, создаём алиасы
            if step.outputs:
                # Случай 1: 1 выход в плане, 1 выход по факту -> Алиас
                if len(result) == 1:
                    plan_name = step.outputs[0]
                    val = list(result.values())[0]
                    if plan_name not in runtime_ctx:
                        runtime_ctx[plan_name] = val
                        logger.debug(f"Created output alias: {plan_name} -> {type(val).__name__}")
                    output_keys = [plan_name]
                
                # Случай 2: Количество совпадает (N к N) -> Пытаемся мапить по порядку (редкий кейс)
                elif len(step.outputs) == len(result):
                    result_vals = list(result.values())
                    new_keys = []
                    for idx, plan_name in enumerate(step.outputs):
                        runtime_ctx[plan_name] = result_vals[idx]
                        new_keys.append(plan_name)
                    output_keys = new_keys
            
            # Привязываем историю ко всем выходным переменным (и реальным, и алиасам)
            for key in output_keys:
                var_ancestors[key] = current_ancestors.copy()

            if step.give_to_user:
                logger.info(f"Step {step.id} marked for user delivery. Outputs: {output_keys}")
                for out_name in output_keys:
                    # Важно брать из runtime_ctx, т.к. output_keys теперь точно там есть
                    if out_name in runtime_ctx:
                        user_deliverables[out_name] = runtime_ctx[out_name]

            logger.info(f"Step {step.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}", exc_info=True)
            raise GroundingError(
                f"Ошибка выполнения шага {step.id} ({step.op_type.value}): {e}"
            ) from e
    
    # Формирование читаемой истории для финальных результатов
    final_provenance = {}
    for key in user_deliverables:
        ancestor_ids = var_ancestors.get(key, set())
        
        # Сортировка ID
        def sort_key(x):
            if x.startswith('s') and x[1:].isdigit():
                return int(x[1:])
            return x
            
        sorted_ids = sorted(list(ancestor_ids), key=sort_key)
        
        history = []
        for sid in sorted_ids:
            if sid in step_descriptions:
                history.append(f"{sid}: {step_descriptions[sid]}")
        final_provenance[key] = history
    
    logger.info(f"Executor completed. Returning {len(user_deliverables)} deliverables with provenance.")
    return user_deliverables, final_provenance


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
            return f"{type(value).__name__}(shape={getattr(value, 'shape', 'unknown')})"
        elif hasattr(value, '__len__') and len(value) > 10:
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
    Материализует входные параметры шага из контекста выполнения.
    Требует, чтобы исходный датафрейм был в ctx под ключом 'dataset'.
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
    
    # Автоматическое добавление dataset если нужно
    if "dataset" in ctx and "dataset" not in kwargs:
        kwargs["dataset"] = ctx["dataset"]
        logger.debug("  dataset = ctx['dataset'] (auto)")
    
    return kwargs
