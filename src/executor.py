import logging
from collections import defaultdict, deque
from typing import Any

from .grounder import GroundedStep
from .operations import OP_REGISTRY, GroundingError
from .schemas import GrounderOut

logger = logging.getLogger(__name__)


def executor(grounded_plan: GrounderOut, runtime_ctx: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """ Выполняет план операций с учётом зависимостей и топологической сортировки """
    logger.info(f"Starting executor with {len(grounded_plan.steps)} steps")
    
    # Валидация зависимостей и сортировка
    _validate_dependencies(grounded_plan.steps)
    sorted_steps = _topological_sort(grounded_plan.steps)
    
    # Структуры для отслеживания истории
    var_ancestors: dict[str, set[str]] = defaultdict(set)
    step_descriptions: dict[str, str] = {}

    # --- ЦИКЛ ВЫПОЛНЕНИЯ ШАГОВ ---
    for i, step in enumerate(sorted_steps):
        logger.info(f"Executing step {i+1}/{len(sorted_steps)}: {step.id} ({step.op_type.value})")
        step_descriptions[step.id] = f"[{step.op_type.value}] {step.goal}"

        # Сбор предков (provenance)
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
            # Материализация и выполнение
            kwargs = _materialize_inputs(step, runtime_ctx)
            func_impl = OP_REGISTRY.get(step.op_type)
            
            if not func_impl:
                raise ValueError(f"Operation {step.op_type} not found in registry during execution")
                
            # Вызываем функцию
            result = func_impl(**kwargs) 
            
            if not isinstance(result, dict):
                raise GroundingError(f"Шаг {step.id} вернул не dict, а {type(result)}")
            
            # Обновление контекста
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
                    output_keys = [plan_name]
                
                # Случай 2: Количество совпадает (N к N) -> Пытаемся мапить по порядку (редкий кейс)
                elif len(step.outputs) == len(result):
                    result_vals = list(result.values())
                    output_keys = []
                    for idx, plan_name in enumerate(step.outputs):
                        runtime_ctx[plan_name] = result_vals[idx]
                        output_keys.append(plan_name)
            
            # Привязываем историю ко всем выходам шага
            for key in output_keys:
                var_ancestors[key] = current_ancestors.copy()

            logger.info(f"Step {step.id} completed. Outputs: {output_keys}")

        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}", exc_info=True)
            raise GroundingError(f"Ошибка выполнения {step.id}: {e}") from e

    # --- СБОР РЕЗУЛЬТАТОВ ДЛЯ ПОЛЬЗОВАТЕЛЯ ---
    user_deliverables = {}
    
    # Берем список переменных, который LLM попросила вернуть
    targets = getattr(grounded_plan, "export_variables", [])
    
    logger.info(f"Collecting user deliverables: {targets}")
    
    for var_name in targets:
        if var_name in runtime_ctx:
            user_deliverables[var_name] = runtime_ctx[var_name]
        else:
            logger.warning(
                f"Переменная '{var_name}' была в списке export_variables, "
                "но не была найдена в контексте после выполнения плана."
            )

    # --- ФОРМИРОВАНИЕ ИСТОРИИ (PROVENANCE) ---
    final_provenance = {}
    for key in user_deliverables:
        ancestor_ids = var_ancestors.get(key, set())
        
        def sort_key(x):
            return int(x[1:]) if x.startswith('s') and x[1:].isdigit() else x
            
        sorted_ids = sorted(list(ancestor_ids), key=sort_key)
        
        history = []
        for sid in sorted_ids:
            if sid in step_descriptions:
                history.append(f"{sid}: {step_descriptions[sid]}")
        final_provenance[key] = history
    
    logger.info(f"Executor completed. Returning {len(user_deliverables)} items.")
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
