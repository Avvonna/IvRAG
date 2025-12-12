import logging

from .capability_spec import OperationType
from .operations import OP_REGISTRY, GroundingError
from .schemas import GroundedStep, GrounderOut, PlannerOut

logger = logging.getLogger(__name__)

def grounder(plan: PlannerOut) -> GrounderOut:
    logger.info("Starting grounder")
    
    steps_in = plan.steps
    
    if not steps_in:
        logger.warning("Empty plan received")
        return GrounderOut(steps=[])

    grounded_steps: list[GroundedStep] = []
    seen_ids: set[str] = set()

    for i, s in enumerate(steps_in):
        sid = s.id
        if not sid:
            raise GroundingError(f"Шаг {i} не имеет ID")
        if sid in seen_ids:
            raise GroundingError(f"Дубликат ID шага: {sid}")
        seen_ids.add(sid)

        op_type: OperationType = s.operation
        
        # ВАЛИДАЦИЯ: Проверяем, есть ли такая функция, но НЕ сохраняем её
        if op_type not in OP_REGISTRY:
            logger.error(f"No implementation for operation: {op_type.value}")
            raise GroundingError(f"Нет реализации для операции: {op_type.value}")

        # Тут можно добавить валидацию inputs (соответствует ли сигнатуре функции)
        # ...

        # Создаем Pydantic модель
        grounded_steps.append(GroundedStep(
            id=sid,
            goal=s.goal or "",
            op_type=op_type,
            inputs=s.inputs or {},
            outputs=s.outputs or [],
            depends_on=s.depends_on or [],
        ))

    exports = getattr(plan, "export_variables", [])
    logger.info(f"Grounder completed: {len(grounded_steps)} steps ready. Exports: {exports}")

    return GrounderOut(steps=grounded_steps, export_variables=exports)