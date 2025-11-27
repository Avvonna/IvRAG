import logging
from dataclasses import dataclass
from typing import Any, Callable

from capability_spec import OperationType
from operations import OP_REGISTRY, GroundingError
from schemas import PlannerOut

logger = logging.getLogger(__name__)


@dataclass
class GroundedStep:
    """Шаг плана с привязанной реализацией"""
    id: str
    goal: str
    op_type: OperationType
    impl: Callable[..., Any]
    inputs: dict[str, str]
    outputs: list[str]
    depends_on: list[str]
    give_to_user: bool


@dataclass
class GrounderOut:
    """Результат работы grounder"""
    steps: list[GroundedStep]

    def __str__(self):
        res = []
        for i, s in enumerate(self.steps):
            res.append(f"{i}. [{s.id}] {s.op_type}")
            res.append(f"\tGoal: {s.goal}")
            res.append(f"\tInputs: {s.inputs}")
            res.append(f"\tOutputs: {s.outputs}")
        return "\n".join(res)


def grounder(plan: PlannerOut) -> GrounderOut:
    logger.info("Starting grounder")
    
    steps_in = plan.steps
    
    if not isinstance(steps_in, list) or not steps_in:
        logger.warning("Empty plan received")
        return GrounderOut(steps=[])

    grounded: list[GroundedStep] = []
    seen_ids: set[str] = set()

    for i, s in enumerate(steps_in):
        sid = s.id
        if not sid:
            raise GroundingError(f"Шаг {i} не имеет ID")
        if sid in seen_ids:
            raise GroundingError(f"Дубликат ID шага: {sid}")
        seen_ids.add(sid)

        op_type: OperationType = s.operation
        impl = OP_REGISTRY.get(op_type)
        
        if not impl:
            logger.error(f"No implementation for operation: {op_type.value}")
            raise GroundingError(f"Нет реализации для операции: {op_type.value}")

        inputs_map = {}
        if isinstance(s.inputs, dict):
            inputs_map = s.inputs
        elif s.inputs is None:
            inputs_map = {}
        else:
            logger.warning(f"Step {sid}: inputs expected as dict, got {type(s.inputs)}. Ignoring inputs.")
            inputs_map = {}

        grounded.append(GroundedStep(
            id=sid,
            goal=s.goal or "",
            op_type=op_type,
            impl=impl,
            inputs=inputs_map,
            outputs=list(s.outputs or []),
            depends_on=list(s.depends_on or []),
            give_to_user=s.give_to_user
        ))
        
    logger.info(f"Grounder completed: {len(grounded)} steps ready")
    return GrounderOut(steps=grounded)