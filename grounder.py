import logging
from dataclasses import dataclass
from typing import Any, Callable

from capability_spec import OperationType
from config import PipelineConfig
from operations import OP_REGISTRY, GroundingError
from schemas import PlannerOut

logger = logging.getLogger(__name__)


@dataclass
class GroundedStep:
    """Шаг плана с привязанной реализацией"""
    id: str
    goal: str
    op_type: OperationType
    impl: Callable[..., dict[str, Any]]
    inputs: dict[str, Any]
    outputs: list[str]
    constraints: dict[str, Any]
    depends_on: list[str]


@dataclass
class GrounderOut:
    """Результат работы grounder"""
    analysis: str
    steps: list[GroundedStep]


def grounder(plan: PlannerOut, config: PipelineConfig) -> GrounderOut:
    logger.info("Starting grounder")
    
    analysis = plan.analysis
    steps_in = plan.steps
    
    if not isinstance(steps_in, list) or not steps_in:
        raise GroundingError("Пустой список шагов")

    grounded: list[GroundedStep] = []
    seen: set[str] = set()

    for i, s in enumerate(steps_in):
        logger.debug(f"Grounding step {i+1}/{len(steps_in)}: {s.id}")
        
        sid = s.id
        if not sid or sid in seen:
            raise GroundingError(f"Проблема с id шага: {sid}")
        seen.add(sid)

        op_type: OperationType = s.operation
        impl = OP_REGISTRY.get(op_type)
        
        if not impl:
            logger.error(f"No implementation for operation: {op_type.value}")
            raise GroundingError(f"Нет реализации для операции: {op_type.value}")

        inputs = s.inputs or {}
        if isinstance(inputs, list):
            logger.warning(f"Step {sid}: inputs is list, converting to empty dict")
            inputs = {}

        grounded.append(GroundedStep(
            id=sid,
            goal=s.goal or "",
            op_type=op_type,
            impl=impl,
            inputs=inputs,
            outputs=list(s.outputs or []),
            constraints=dict(s.constraints or {}),
            depends_on=list(s.depends_on or []),
        ))
        
        logger.debug(f"Successfully grounded step {sid}: {op_type.value}")

    logger.info(f"Grounder completed: {len(grounded)} steps grounded")
    return GrounderOut(analysis=analysis, steps=grounded)