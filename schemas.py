from typing import Any

from pydantic import BaseModel, Field

from capability_spec import OperationType


class QuestionInfo(BaseModel):
    id: str
    waves: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)


class ScoredQuestion(BaseModel):
    question: str = Field(..., description="Точная формулировка вопроса из базы")
    reason: str = Field(..., description="Почему этот вопрос полезен для ответа на запрос")
    relevance: float = Field(..., description="Оценка релевантности 0–100")


class RetrieverOut(BaseModel):
    results: list[ScoredQuestion] = Field(
        default_factory=list,
        description="Список релевантных вопросов с объяснениями и оценками"
    )

    def clean_list(self):
        return [q.question for q in self.results]
    
    def __str__(self):
        lines = []
        for i, sq in enumerate(self.results, 1):
            lines.append(f"{i}. [{sq.relevance:.0f}/100] {sq.question}")
            lines.append(f"\tReason: {sq.reason}")
        return "\n".join(lines)


class PlanStep(BaseModel):
    id: str = Field(..., description="Уникальный идентификатор шага (s1, s2, ...)")
    goal: str = Field("", description="Человекочитаемая цель шага")
    operation: OperationType = Field(..., description="Тип операции из OperationType")
    inputs: dict[str, Any] | list | None = Field(
        default_factory=dict,
        description="Именованные входы ИЛИ []"
    )
    outputs: list[str] | None = Field(
        default_factory=list,
        description="Имена выходов, которые появятся в контексте"
    )
    constraints: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Параметры операции"
    )
    depends_on: list[str] | None = Field(
        default_factory=list,
        description="ID шагов, от которых зависит текущий"
    )


class PlannerOut(BaseModel):
    analysis: str = Field("", description="Короткий комментарий стратегии")
    steps: list[PlanStep]

    def __str__(self):
        res = []
        res.append(f"АНАЛИЗ: {self.analysis}")
        for i, s in enumerate(self.steps):
            res.append(f"{i}. [{s.id}] {s.operation}")
            res.append(f"\tGoal: {s.goal}")
            res.append(f"\tInputs: {s.inputs}")
            res.append(f"\tOutputs: {s.outputs}")
            res.append(f"\tConstraints: {s.constraints}")
            res.append(f"\tDepends on: {s.depends_on}")
        return "\n".join(res)