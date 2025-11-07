from typing import Any, Literal

from capability_spec import OperationType
from pydantic import BaseModel, Field


class QuestionInfo(BaseModel):
    id: str
    waves: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)

    def stringify(
        self,
        include: list[Literal["q_clean", "answers", "details", "options", "waves"]] = [
            "q_clean",
            "answers",
            "details",
            "options"
        ]
    ) -> str:
        parts = [f"{self.id}"]
        if "answers" in include and self.answers:
            parts.append(f"\n\tAnswers: {self.answers}")
        if "details" in include and self.details:
            parts.append(f"\n\tDetails: {self.details}")
        if "options" in include and self.options:
            parts.append(f"\n\tOptions: {self.options}")
        if "waves" in include and self.waves:
            parts.append(f"\n\tWaves: {self.waves}")
        return "".join(parts)


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