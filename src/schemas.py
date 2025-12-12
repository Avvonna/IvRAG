from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from .capability_spec import OperationType

T = TypeVar('T', bound='SaveableModel')

class SaveableModel(BaseModel):
    """Базовый класс для моделей с возможностью сохранения/загрузки"""

    reasoning: SkipJsonSchema[str] = Field(
        default="", 
        description="Цепочка рассуждений"
    )
    
    @classmethod
    def load(cls: Type[T], path: str | Path) -> T:
        path = Path(path) if isinstance(path, str) else path
        
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON в файле {path}: {e}")
        
        try:
            return cls.model_validate(data)
        except Exception as e:
            raise ValueError(f"Ошибка валидации данных в файле {path}: {e}")
    
    def save(self, path: str | Path):
        path = Path(path) if isinstance(path, str) else path
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                data = self.model_dump(mode='json')
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"Ошибка сохранения в файл {path}: {e}")

### Формат вопроса в каталоге

class QuestionInfo(BaseModel):
    id: str
    waves: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)

### Retriever

class ScoredQuestion(BaseModel):
    question: str = Field(..., description="Точная формулировка вопроса из базы")
    reason: str = Field(..., description="Почему этот вопрос полезен для ответа на запрос")

class RetrieverOut(SaveableModel):
    results: list[ScoredQuestion] = Field(
        default_factory=list,
        description="Список релевантных вопросов с объяснениями и оценками"
    )

    def clean_list(self):
        return [q.question for q in self.results]
    
    def __str__(self):
        lines = []
        for i, sq in enumerate(self.results, 1):
            lines.append(f"{i}. '{sq.question}'")
            lines.append(f"\tReason: {sq.reason}")
        return "\n".join(lines)

### Planner

class PlanStep(BaseModel):
    id: str = Field(..., description="Уникальный идентификатор шага (s1, s2, ...)")
    operation: OperationType = Field(..., description="Тип операции из OperationType")
    goal: str = Field(default="", description="Человекочитаемая цель шага, кратко")
    inputs: dict[str, str | int | float | list[str]] = Field(
        ...,
        description=(
            "Словарь аргументов для операции. "
            "Ключи ДОЛЖНЫ совпадать с 'Входные параметры' из спецификации (например: 'dataset', 'waves', 'question'). "
            "Значения — это конкретные данные или имена переменных из предыдущих шагов."
        )
    )
    
    outputs: list[str] = Field(..., description="Имена переменных, которые создает этот шаг")
    depends_on: list[str] = Field(default_factory=list, description="ID шагов, от которых зависит этот шаг")

class PlannerOut(SaveableModel):
    # analysis: str = Field(default="", description="Короткий комментарий стратегии")
    steps: list[PlanStep] = Field(default_factory=list)
    export_variables: list[str] = Field(
        default_factory=list,
        description="Список имен переменных (из outputs шагов), которые нужно вернуть пользователю"
    )

    def __str__(self):
        res = []
        # res.append(f"АНАЛИЗ: {self.analysis}")
        for i, s in enumerate(self.steps):
            res.append(f"{i}. [{s.id}] {s.operation}")
            res.append(f"\tGoal: {s.goal}")
            res.append(f"\tInputs: {s.inputs}")
            res.append(f"\tOutputs: {s.outputs}")
            res.append(f"\tDepends on: {s.depends_on}")
        res.append(f"\nEXPORT VARIABLES: {self.export_variables}")
        return "\n".join(res)

# Grounder

class GroundedStep(BaseModel):
    """Шаг с валидированными аргументами, готовый к исполнению"""
    id: str
    goal: str
    op_type: OperationType
    inputs: dict[str, Any]
    outputs: list[str]
    depends_on: list[str]

class GrounderOut(SaveableModel):
    """Результат работы grounder - сохраняемый"""
    steps: list[GroundedStep] = Field(default_factory=list)
    export_variables: list[str] = Field(default_factory=list)

    def __str__(self):
        res = []
        for i, s in enumerate(self.steps):
            res.append(f"{i}. [{s.id}] {s.op_type}")
            res.append(f"\tGoal: {s.goal}")
            res.append(f"\tInputs: {s.inputs}")
            res.append(f"\tOutputs: {s.outputs}")
        res.append(f"\nEXPORT VARIABLES: {self.export_variables}")
        return "\n".join(res)