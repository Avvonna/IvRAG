import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class OperationType(str, Enum):
    LOAD_DATA = "LOAD_DATA"
    FILTER = "FILTER"
    PIVOT = "PIVOT"


@dataclass
class OperationSpec:
    name: OperationType
    description: str
    inputs: dict[str, str]  # {param_name: description}
    outputs: list[str]
    constraints: dict[str, str] = field(default_factory=dict)
    example: dict = field(default_factory=dict)
    category: str = field(default_factory=str)


@dataclass
class CapabilitySpec:
    operations: list[OperationSpec] = field(default_factory=list)
    limits: dict[str, int] = field(default_factory=lambda: {
        "max_steps": 10,
        "max_operations_per_type": 3
    })

    def __post_init__(self):
        """Автоматически определяем категории операций"""
        if not self.operations:
            self.operations = self._create_default_operations()
        
        # Автозаполнение категорий из enum
        for op in self.operations:
            if not op.category:
                op.category = self._infer_category(op.name)

    def _infer_category(self, op_type: OperationType) -> str:
        name = op_type.value
        if "LOAD" in name:
            return "Data Loading"
        elif "FILTER" in name:
            return "Filtering"
        elif "PIVOT" in name:
            return "Aggregation"
        return "Other"

    def list_operations(self, category: str | None = None) -> list[OperationSpec]:
        if category is None:
            return self.operations
        return [op for op in self.operations if op.category == category]

    def get_categories(self) -> list[str]:
        return sorted(set(op.category for op in self.operations))

    @staticmethod
    def _create_default_operations() -> list[OperationSpec]:
        return [
            # ==================== DATA LOADING ====================
            OperationSpec(
                name=OperationType.LOAD_DATA,
                description="Загрузить данные результатов опросов",
                inputs={"waves": "Список волн, которые следует взять для анализа"},
                outputs=["dataset"],
                example={"waves": ["2025-01"]}
            ),
            
            # ==================== FILTERING ====================
            OperationSpec(
                name=OperationType.FILTER,
                description="Отфильтровать респондентов по ответам на конкретный вопрос",
                inputs={
                    "dataset": "Исходный датасет",
                    "question": "Вопрос из allowed_questions",
                    "answer_values": "Список допустимых ответов (если пусто - все ответившие)",
                    "logic": "Логика фильтрации: 'include' (есть эти ответы) или 'exclude' (нет этих ответов)"
                },
                outputs=["filtered_dataset"],
                constraints={
                    "question": "MUST be from allowed_questions",
                    "logic": "One of: ['include', 'exclude']"
                },
                example={
                    "dataset": "dataset",
                    "question": "[Тег] Вопрос?",
                    "answer_values": ["Ответ 1", "Ответ 2"],
                    "logic": "include"
                }
            ),
            
            # ==================== AGGREGATION ====================
            OperationSpec(
                name=OperationType.PIVOT,
                description="Вычислить распределение ответов на вопрос",
                inputs={
                    "dataset": "Датасет (может быть отфильтрованным)",
                    "question": "Вопрос из allowed_questions"
                },
                outputs=["pivot"],
                constraints={
                    "question": "MUST be from allowed_questions",
                },
                example={
                    "dataset": "filtered_dataset",
                    "question": "В каких магазинах Вы делаете покупки?"
                }
            )
        ]

    def to_prompt_context(
        self,
        format: Literal["detailed", "compact", "json"] = "detailed",
        include_examples: bool = True,
        include_constraints: bool = True
    ) -> str:
        if format == "json":
            return self._to_json()
        elif format == "compact":
            return self._to_compact()
        else:  # detailed
            return self._to_detailed(include_examples, include_constraints)

    def _to_detailed(self, include_examples: bool, include_constraints: bool) -> str:
        lines = ["# ДОСТУПНЫЕ ОПЕРАЦИИ\n"]
        
        for category in self.get_categories():
            lines.append(f"## {category}\n")
            
            for op in self.list_operations(category):
                lines.append(f"### {op.name.value}")
                lines.append(f"**Описание:** {op.description}\n")
                
                # Входы
                lines.append("**Входные параметры:**")
                for param, desc in op.inputs.items():
                    lines.append(f"  - `{param}`: {desc}")
                lines.append("")
                
                # Выходы
                lines.append(f"**Выходы:** {', '.join(f'`{o}`' for o in op.outputs)}\n")
                
                # Ограничения
                if include_constraints and op.constraints:
                    lines.append("**Ограничения:**")
                    for key, value in op.constraints.items():
                        lines.append(f"  - {key}: {value}")
                    lines.append("")
                
                # Пример
                if include_examples and op.example:
                    example_json = json.dumps(op.example, ensure_ascii=False, indent=2)
                    lines.append("**Пример использования:**")
                    lines.append("```json")
                    lines.append(example_json)
                    lines.append("```\n")
                
                lines.append("---\n")
        
        lines.append("## Ограничения системы\n")
        for limit_name, limit_value in self.limits.items():
            lines.append(f"- **{limit_name}**: {limit_value}")
        
        return "\n".join(lines)

    def _to_compact(self) -> str:
        lines = []
        for op in self.operations:
            inputs_str = ", ".join(op.inputs.keys())
            outputs_str = ", ".join(op.outputs)
            lines.append(f"- **{op.name.value}**({inputs_str}) → [{outputs_str}]")
            lines.append(f"  {op.description}")
        return "\n".join(lines)

    def _to_json(self) -> str:
        data = {
            "operations": [
                {
                    "name": op.name.value,
                    "description": op.description,
                    "inputs": op.inputs,
                    "outputs": op.outputs,
                    "constraints": op.constraints,
                    "example": op.example,
                    "category": op.category
                }
                for op in self.operations
            ],
            "limits": self.limits
        }
        return json.dumps(data, ensure_ascii=False, indent=2)