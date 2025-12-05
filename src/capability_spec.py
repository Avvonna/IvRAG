import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class OperationType(str, Enum):
    LOAD_DATA = "LOAD_DATA"
    FILTER = "FILTER"
    PIVOT = "PIVOT"
    INTERSECT = "INTERSECT"
    UNION = "UNION"
    CALCULATE_AVERAGE = "CALCULATE_AVERAGE"


@dataclass
class OperationSpec:
    name: OperationType
    description: str
    inputs: dict[str, str]  # {param_name: description}
    outputs: list[str]
    example: dict = field(default_factory=dict)
    category: str = field(default_factory=str)


@dataclass
class CapabilitySpec:
    operations: list[OperationSpec] = field(default_factory=list)
    limits: dict[str, int] = field(default_factory=lambda: {})

    def __post_init__(self):
        """Автоматически определяем категории операций"""
        if not self.operations:
            self.operations = self._create_default_operations()
        
        # Автозаполнение категорий из enum
        for op in self.operations:
            if not op.category:
                op.category = self._infer_category(op.name)

    def _infer_category(self, op_type: OperationType) -> str:
        match op_type:
            case OperationType.LOAD_DATA:
                return "Data Loading"
            case OperationType.FILTER | OperationType.INTERSECT | OperationType.UNION:
                return "Filtering"
            case OperationType.PIVOT | OperationType.CALCULATE_AVERAGE:
                return "Aggregation"
            case _:
                return "Other"

    def list_operations(self, category: str | None = None) -> list[OperationSpec]:
        if category is None:
            return self.operations
        return [op for op in self.operations if op.category == category]

    def get_categories(self) -> list[str]:
        return sorted(list(set(op.category for op in self.operations)))

    @staticmethod
    def _create_default_operations() -> list[OperationSpec]:
        return [
            # ==================== DATA LOADING ====================

            OperationSpec(
                name=OperationType.LOAD_DATA,
                description="Загрузить данные результатов опросов.",
                inputs={
                    "waves": "List[str] — Список кодов волн (например, ['2025-01']). Если пустой список — загружаются все."
                },
                outputs=["dataset"],
                example={"waves": ["2025-01"]}
            ),

            # ==================== FILTERING ====================

            OperationSpec(
                name=OperationType.FILTER,
                description="Оставить в датасете только тех респондентов, которые дали определенные ответы на указанный вопрос.",
                inputs={
                    "dataset": "DataFrame — Исходный датасет",
                    "question": "str — Точная формулировка вопроса для фильтрации",
                    "answer_values": "List[str] — Список ответов. Если None/пустой — берутся все, кто ответил на вопрос.",
                    "logic": "str — 'include' (оставить этих) или 'exclude' (убрать этих). Default: 'include'."
                },
                outputs=["filtered_dataset"],
                example={
                    "dataset": "dataset",
                    "question": "Ваш возраст?",
                    "answer_values": ["18-24", "25-34"],
                    "logic": "include"
                }
            ),

            OperationSpec(
                name=OperationType.INTERSECT,
                description="Найти пересечение аудиторий (Логическое И). Возвращает датасет с респондентами, которые присутствуют ВО ВСЕХ переданных датасетах.",
                inputs={
                    "datasets": "List[DataFrame] — Список переменных с датасетами (минимум 2)",
                },
                outputs=["intersected_dataset"],
                example={
                    "datasets": ["df_men", "df_young"]
                }
            ),

            OperationSpec(
                name=OperationType.UNION,
                description="Найти объединение аудиторий (Логическое ИЛИ). Возвращает датасет с респондентами, которые есть ХОТЯ БЫ В ОДНОМ из датасетов.",
                inputs={
                    "datasets": "List[DataFrame] — Список переменных с датасетами (минимум 2)"
                },
                outputs=["union_dataset"],
                example={
                    "datasets": ["df_customers_a", "df_customers_b"]
                }
            ),

            # ==================== AGGREGATION ====================

            OperationSpec(
                name=OperationType.PIVOT,
                description=(
                    "Построить кросс-таблицу. "
                    "Rows (Индекс): Ответы на указанные вопросы. "
                    "Columns: Волны опросов. "
                    "Values: Количество уникальных респондентов. "
                    "Если указано несколько вопросов, создается MultiIndex."
                ),
                inputs={
                    "dataset": "DataFrame — Датасет",
                    "questions": "List[str] — Список вопросов для группировки (строки таблицы)."
                },
                outputs=["pivot"],
                example={
                    "dataset": "filtered_dataset",
                    "questions": ["Ваш пол", "Ваш возраст"]
                }
            ),

            OperationSpec(
                name=OperationType.CALCULATE_AVERAGE,
                description=(
                    "Рассчитать взвешенное среднее (Weighted Mean) по шкале. "
                    "ВАЖНО: Работает корректно только если PIVOT был построен по ОДНОМУ вопросу. "
                    "Ключи в `scale` должны точно совпадать с вариантами ответов в строках таблицы."
                ),
                inputs={
                    "pivot_table": "DataFrame — Результат операции PIVOT",
                    "scale": "Dict[str, float] — Словарь весов {'Ответ': Вес}."
                },
                outputs=["average_table"],
                example={
                    "pivot_table": "pivot_nps",
                    "scale": {'Полностью согласен': 5, 'Скорее согласен': 4, 'Нет': 1}
                }
            )
        ]

    def to_prompt_context(
        self,
        format: Literal["detailed", "compact", "json"] = "detailed",
        include_examples: bool = True
    ) -> str:
        if format == "json":
            return self._to_json()
        elif format == "compact":
            return self._to_compact()
        else:  # detailed
            return self._to_detailed(include_examples)

    def _to_detailed(self, include_examples: bool) -> str:
        lines = ["# ДОСТУПНЫЕ ОПЕРАЦИИ\n"]
        
        for category in self.get_categories():
            lines.append(f"## {category}\n")
            
            for op in self.list_operations(category):
                lines.append(f"### {op.name.value}")
                lines.append(f"{op.description}\n")
                
                # Входы
                lines.append("**Входные параметры:**")
                for param, desc in op.inputs.items():
                    lines.append(f"  - `{param}`: {desc}")
                lines.append("")
                
                # Выходы
                lines.append(f"**Создает переменные:** {', '.join(f'`{o}`' for o in op.outputs)}\n")
                
                # Пример
                if include_examples and op.example:
                    example_json = json.dumps(op.example, ensure_ascii=False, indent=2)
                    lines.append("**Пример использования:**")
                    lines.append("```json")
                    lines.append(example_json)
                    lines.append("```\n")
                
                lines.append("---\n")
        
        if self.limits:
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
                    "example": op.example,
                    "category": op.category
                }
                for op in self.operations
            ],
            "limits": self.limits
        }
        return json.dumps(data, ensure_ascii=False, indent=2)