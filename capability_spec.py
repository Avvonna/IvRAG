import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class OperationType(str, Enum):
    # === DATA LOADING ===
    LOAD_WAVE_DATA = "LOAD_WAVE_DATA"
    
    # === FILTERING ===
    FILTER_BY_QUESTION = "FILTER_BY_QUESTION"
    FILTER_BY_COLUMN = "FILTER_BY_COLUMN"
    
    # === AGGREGATION ===
    COMPUTE_DISTRIBUTION = "COMPUTE_DISTRIBUTION"
    COMPUTE_CROSSTAB = "COMPUTE_CROSSTAB"
    COMPUTE_STATISTICS = "COMPUTE_STATISTICS"
    
    # === TRANSFORMATION ===
    RANK_RESULTS = "RANK_RESULTS"
    COMPARE_GROUPS = "COMPARE_GROUPS"
    
    # === VALIDATION ===
    CHECK_DATA_AVAILABILITY = "CHECK_DATA_AVAILABILITY"
    
    # === OUTPUT ===
    FORMAT_REPORT = "FORMAT_REPORT"


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
        elif "COMPUTE" in name:
            return "Aggregation"
        elif "RANK" in name or "COMPARE" in name:
            return "Transformation"
        elif "CHECK" in name or "VALIDATE" in name:
            return "Validation"
        elif "FORMAT" in name or "REPORT" in name:
            return "Output"
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
                name=OperationType.LOAD_WAVE_DATA,
                description="Загрузить данные конкретной волны опроса",
                inputs={"wave_id": "ID волны (например, '2025-03')"},
                outputs=["dataset"],
                example={"wave_id": "2025-03"}
            ),
            
            # ==================== FILTERING ====================
            OperationSpec(
                name=OperationType.FILTER_BY_QUESTION,
                description="Отфильтровать респондентов по ответам на конкретный вопрос",
                inputs={
                    "dataset": "Исходный датасет",
                    "question_id": "ID вопроса из allowed_questions",
                    "answer_values": "Список допустимых ответов (если пусто - все ответившие)",
                    "logic": "Логика фильтрации: 'include' (есть эти ответы) или 'exclude' (нет этих ответов)"
                },
                outputs=["filtered_dataset"],
                constraints={
                    "question_id": "MUST be from allowed_questions",
                    "logic": "One of: ['include', 'exclude']"
                },
                example={
                    "dataset": "dataset",
                    "question_id": "В каких магазинах Вы делаете покупки? ",
                    "answer_values": ["Пятерочка", "Магнит"],
                    "logic": "include"
                }
            ),
            
            # ==================== AGGREGATION ====================
            OperationSpec(
                name=OperationType.COMPUTE_DISTRIBUTION,
                description="Вычислить распределение ответов на вопрос (частоты и проценты)",
                inputs={
                    "dataset": "Датасет (может быть отфильтрованным)",
                    "question_id": "ID вопроса из allowed_questions",
                    "metric": "Метрика: 'count' (количество), 'percentage' (проценты), 'both'",
                    "sort_by": "Сортировка: 'value' (по алфавиту), 'count' (по частоте), 'none'",
                    "top_n": "Ограничить топ-N результатов (опционально)"
                },
                outputs=["distribution_table"],
                constraints={
                    "question_id": "MUST be from allowed_questions",
                    "metric": "One of: ['count', 'percentage', 'both']"
                },
                example={
                    "dataset": "filtered_dataset",
                    "question_id": "В каких магазинах Вы делаете покупки? ",
                    "metric": "percentage",
                    "sort_by": "count",
                    "top_n": 10
                }
            ),
            
            OperationSpec(
                name=OperationType.COMPUTE_CROSSTAB,
                description="Построить двумерную таблицу (crosstab) по двум вопросам",
                inputs={
                    "dataset": "Датасет",
                    "question_id_rows": "Вопрос для строк",
                    "question_id_cols": "Вопрос для столбцов",
                    "values": "Что показывать: 'count', 'percentage_row', 'percentage_col', 'percentage_total'",
                    "normalize": "Нормализация: 'none', 'rows', 'columns', 'all'"
                },
                outputs=["crosstab_table"],
                constraints={
                    "question_id_rows": "MUST be from allowed_questions",
                    "question_id_cols": "MUST be from allowed_questions"
                },
                example={
                    "dataset": "dataset",
                    "question_id_rows": "В каких магазинах Вы делаете покупки? ",
                    "question_id_cols": "Где Вы покупаете продукты питания? ",
                    "values": "count",
                    "normalize": "rows"
                }
            ),
            
            OperationSpec(
                name=OperationType.COMPUTE_STATISTICS,
                description="Вычислить базовую статистику по группам",
                inputs={
                    "dataset": "Датасет",
                    "group_by_question": "Вопрос для группировки (опционально)",
                    "metrics": "Список метрик: ['total_respondents', 'unique_answers', 'response_rate']"
                },
                outputs=["stats_table"],
                example={
                    "dataset": "dataset",
                    "group_by_question": "В каких магазинах Вы делаете покупки? ",
                    "metrics": ["total_respondents", "unique_answers"]
                }
            ),
            
            # ==================== TRANSFORMATION ====================
            OperationSpec(
                name=OperationType.RANK_RESULTS,
                description="Отсортировать и взять топ-N результатов",
                inputs={
                    "table": "Входная таблица",
                    "sort_column": "Колонка для сортировки",
                    "descending": "True для убывания, False для возрастания",
                    "top_n": "Количество записей (опционально)"
                },
                outputs=["ranked_table"],
                example={
                    "table": "distribution_table",
                    "sort_column": "percentage",
                    "descending": True,
                    "top_n": 10
                }
            ),
            
            OperationSpec(
                name=OperationType.COMPARE_GROUPS,
                description="Сравнить метрики между двумя группами респондентов",
                inputs={
                    "dataset": "Датасет",
                    "question_id": "Вопрос для анализа",
                    "split_by_column": "Колонка для разделения на группы",
                    "group_a_value": "Значение для группы A",
                    "group_b_value": "Значение для группы B",
                    "metric": "'percentage' или 'count'"
                },
                outputs=["comparison_table"],
                example={
                    "dataset": "dataset",
                    "question_id": "В каких магазинах Вы делаете покупки? ",
                    "split_by_column": "wave",
                    "group_a_value": "2025-03",
                    "group_b_value": "2024-12",
                    "metric": "percentage"
                }
            ),
            
            # ==================== VALIDATION ====================
            OperationSpec(
                name=OperationType.CHECK_DATA_AVAILABILITY,
                description="Проверить доступность необходимых данных (вопросов, колонок)",
                inputs={
                    "required_questions": "Список ID вопросов, которые нужны",
                    "required_columns": "Список колонок, которые нужны"
                },
                outputs=["validation_report"],
                example={
                    "required_questions": ["В каких магазинах Вы делаете покупки? "],
                    "required_columns": ["city", "wave"]
                }
            ),
            
            # ==================== OUTPUT ====================
            OperationSpec(
                name=OperationType.FORMAT_REPORT,
                description="Отформатировать финальную таблицу для отчёта",
                inputs={
                    "table": "Таблица для форматирования",
                    "format": "'markdown', 'csv', или 'json'",
                    "round_decimals": "Количество знаков после запятой (опционально)",
                    "add_summary": "Добавить текстовое резюме (True/False)"
                },
                outputs=["formatted_report"],
                example={
                    "table": "ranked_table",
                    "format": "markdown",
                    "round_decimals": 2,
                    "add_summary": True
                }
            ),
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