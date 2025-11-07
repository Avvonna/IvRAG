from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import dotenv
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from utils import (
    find_top_match,
    get_unique_questions_info,
    retry_call,
)

dotenv.load_dotenv()


DEFAULT_MODEL       = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_BASE_DELAY  = 1.0
DEFAULT_N_RETRIES   = 3


#################
# КОНФИГИ
#################

### Retriever ###

@dataclass
class RetrieverConfig():
    model: str          = field(default=DEFAULT_MODEL)
    temperature: float  = field(default=DEFAULT_TEMPERATURE)
    base_delay: float   = field(default=DEFAULT_BASE_DELAY)
    retries: int        = field(default=DEFAULT_N_RETRIES)

### Planner ###

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
                inputs={
                    "wave_id": "ID волны (например, '2025-03')"
                },
                outputs=["dataset"],
                example={
                    "wave_id": "2025-03"
                }
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

@dataclass
class PlannerConfig:
    model: str          = field(default=DEFAULT_MODEL)
    temperature: float  = field(default=DEFAULT_TEMPERATURE)
    base_delay: float   = field(default=DEFAULT_BASE_DELAY)
    retries: int        = field(default=DEFAULT_N_RETRIES)

    capability_spec: CapabilitySpec = field(default_factory=CapabilitySpec)

### Grounder ###

@dataclass
class GroundedStep:
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
    analysis: str
    steps: list[GroundedStep]

@dataclass
class GrounderConfig:
    model: str          = field(default=DEFAULT_MODEL)
    temperature: float  = field(default=DEFAULT_TEMPERATURE)
    base_delay: float   = field(default=DEFAULT_BASE_DELAY)
    retries: int        = field(default=DEFAULT_N_RETRIES)

### Executor ###

@dataclass
class ExecutorConfig:
    model: str          = field(default=DEFAULT_MODEL)
    temperature: float  = field(default=DEFAULT_TEMPERATURE)
    base_delay: float   = field(default=DEFAULT_BASE_DELAY)
    retries: int        = field(default=DEFAULT_N_RETRIES)

### Pipeline ###

@dataclass
class PipelineConfig:
    """ Конфигурация Pipeline """
    client: OpenAI

    retriever_config: RetrieverConfig
    planner_config: PlannerConfig
    grounder_config: GrounderConfig
    executor_config: ExecutorConfig

    df_schema: list
    catalog: QuestionCatalog

    # Необходимо для контекста LLM
    all_QS_clean_list: list = field(init=False)     # Список вопросов (q_clean)
    all_QS_info_dict:  dict = field(init=False)     # Словарь вопросов со всеми опциями, ответами и пр.

    def __post_init__(self):
        self.all_QS_clean_list = self.catalog.allowed_question_ids()
        self.all_QS_info_dict  = self.catalog.as_value_catalog()


#################
# PYDANTIC СХЕМЫ
#################

### Questions ###

class QuestionInfo(BaseModel):
    id: str
    waves: list[str]   = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)

    def stringify(
        self,
        include: list[Literal["q_clean", "answers", "details", "options", "waves"]] = ["q_clean","answers","details","options"]
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

class QuestionCatalog(BaseModel):
    questions: list[QuestionInfo] = Field(default_factory=list)

    def allowed_question_ids(self) -> list[str]:
        return [q.id for q in self.questions]

    def as_value_catalog(self, limit=30) -> dict[str, dict]:
        def clip(xs: list[str]) -> list[str]:
            return xs[:limit] if limit and len(xs) > limit else xs
        return {
            q.id: {
                "answers": clip(q.answers),
                "details": clip(q.details),
                "options": clip(q.options),
                "waves":   clip(q.waves),
            }
            for q in self.questions
        }
    
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame
    ) -> QuestionCatalog:
        questions = [
            QuestionInfo(
                id=getattr(row, "q_clean"),
                waves=getattr(row, "waves"),
                options=getattr(row, "options"),
                details=getattr(row, "details"),
                answers=getattr(row, "answers"),
            )
            for row in df.itertuples(index=False)
        ]
        return cls(questions=sorted(questions, key=lambda q: q.id))

### Retriever ###

class ScoredQuestion(BaseModel):
    question: str       = Field(..., description="Точная формулировка вопроса из базы")
    reason: str         = Field(..., description="Почему этот вопрос полезен для ответа на запрос")
    relevance: float    = Field(..., description="Оценка релевантности 0–100")

class RetrieverOut(BaseModel):
    results: list[ScoredQuestion] = Field(
        default_factory=list,
        description="Список релевантных вопросов с объяснениями и оценками"
    )

    def clean_list(self):
        return [q.question for q in self.results]

### Planner ###

class PlanStep(BaseModel):
    id: str = Field(..., description="Уникальный идентификатор шага (s1, s2, ...)")
    goal: str = Field("", description="Человекочитаемая цель шага")
    operation: OperationType = Field(..., description="Тип операции из OperationType")
    inputs: dict[str, Any] | list | None = Field(default_factory=dict, description="Именованные входы ИЛИ []")
    outputs: list[str] | None = Field(default_factory=list, description="Имена выходов, которые появятся в контексте")
    constraints: dict[str, Any] | None = Field(default_factory=dict, description="Параметры операции")
    depends_on: list[str] | None = Field(default_factory=list, description="ID шагов, от которых зависит текущий")

class PlannerOut(BaseModel):
    analysis: str = Field("", description="Короткий комментарий стратегии")
    steps: list[PlanStep]

#################
# ФУНКЦИИ
#################

### Retriever ###

def retriever(
    user_query: str,
    config: PipelineConfig
) -> RetrieverOut:
    if not config.all_QS_clean_list:
        return RetrieverOut(results=[])

    questions_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(config.all_QS_info_dict))

    rc = config.retriever_config

    prompt = f"""
Запрос пользователя: {user_query}

Список доступных вопросов:
{questions_block}

---
Инструкция:

1. Проанализируй запрос
2. Подбери релевантные вопросы из списка с оценкой релевантности (0-100)

Формат вывода:

**Рассуждение:**
[Краткий анализ запроса: 2-5 предложений о том, что нужно пользователю]

**Рекомендованные вопросы:**

1. **[90/100]** "[Исходная формулировка вопроса]"
   • **Обоснование:** [Почему вопрос напрямую отвечает на запрос]

2. **[80/100]** "[Исходная формулировка вопроса]"
   • **Обоснование:** [Как вопрос связан с ключевой темой]

3. **[60/100]** "[Исходная формулировка вопроса]"
   • **Обоснование:** [Какой контекст или смежную тему раскрывает]

4. **[50/100]** "[Исходная формулировка вопроса]"
   • **Обоснование:** [Чем может быть полезен для понимания]
---
"""

    def _call():
        resp = config.client.chat.completions.create(
            model=rc.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=rc.temperature,
        )
        assert resp.choices[0].message.content
        
        return resp.choices[0].message.content.strip()

    response_text = retry_call(_call, retries=rc.retries, base_delay=rc.base_delay)
    retriever_struct_out = get_retriever_struct_out(response_text)

    for qs in retriever_struct_out.results:
        qs.question = find_top_match(qs.question, config.all_QS_clean_list)

    return retriever_struct_out

def get_retriever_struct_out(text):
    pattern = r'''
        \*{0,2}\[(?P<score>\d+)/100\]\*{0,2}\s*
        ["«]?(?P<question>[^"\n«»]+?)["»]?\s*
        [•\-*]?\s*\*{0,2}Обоснование\*{0,2}:?\s*
        (?P<reason>[^\n]+)
    '''

    matches = re.finditer(pattern, text, re.VERBOSE | re.IGNORECASE)

    scored_questions = []
    for match in matches:
        scored_questions.append(
            ScoredQuestion(
                question = match.group("question"),
                reason = match.group("reason").strip(),
                relevance = float(match.group("score"))
            )
        )

    return RetrieverOut(results=scored_questions)

### Planner ###

def make_planner_prompt(user_query: str, context: dict, capability_spec: CapabilitySpec) -> str:
    context_json = json.dumps(context or {}, ensure_ascii=False)

    prompt = f"""
Роль: Ты — ПЛАНИРОВЩИК (tool-agnostic). Составь абстрактный план как DAG шагов для решения задачи.

Цель:
- user_query: {user_query}
- context: {context_json}

{capability_spec.to_prompt_context("detailed")}

ВАЖНО:
- Используй ТОЛЬКО операции из CapabilitySpec.
- Используй ТОЛЬКО вопросы из allowed_questions. НЕ ПРИДУМЫВАЙ новые.
- Оперируй ТОЛЬКО колонками из dataset_schema. Если операция требует поля, которого нет, не используй её.
- Если для решения требуется вопрос, которого нет в allowed_questions, верни план из одного шага CHECK_DATA_AVAILABILITY
  с понятным сообщением об отсутствии вопроса и не пытайся подменять его.

Правила построения плана:
1) До 10 шагов (если не указано иное в лимитах). id шагов: s1, s2, …
2) У каждого шага: {{ "id", "goal", "operation", "inputs", "outputs", "constraints", "depends_on"(опц.) }}.
3) Все inputs должны быть доступны из user_query/контекста или outputs предыдущих шагов.

Ответ ТОЛЬКО ЧИСТЫМ JSON (без Markdown, без комментариев).

Формат ответа (СТРОГО):
{{
  "analysis": "1–2 предложения о стратегии",
  "steps": [
    {{"id":"s1","goal":"...","operation":"LOAD_WAVE_DATA","inputs":[],"outputs":["dataset"],"constraints":{{"wave_id":"<wave_id>"}}}}
  ]
}}
""".strip()
    return prompt

def planner(
    user_query: str,
    retriever_out: RetrieverOut,
    config: PipelineConfig
) -> PlannerOut:
    plc = config.planner_config

    # Фильтруем вопросы
    chosen_clean_questions_list = retriever_out.clean_list()
    chosen_questions_catalog = QuestionCatalog(
        questions=[q for q in config.catalog.questions if q.id in chosen_clean_questions_list]
    )
    
    # Подмешиваем их в контекст
    context = {
        "allowed_questions": chosen_questions_catalog.as_value_catalog(),
        "dataset_schema": config.df_schema,
    }

    prompt = make_planner_prompt(user_query, context, plc.capability_spec)
    print(prompt)

    # def _call():
    #     resp = config.client.chat.completions.create(
    #         model=plc.model,
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=plc.temperature,
    #         response_format={"type": "json_object"}
    #     )
    #     content = resp.choices[0].message.content
    #     assert content, "Empty planner response"
    #     return content.strip()

    def _call():
        resp = config.client.responses.parse(
            model=plc.model,
            input=[{"role": "user", "content": prompt}],
            temperature=plc.temperature,
            text_format=PlannerOut
        )
        return resp.output_parsed
    
    return retry_call(_call, retries=plc.retries, base_delay=plc.base_delay)

### Grounder ###

class GroundingError(Exception):
    pass

def op_LOAD_WAVE_DATA(*, wave_id: str, df_full: pd.DataFrame, **_) -> dict[str, Any]:
    if "wave" not in df_full.columns:
        raise GroundingError("В данных нет колонки 'wave'")
    ds = df_full[df_full["wave"] == wave_id].copy()
    return {"dataset": ds}

def op_FILTER_BY_QUESTION(
    *, dataset: pd.DataFrame, question_id: str,
    answer_values: Optional[list[str]] = None,
    logic: str = "include", **_
) -> dict[str, Any]:
    if question_id not in dataset.columns:
        raise GroundingError(f"Нет колонки-вопроса: {question_id}")
    ser = dataset[question_id]
    if answer_values and len(answer_values) > 0:
        mask = ser.isin(answer_values)
    else:
        mask = ser.notna()  # «все ответившие»
    if logic == "exclude":
        mask = ~mask
    out = dataset[mask].copy()
    return {"filtered_dataset": out}

def op_FILTER_BY_COLUMN(
    *, dataset: pd.DataFrame, column: str,
    values: Optional[list[Any]] = None,
    op: str = "in", **_
) -> dict[str, Any]:
    if column not in dataset.columns:
        raise GroundingError(f"Нет колонки: {column}")
    if op == "in":
        mask = dataset[column].isin(values or [])
    elif op == "eq":
        if not values:
            raise GroundingError("Для eq нужен один values[0]")
        mask = dataset[column] == values[0]
    else:
        raise GroundingError(f"Неизвестный op: {op}")
    return {"filtered_dataset": dataset[mask].copy()}

def op_COMPUTE_CROSSTAB(
    *, dataset: pd.DataFrame,
    question_id_rows: str, question_id_cols: str,
    values: str = "count", normalize: str = "none", **_
) -> dict[str, Any]:
    r, c = question_id_rows, question_id_cols
    if r not in dataset.columns or c not in dataset.columns:
        missing = [x for x in (r, c) if x not in dataset.columns]
        raise GroundingError(f"Нет колонок: {missing}")
    norm_map = {"none": False, "rows": "index", "columns": "columns", "all": "all"}
    norm = norm_map.get(normalize, False)

    if values == "count":
        ct = pd.crosstab(dataset[r], dataset[c], dropna=False)
    elif values == "percentage_row":
        ct = pd.crosstab(dataset[r], dataset[c], normalize="index", dropna=False) * 100.0
    elif values == "percentage_col":
        ct = pd.crosstab(dataset[r], dataset[c], normalize="columns", dropna=False) * 100.0
    elif values == "percentage_total":
        ct = pd.crosstab(dataset[r], dataset[c], normalize="all", dropna=False) * 100.0
    else:
        ct = pd.crosstab(dataset[r], dataset[c], normalize=norm, dropna=False)
        if norm is not None:
            ct = ct * 100.0

    return {"crosstab_table": ct.reset_index()}

# Реестр
OP_REGISTRY: dict[OperationType, Callable[..., dict[str, Any]]] = {
    OperationType.LOAD_WAVE_DATA: op_LOAD_WAVE_DATA,
    OperationType.FILTER_BY_QUESTION: op_FILTER_BY_QUESTION,
    OperationType.FILTER_BY_COLUMN: op_FILTER_BY_COLUMN,
    OperationType.COMPUTE_CROSSTAB: op_COMPUTE_CROSSTAB,
}

def grounder(plan: PlannerOut, config: PipelineConfig) -> GrounderOut:
    analysis = plan.analysis
    steps_in = plan.steps
    if not isinstance(steps_in, list) or not steps_in:
        raise GroundingError("Пустой список шагов")

    grounded: list[GroundedStep] = []
    seen: set[str] = set()

    for s in steps_in:
        sid = s.id
        if not sid or sid in seen:
            raise GroundingError(f"Проблема с id шага: {sid}")
        seen.add(sid)

        op_type: OperationType = s.operation
        impl = OP_REGISTRY.get(op_type)
        if not impl:
            raise GroundingError(f"Нет реализации для операции: {op_type.value}")

        inputs = s.inputs or {}
        if isinstance(inputs, list):
            inputs = {}

        grounded.append(GroundedStep(
            id=sid,
            goal=s.goal or "",
            op_type=op_type,                 # ← enum
            impl=impl,
            inputs=inputs,                   # ← dict[str, Any]
            outputs=list(s.outputs or []),
            constraints=dict(s.constraints or {}),
            depends_on=list(s.depends_on or []),
        ))

    return GrounderOut(analysis=analysis, steps=grounded)

def _materialize_inputs(step: GroundedStep, ctx: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    for param_name, ref in (step.inputs or {}).items():
        kwargs[param_name] = ctx.get(ref, ref)

    kwargs.update(step.constraints or {})

    if "df_full" in ctx and "df_full" not in kwargs:
        kwargs["df_full"] = ctx["df_full"]
    
    return kwargs

def executor(grounded_plan: GrounderOut, runtime_ctx: dict[str, Any]) -> dict[str, Any]:
    for st in grounded_plan.steps:
        for dep in st.depends_on or []:
            if dep not in [x.id for x in grounded_plan.steps]:
                raise GroundingError(f"Неизвестная зависимость: {dep}")
        kwargs = _materialize_inputs(st, runtime_ctx)
        result = st.impl(**kwargs)
        if not isinstance(result, dict):
            raise GroundingError(f"Шаг {st.id} вернул не словарь")
        runtime_ctx.update(result)
        if st.outputs and len(result) == 1:
            only_val = list(result.values())[0]
            for out_name in st.outputs:
                runtime_ctx.setdefault(out_name, only_val)
    return runtime_ctx

def run_pipeline(
    user_query: str,
    config: PipelineConfig
):
    """ Мастер-функция """
    # 1. Retriever
    retriever_out = retriever(user_query, config)
    print("Выбранные вопросы:")
    for i, q in enumerate(retriever_out.results):
        print(f"{i}. '{q.question}'")

    # 2. Planner
    print("\n\nPLANNER:\n\n")
    planner_out = planner(user_query, retriever_out, config)
    print(planner_out)

    # 3. Grounder
    gplan = grounder(planner_out, config)

    # 4. Executor
    ctx = {"df_full": df}  # сюда можно положить ещё что-то, если нужно
    final_ctx = executor(gplan, ctx)

    # Для отладки выведем возможные артефакты
    for key in ["dataset", "filtered_dataset", "crosstab_table"]:
        if key in final_ctx:
            print(f"\n[{key}]\n", final_ctx[key].head() if isinstance(final_ctx[key], pd.DataFrame) else final_ctx[key])



if __name__ == "__main__":
    API_KEY = os.getenv("OR_API_KEY")
    DB_PATH = os.getenv("DB_PATH")

    assert API_KEY
    assert DB_PATH

    ### Client ###

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )

    ### DB ###
    
    df = pd.read_parquet(Path(DB_PATH), engine="fastparquet")
    df = df[df["wave"] == "2025-03"]
    print("Размерность данных", df.shape)

    ### Questions ###

    df_questions_info = get_unique_questions_info(df)
    print("Информация о вопросах:", df_questions_info.shape)

    catalog=QuestionCatalog.from_df(df_questions_info)

    ### Pipeline config ###

    ReC = RetrieverConfig()

    # Урезанный набор опций
    all_ops = CapabilitySpec._create_default_operations()
    allowed = {OperationType.LOAD_WAVE_DATA, OperationType.FILTER_BY_QUESTION, OperationType.COMPUTE_CROSSTAB}
    mini_ops = [op for op in all_ops if op.name in allowed]
    PlC = PlannerConfig(capability_spec=CapabilitySpec(operations=mini_ops))

    GrC = GrounderConfig()
    ExC = ExecutorConfig()

    pc = PipelineConfig(
        client=client,
        retriever_config=ReC,
        planner_config=PlC,
        grounder_config=GrC,
        executor_config=ExC,
        df_schema=df.columns.to_list(),
        catalog=catalog
    )

    ### Run pipeline ###

    uq = "Какое распределение посетителей торговых сетей? Мне нужна только выборка по жителям Москвы и Московской области"
    run_pipeline(uq, pc)