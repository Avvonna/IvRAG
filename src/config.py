from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Type, TypedDict

import pandas as pd
from openai import OpenAI
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel

from .capability_spec import CapabilitySpec
from .catalog import QuestionCatalog
from .schemas import RetrieverOut
from .utils import get_unique_questions_info

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_PROVIDER_SORT = None
DEFAULT_REASONING = None
DEFAULT_BASE_DELAY = 1.0
DEFAULT_N_RETRIES = 3
DEFAULT_MAX_TOKENS = None


class AgentParams(TypedDict, total=False):
    """Type definition for agent configuration parameters."""
    model: str
    temperature: float
    response_model: Type[BaseModel]
    max_tokens: int
    provider_sort: Literal["latency", "price", "throughput"]
    reasoning_effort: ReasoningEffort
    base_delay: float
    retries: int

@dataclass
class BaseAgentConfig:
    """Базовая конфигурация для всех агентов"""
    model: str = field(default=DEFAULT_MODEL)
    temperature: float = field(default=DEFAULT_TEMPERATURE)
    response_model: Type[BaseModel] | None = field(default=None)
    max_tokens: int | None = field(default=DEFAULT_MAX_TOKENS)
    provider_sort: Literal["latency", "throughput", "price"] | None = field(default=DEFAULT_PROVIDER_SORT)
    reasoning_effort: ReasoningEffort = field(default=DEFAULT_REASONING)
    base_delay: float = field(default=DEFAULT_BASE_DELAY)
    retries: int = field(default=DEFAULT_N_RETRIES)


@dataclass
class RetrieverConfig(BaseAgentConfig):
    """Конфигурация для Retriever"""
    n_questions_splits: int = field(default=2)
    response_model: Type[BaseModel] | None = field(default=RetrieverOut)
    pass

@dataclass
class PlannerConfig(BaseAgentConfig):
    """Конфигурация для Planner"""
    capability_spec: CapabilitySpec = field(default_factory=CapabilitySpec)


@dataclass
class PipelineConfig:
    """Конфигурация Pipeline"""
    client: OpenAI

    # Конфиги агентов
    retriever_config: RetrieverConfig
    planner_config: PlannerConfig

    # Входные данные
    source_df: pd.DataFrame = field(repr=False)
    # По умолчанию берутся вопросы только за последнюю волну
    question_waves: list[str] | None = field(default_factory=lambda: ["2025-03"])

    # Служебные поля
    catalog: QuestionCatalog = field(init=False)
    all_QS_info_dict: dict = field(init=False, repr=False)
    relevant_questions: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        # Валидация и фильтрация DataFrame
        if self.question_waves:
            available_waves = set(self.source_df["wave"].unique())
            requested_waves = set(self.question_waves)
            
            diff_waves = requested_waves - available_waves
            if diff_waves:
                raise KeyError(f"В БД отсутствуют запрашиваемые волны: {sorted(diff_waves)}")
        
        # Создаем каталог
        self.catalog = QuestionCatalog.from_df(
            get_unique_questions_info(self.source_df)
        )
        
        # Заполняем вспомогательные поля
        self.all_QS_info_dict = self.catalog.as_value_catalog()
    
    def relevant_as_value_catalog(self):
        if self.relevant_questions:
            return self.catalog\
                .filter(self.relevant_questions)\
                .as_value_catalog()
        raise ValueError("No relevant questions specified")

    def update_context(self, source: RetrieverOut | list[str]):
        """Обновляет контекст релевантными вопросами из RetrieverOut или списка"""
        if isinstance(source, list):
            self.relevant_questions = source
        elif hasattr(source, "clean_list"):
            self.relevant_questions = source.clean_list()
        else:
            raise TypeError("Source must be RetrieverOut or list[str]")

    @classmethod
    def setup(
        cls,
        df: pd.DataFrame,
        question_waves: list[str] | None,
        client: OpenAI,
        retriever_params: AgentParams = {},
        planner_params: AgentParams = {},
    ) -> PipelineConfig:

        rc = RetrieverConfig(**retriever_params)
        pp = PlannerConfig(**planner_params)

        return cls(
            client=client,
            retriever_config=rc,
            planner_config=pp,
            source_df=df,
            question_waves=question_waves
        )
