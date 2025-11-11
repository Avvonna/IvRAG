from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from openai import OpenAI

from catalog import QuestionCatalog
from utils import get_unique_questions_info

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_BASE_DELAY = 1.0
DEFAULT_N_RETRIES = 3


@dataclass
class BaseAgentConfig:
    """Базовая конфигурация для всех агентов"""
    model: str = field(default=DEFAULT_MODEL)
    temperature: float = field(default=DEFAULT_TEMPERATURE)
    base_delay: float = field(default=DEFAULT_BASE_DELAY)
    retries: int = field(default=DEFAULT_N_RETRIES)


@dataclass
class RetrieverConfig(BaseAgentConfig):
    """Конфигурация для Retriever"""
    pass


@dataclass
class PlannerConfig(BaseAgentConfig):
    """Конфигурация для Planner"""
    from capability_spec import CapabilitySpec
    capability_spec: CapabilitySpec = field(default_factory=CapabilitySpec)


@dataclass
class PipelineConfig:
    """Конфигурация Pipeline"""
    client: OpenAI

    retriever_config: RetrieverConfig
    planner_config: PlannerConfig

    df_schema: list
    catalog: QuestionCatalog

    # Необходимо для контекста LLM
    all_QS_clean_list: list = field(init=False)
    all_QS_info_dict: dict = field(init=False)

    def __post_init__(self):
        self.all_QS_clean_list = self.catalog.allowed_question_ids()
        self.all_QS_info_dict = self.catalog.as_value_catalog()
    
    @classmethod
    def setup(
        cls,
        df: pd.DataFrame,
        client: OpenAI,
        retriever_params: dict[Literal["model", "temperature"], str|float] = {},
        planner_params: dict[Literal["model", "temperature"], str|float] = {},
    ):
        rc = RetrieverConfig(**retriever_params)
        pp = PlannerConfig(**planner_params)

        return cls(
            client=client, retriever_config=rc, planner_config=pp,
            df_schema=df.columns.to_list(),
            catalog=QuestionCatalog.from_df(
                get_unique_questions_info(df)
            )
        )