import logging
from typing import Any, Callable, Optional

import pandas as pd
from utils import find_top_match
from capability_spec import OperationType

logger = logging.getLogger(__name__)


class GroundingError(Exception):
    pass


def op_LOAD_DATA(*, waves: list[str], dataset: pd.DataFrame, **_) -> dict[str, Any]:
    logger.info(f"Loading wave data: {waves}")
    
    if "wave" not in dataset.columns:
        raise GroundingError("В данных нет колонки 'wave'")
    
    if len(waves) == 0:
        logger.info(f"Loaded {len(ds)} rows for all waves")
        return dataset.copy()
    
    ds = dataset[dataset["wave"].isin(waves)].copy()
    logger.info(f"Loaded {len(ds)} rows for wave {waves}")
    
    return {"dataset": ds}


def op_FILTER(
    *,
    dataset: pd.DataFrame,
    question: str,
    answer_values: Optional[list[str]] = None,
    logic: str = "include",
    **_
) -> dict[str, Any]:
    logger.info(f"Filtering by question: {question}, logic: {logic}")
    
    if question not in dataset["question"]:
        raise GroundingError(
            f"Нет вопроса: '{question}'"
            f"Найден: '{find_top_match(question, dataset["question"].drop_duplicates().to_list())}'"
        )
    
    ser = dataset.loc[dataset["question"] == question, "answer"]
    
    if answer_values and len(answer_values) > 0:
        mask = ser.isin(answer_values)
        logger.debug(f"Filtering by values: {answer_values}")
    else:
        mask = ser.notna()
        logger.debug("Filtering: all respondents who answered")
    
    if logic == "exclude":
        mask = ~mask
    
    filtered_respondents = dataset.loc[mask, "respondent_uid"].unique()
    out = dataset[dataset["respondent_uid"].isin(filtered_respondents)].copy()
    
    logger.info(f"Filtered: {len(dataset)} -> {len(out)} rows")
    logger.info(f"Respondents kept: {len(filtered_respondents)}")
    
    return {"filtered_dataset": out}

def op_PIVOT(
    *,
    dataset: pd.DataFrame,
    question: str,
    **_
):
    if question not in dataset["question"]:
        raise GroundingError(
            f"Нет вопроса: '{question}'"
            # f"Найден: '{find_top_match(question, dataset["question"].drop_duplicates())}'"
        )
    
    pivot = pd.pivot_table(
        data=dataset[dataset["question"] == "question"],
        index="answer",
        columns="wave",
        values="respondent_uid",
        aggfunc="nunique"
    )

    return {"pivot": pivot}

# Реестр операций
OP_REGISTRY: dict[OperationType, Callable[..., dict[str, Any]]] = {
    OperationType.LOAD_DATA: op_LOAD_DATA,
    OperationType.FILTER: op_FILTER,
    OperationType.PIVOT: op_PIVOT
}