import logging
from typing import Any, Callable, Optional

import pandas as pd

from capability_spec import OperationType

logger = logging.getLogger(__name__)


class GroundingError(Exception):
    pass


def op_LOAD_WAVE_DATA(*, wave_id: str, df_full: pd.DataFrame, **_) -> dict[str, Any]:
    logger.info(f"Loading wave data: {wave_id}")
    
    if "wave" not in df_full.columns:
        raise GroundingError("В данных нет колонки 'wave'")
    
    ds = df_full[df_full["wave"] == wave_id].copy()
    logger.info(f"Loaded {len(ds)} rows for wave {wave_id}")
    
    return {"dataset": ds}


def op_FILTER_BY_QUESTION(
    *,
    dataset: pd.DataFrame,
    question_id: str,
    answer_values: Optional[list[str]] = None,
    logic: str = "include",
    **_
) -> dict[str, Any]:
    logger.info(f"Filtering by question: {question_id}, logic: {logic}")
    
    if question_id not in dataset.columns:
        raise GroundingError(f"Нет колонки-вопроса: {question_id}")
    
    ser = dataset[question_id]
    
    if answer_values and len(answer_values) > 0:
        mask = ser.isin(answer_values)
        logger.debug(f"Filtering by values: {answer_values}")
    else:
        mask = ser.notna()
        logger.debug("Filtering: all respondents who answered")
    
    if logic == "exclude":
        mask = ~mask
    
    out = dataset[mask].copy()
    logger.info(f"Filtered: {len(dataset)} -> {len(out)} rows")
    
    return {"filtered_dataset": out}


def op_FILTER_BY_COLUMN(
    *,
    dataset: pd.DataFrame,
    column: str,
    values: Optional[list[Any]] = None,
    op: str = "in",
    **_
) -> dict[str, Any]:
    logger.info(f"Filtering by column: {column}, op: {op}")
    
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
    
    out = dataset[mask].copy()
    logger.info(f"Filtered: {len(dataset)} -> {len(out)} rows")
    
    return {"filtered_dataset": out}


def op_COMPUTE_CROSSTAB(
    *,
    dataset: pd.DataFrame,
    question_id_rows: str,
    question_id_cols: str,
    values: str = "count",
    normalize: str = "none",
    **_
) -> dict[str, Any]:
    logger.info(f"Computing crosstab: {question_id_rows} x {question_id_cols}")
    
    r, c = question_id_rows, question_id_cols
    if r not in dataset.columns or c not in dataset.columns:
        missing = [x for x in (r, c) if x not in dataset.columns]
        raise GroundingError(f"Нет колонок: {missing}")
    
    norm_map = {
        "none": False,
        "rows": "index",
        "columns": "columns",
        "all": "all"
    }
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

    logger.info(f"Crosstab shape: {ct.shape}")
    return {"crosstab_table": ct.reset_index()}


# Реестр операций
OP_REGISTRY: dict[OperationType, Callable[..., dict[str, Any]]] = {
    OperationType.LOAD_WAVE_DATA: op_LOAD_WAVE_DATA,
    OperationType.FILTER_BY_QUESTION: op_FILTER_BY_QUESTION,
    OperationType.FILTER_BY_COLUMN: op_FILTER_BY_COLUMN,
    OperationType.COMPUTE_CROSSTAB: op_COMPUTE_CROSSTAB,
}