import logging
from typing import Any, Callable, Optional

import pandas as pd

from capability_spec import OperationType
from utils import find_top_match

logger = logging.getLogger(__name__)


class GroundingError(Exception):
    """Исключение для ошибок при выполнении операций"""
    pass


def op_LOAD_DATA(*, waves: list[str], dataset: pd.DataFrame, **_) -> dict[str, Any]:
    """
    Загружает данные для указанных волн опроса
    
    Args:
        waves: Список волн для загрузки (например, ["2025-01", "2025-02"])
        dataset: Исходный датафрейм со всеми данными
        
    Returns:
        dict с ключом "dataset" и отфильтрованным датафреймом
    """
    logger.info(f"Loading wave data: {waves}")
    
    if "wave" not in dataset.columns:
        raise GroundingError("В данных нет колонки 'wave'")
    
    # Если waves пустой - возвращаем все данные
    if len(waves) == 0:
        logger.info(f"Loaded {len(dataset)} rows for all waves")
        return {"dataset": dataset.copy()}
    
    # Фильтруем по указанным волнам
    ds = dataset[dataset["wave"].isin(waves)].copy()
    logger.info(f"Loaded {len(ds)} rows for waves {waves}")
    
    return {"dataset": ds}


def op_FILTER(
    *,
    dataset: pd.DataFrame,
    question: str,
    answer_values: Optional[list[str]] = None,
    logic: str = "include",
    **_
) -> dict[str, Any]:
    """
    Фильтрует респондентов по ответам на конкретный вопрос
    
    Args:
        dataset: Исходный датафрейм
        question: Вопрос из каталога
        answer_values: Список допустимых ответов (None = все ответившие)
        logic: "include" (оставить эти ответы) или "exclude" (исключить эти ответы)
        
    Returns:
        dict с ключом "filtered_dataset"
    """
    logger.info(f"Filtering by question: {question}, logic: {logic}")
    
    # Проверка наличия вопроса в данных
    if question not in dataset["question"].values:
        closest = find_top_match(question, dataset["question"].drop_duplicates().to_list())
        raise GroundingError(
            f"Вопрос '{question}' не найден в данных. "
            f"Возможно, вы имели в виду: '{closest}'?"
        )
    
    # Фильтруем указанный вопрос
    question_mask = dataset["question"] == question

    # Получаем ответы на указанный вопрос
    ser = dataset.loc[question_mask, "answer"]
    
    # Создаём маску для фильтрации
    if answer_values and len(answer_values) > 0:
        mask = ser.isin(answer_values)
        logger.debug(f"Filtering by specific values: {answer_values}")
    else:
        mask = ser.notna()
        logger.debug("Filtering: all respondents who answered this question")
    
    # Инвертируем маску если нужно исключить
    if logic == "exclude":
        mask = ~mask
        logger.debug("Logic: exclude")
    
    # Получаем список респондентов, прошедших фильтр
    filtered_respondents = dataset[question_mask]\
        .loc[mask, "respondent_uid"].unique()
    
    # Фильтруем весь датасет по этим респондентам
    out = dataset[dataset["respondent_uid"].isin(filtered_respondents)].copy()
    
    logger.info(f"Filtered: {len(dataset)} -> {len(out)} rows")
    logger.info(f"Respondents kept: {len(filtered_respondents)}")

    if len(filtered_respondents) == 0:
        logger.error("Пустой результат фильтрации")
    
    return {"filtered_dataset": out}


def op_PIVOT(
    *,
    dataset: pd.DataFrame,
    question: str,
    **_
) -> dict[str, Any]:
    """
    Создаёт сводную таблицу с распределением ответов на вопрос
    
    Args:
        dataset: Датафрейм (может быть отфильтрованным)
        question: Вопрос для анализа
        
    Returns:
        dict с ключом "pivot" и сводной таблицей
    """
    logger.info(f"Creating pivot for question: {question}")
    
    # Проверка наличия вопроса
    if question not in dataset["question"].values:
        closest = find_top_match(question, dataset["question"].drop_duplicates().to_list())
        logger.debug(f"Normalized question: '{question}' -> '{closest}'")
        question = closest
    
    # Создаём pivot table
    pivot = pd.pivot_table(
        data=dataset[dataset["question"] == question],
        index="answer",
        columns="wave",
        values="respondent_uid",
        aggfunc="nunique",
        fill_value=0,
        observed=True
    )
    
    logger.info(f"Pivot table created: {pivot.shape}")
    logger.debug(f"Pivot columns: {list(pivot.columns)}")
    logger.debug(f"Pivot index: {list(pivot.index)}")
    
    return {"pivot": pivot}


# Реестр операций - связывает OperationType с реализацией
OP_REGISTRY: dict[OperationType, Callable[..., dict[str, Any]]] = {
    OperationType.LOAD_DATA: op_LOAD_DATA,
    OperationType.FILTER: op_FILTER,
    OperationType.PIVOT: op_PIVOT
}
