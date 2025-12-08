import logging
from typing import Any, Callable, Optional

import pandas as pd

from .capability_spec import OperationType
from .utils import find_top_match

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
        logger.warning(f"Вопрос '{question}' не найден в данных. Замена на: '{closest}'")
        question = closest
    
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
        raise GroundingError
    
    return {"filtered_dataset": out}

def op_INTERSECT(
    *,
    datasets: list[pd.DataFrame],
    **_
) -> dict[str, Any]:
    """
    Находит пересечение респондентов из нескольких датасетов (логическое И)
    и склеивает их данные
    
    Args:
        datasets: Список датасетов для пересечения и склейки
        dataset_names: Опциональные имена датасетов для логирования
        
    Returns:
        dict с ключом "intersected_dataset"
    """
    logger.info(f"Finding intersection and concatenating {len(datasets)} datasets")
    
    # Валидация входных данных
    if len(datasets) < 2:
        raise GroundingError("Для операции INTERSECT требуется минимум 2 датасета")
    
    # Проверяем наличие необходимых колонок
    for i, ds in enumerate(datasets):
        if "respondent_uid" not in ds.columns:
            name = f"{datasets[i]=}".split('=')[0]
            raise GroundingError(f"Датасет '{name}' не содержит колонку 'respondent_uid'")
    
    # Получаем множества респондентов из каждого датасета
    respondent_sets = []
    for i, ds in enumerate(datasets):
        respondents = set(ds["respondent_uid"].unique())
        name = f"{datasets[i]=}".split('=')[0]
        logger.info(f"Dataset '{name}': {len(respondents)} уникальных респондентов")
        respondent_sets.append(respondents)
    
    # Находим пересечение
    common_respondents = set.intersection(*respondent_sets)
    logger.info(f"Common respondents found: {len(common_respondents)}")
    
    if len(common_respondents) == 0:
        logger.warning("Пересечение датасетов пустое - нет общих респондентов")
        # Возвращаем пустой датафрейм с той же структурой
        empty_result = datasets[0][datasets[0]["respondent_uid"].isin([])].copy()
        return {"intersected_dataset": empty_result}
    
    # Склеиваем данные всех датасетов, фильтруя только общих респондентов
    filtered_datasets = []
    for i, ds in enumerate(datasets):
        filtered_ds = ds[ds["respondent_uid"].isin(common_respondents)].copy()
        filtered_datasets.append(filtered_ds)
    
    # Конкатенируем все отфильтрованные датасеты
    result_dataset = pd.concat(filtered_datasets, ignore_index=True)
    
    logger.info(f"Intersection result: {len(result_dataset)} rows after concatenation")
    logger.info(f"Unique respondents in result: {result_dataset['respondent_uid'].nunique()}")
    
    return {"intersected_dataset": result_dataset}

def op_UNION(
    *,
    datasets: list[pd.DataFrame],
    **_
) -> dict[str, Any]:
    """
    Находит объединение респондентов из нескольких датасетов (логическое ИЛИ)
    и склеивает их данные
    
    Args:
        datasets: Список датасетов для объединения и склейки
        dataset_names: Опциональные имена датасетов для логирования
        remove_duplicates: Удалять дубликаты строк
        
    Returns:
        dict с ключом "union_dataset"
    """
    logger.info(f"Finding union and concatenating {len(datasets)} datasets")
    
    # Валидация входных данных
    if len(datasets) < 2:
        raise GroundingError("Для операции UNION требуется минимум 2 датасета")
    
    # Проверяем наличие необходимых колонок
    for i, ds in enumerate(datasets):
        if "respondent_uid" not in ds.columns:
            name = f"{datasets[i]=}".split('=')[0]
            raise GroundingError(f"Датасет '{name}' не содержит колонку 'respondent_uid'")
    
    # Логируем статистику по датасетам
    for i, ds in enumerate(datasets):
        name = f"{datasets[i]=}".split('=')[0]
        logger.info(f"Dataset '{name}': {len(ds)} строк, {ds['respondent_uid'].nunique()} уникальных респондентов")
    
    # Простая склейка всех датасетов
    result_dataset = pd.concat(datasets, ignore_index=True)
    
    logger.info(f"Union result: {len(result_dataset)} rows after concatenation")
    logger.info(f"Unique respondents in result: {result_dataset['respondent_uid'].nunique()}")
    
    return {"union_dataset": result_dataset}

def op_PIVOT(
    *,
    dataset: pd.DataFrame,
    questions: Optional[list[str]] = None,
    **_
) -> dict[str, Any]:
    """
    Создаёт сводную таблицу с распределением ответов.
    
    Args:
        dataset: Датафрейм (long format)
        questions: список вопросов (list[str])
        
    Returns:
        dict с ключом "pivot" и сводной таблицей
    """

    logger.info(f"Creating pivot for questions: {questions}")

    if not questions:
        logger.info("No questions provided. Calculating total counts per wave.")
        
        counts_per_wave = dataset.groupby("wave", observed=True)["respondent_uid"].nunique()
        pivot = counts_per_wave.to_frame(name="Total").T
        
        logger.info(f"Simple count pivot created: {pivot.shape}")
        return {"pivot": pivot}
    
    # Нормализация названий вопросов (проверка наличия)
    normalized_questions = []
    all_questions = dataset["question"].drop_duplicates().to_list()
    
    for q in questions:
        if q not in dataset["question"].values:
            closest = find_top_match(q, all_questions) 
            logger.debug(f"Normalized question: '{q}' -> '{closest}'")
            normalized_questions.append(closest)
        else:
            normalized_questions.append(q)
            
    subset = dataset[dataset["question"].isin(normalized_questions)].copy()
    
    if subset.empty:
        logger.warning("Dataset is empty after filtering by questions.")
        return {"pivot": pd.DataFrame()}

    # Трансформация Long -> Wide
    try:
        wide_df = subset.pivot_table(
            index=["respondent_uid", "wave"], 
            columns="question", 
            values="answer", 
            aggfunc="first", # type: ignore
            observed=True
        ).reset_index()
    except Exception as e:
        logger.error(f"Error pivoting to wide format: {e}")
        raise e
    
    logger.debug(f"Wide dataframe shape: {wide_df.shape}")
    
    available_cols = [q for q in normalized_questions if q in wide_df.columns]
    
    pivot = pd.pivot_table(
        data=wide_df,
        index=available_cols,
        columns="wave",
        values="respondent_uid",
        aggfunc="count",
        fill_value=0,
        observed=True
    )
    
    logger.info(f"Final pivot table created: {pivot.shape}")
    
    return {"pivot": pivot}

def op_CALCULATE_AVERAGE(
    *,
    pivot_table: pd.DataFrame,
    scale: dict[str, int | float],
    **_
) -> dict[str, Any]:
    """
    Считает взвешенное среднее по колонкам сводной таблицы.
    
    Formula: Sum(Count * Weight) / Sum(Count)
    Только для строк, которые есть в scale.
    
    Args:
        pivot_table: DataFrame, где index - варианты ответов, columns - волны, values - количество.
        scale: Словарь { "Вариант ответа": Вес }.
        
    Returns:
        dict с ключом "average_table" (DataFrame с одной строкой 'Average')
    """
    logger.info("Calculating weighted average")
    logger.debug(f"Scale provided: {scale}")

    valid_answers = [ans for ans in pivot_table.index if ans in scale]

    if not valid_answers:
        raise GroundingError(
            "Ни один вариант ответа из сводной таблицы не найден в шкале (scale). "
            f"Ответы в таблице: {list(pivot_table.index)}. "
            f"Ключи шкалы: {list(scale.keys())}"
        )

    filtered_pivot = pivot_table.loc[valid_answers].copy()
    weights = pd.Series([scale[ans] for ans in valid_answers], index=valid_answers)

    logger.debug(f"Used answers for calculation: {valid_answers}")

    weighted_counts = filtered_pivot.multiply(weights, axis=0)
    sum_weighted = weighted_counts.sum(axis=0)
    sum_counts = filtered_pivot.sum(axis=0)
    averages = sum_weighted / sum_counts

    result_df = averages.to_frame(name="Average").T

    logger.info(f"Averages calculated for waves: {list(result_df.columns)}")
    logger.debug(f"Values: {result_df.values.tolist()}")

    return {"average_table": result_df}


# Реестр операций - связывает OperationType с реализацией
OP_REGISTRY: dict[OperationType, Callable[..., dict[str, Any]]] = {
    OperationType.LOAD_DATA: op_LOAD_DATA,
    OperationType.FILTER: op_FILTER,
    OperationType.INTERSECT: op_INTERSECT,
    OperationType.UNION: op_UNION,
    OperationType.PIVOT: op_PIVOT,
    OperationType.CALCULATE_AVERAGE: op_CALCULATE_AVERAGE
}
