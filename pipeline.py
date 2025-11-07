import logging
from typing import Any

import pandas as pd

from config import PipelineConfig
from executor import executor
from grounder import grounder
from planner import planner
from retriever import retriever

logger = logging.getLogger(__name__)


def run_pipeline(
    user_query: str,
    df: pd.DataFrame,
    config: PipelineConfig
) -> dict[str, Any]:
    """
    Запускает полный pipeline обработки запроса
    
    Args:
        user_query: Запрос пользователя
        df: Датафрейм с данными
        config: Конфигурация pipeline
        
    Returns:
        dict с результатами выполнения (финальный контекст)
    """
    logger.info(f"Starting pipeline for query: {user_query}")
    logger.info(f"Dataset shape: {df.shape}")
    
    # 1. Retriever - поиск релевантных вопросов
    logger.info("=" * 60)
    logger.info("STAGE 1: RETRIEVER")
    logger.info("=" * 60)
    
    retriever_out = retriever(user_query, config)
    
    logger.info(f"Selected {len(retriever_out.results)} questions:")
    for i, q in enumerate(retriever_out.results, 1):
        logger.info(f"  {i}. [{q.relevance}/100] '{q.question}'")
        logger.debug(f"     Reason: {q.reason}")
    
    # 2. Planner - составление плана
    logger.info("=" * 60)
    logger.info("STAGE 2: PLANNER")
    logger.info("=" * 60)
    
    planner_out = planner(user_query, retriever_out, config)
    
    logger.info(f"Plan analysis: {planner_out.analysis}")
    logger.info(f"Plan has {len(planner_out.steps)} steps:")
    for i, step in enumerate(planner_out.steps, 1):
        logger.info(f"  {i}. [{step.id}] {step.operation.value}")
        logger.debug(f"     Goal: {step.goal}")        
        logger.debug(f"     Inputs: {step.inputs}")
        logger.debug(f"     Outputs: {step.outputs}")
    
    # 3. Grounder - привязка к реализациям
    logger.info("=" * 60)
    logger.info("STAGE 3: GROUNDER")
    logger.info("=" * 60)
    
    gplan = grounder(planner_out, config)
    logger.info(f"Grounded {len(gplan.steps)} steps successfully")
    
    # 4. Executor - выполнение плана
    logger.info("=" * 60)
    logger.info("STAGE 4: EXECUTOR")
    logger.info("=" * 60)
    
    ctx = {"dataset": df}
    final_ctx = executor(gplan, ctx)
    
    # Вывод результатов
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    result_keys = [k for k in final_ctx.keys() if k != "dataset"]
    logger.info(f"Available results: {result_keys}")
    
    for key in ["dataset", "filtered_dataset", "crosstab_table"]:
        if key in final_ctx:
            data = final_ctx[key]
            if isinstance(data, pd.DataFrame):
                logger.info(f"\n{key}: shape={data.shape}")
                logger.info(f"\n{data.head()}")
            else:
                logger.info(f"\n{key}: {data}")
    
    logger.info("Pipeline completed successfully")
    return final_ctx