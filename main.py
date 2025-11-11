import logging
import os
from pathlib import Path
from typing import Literal

import dotenv
import pandas as pd
from openai import OpenAI

from config import PipelineConfig
from pipeline import run_pipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

allowed_loggers = ['retriever', 'pipeline', 'planner', 'grounder', 'executor']
for logger_name in allowed_loggers:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def setup_environment():
    """Загружает переменные окружения"""
    dotenv.load_dotenv()
    
    api_key = os.getenv("OR_API_KEY")
    db_path = os.getenv("DB_PATH")
    
    if not api_key:
        raise ValueError("OR_API_KEY not found in environment")
    if not db_path:
        raise ValueError("DB_PATH not found in environment")
    
    logger.info(f"Environment loaded: DB_PATH={db_path}")
    return api_key, db_path


def load_data(db_path: str, wave_filter: list[str] = ["2025-03"]) -> pd.DataFrame:
    """Загружает данные из parquet файла + фильтр по волнам """
    logger.info(f"Loading data from {db_path}")
    
    df = pd.read_parquet(Path(db_path), engine="fastparquet")
    logger.info(f"Full dataset shape: {df.shape}")
    
    if "wave" in df.columns and wave_filter:
        df = df[df["wave"].isin(wave_filter)]
        logger.info(f"Filtered by waves: {wave_filter}, new shape: {df.shape}")
    
    return df


def main(
    user_query: str,
    retriever_params: dict[Literal["model", "temperature"], str|float] = {},
    planner_params: dict[Literal["model", "temperature"], str|float] = {}
):
    """Главная функция"""
    try:
        # Загрузка окружения
        api_key, db_path = setup_environment()
        
        # Создание клиента
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        logger.info("OpenAI client created")
        
        # Загрузка данных
        df = load_data(db_path)
        
        # Настройка pipeline
        PPL_cfg = PipelineConfig.setup(
            df=df, client=client,
            retriever_params=retriever_params,
            planner_params=planner_params
        )
        
        # Запуск pipeline        
        logger.info(f"USER QUERY: {user_query}")
        
        result = run_pipeline(user_query, df, PPL_cfg)
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Output type: {type(result)}")

        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main(user_query = input("Введите ваш запрос: "))