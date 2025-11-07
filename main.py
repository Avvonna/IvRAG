import logging
import os
from pathlib import Path

import dotenv
import pandas as pd
from openai import OpenAI

from capability_spec import CapabilitySpec, OperationType
from catalog import QuestionCatalog
from config import (
    ExecutorConfig,
    GrounderConfig,
    PipelineConfig,
    PlannerConfig,
    RetrieverConfig,
)
from pipeline import run_pipeline
from utils import get_unique_questions_info

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
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


def load_data(db_path: str, wave_filter: str = "2025-03") -> pd.DataFrame:
    """Загружает данные из parquet файла"""
    logger.info(f"Loading data from {db_path}")
    
    df = pd.read_parquet(Path(db_path), engine="fastparquet")
    logger.info(f"Full dataset shape: {df.shape}")
    
    if "wave" in df.columns and wave_filter:
        df = df[df["wave"] == wave_filter]
        logger.info(f"Filtered by wave={wave_filter}, new shape: {df.shape}")
    
    return df


def setup_pipeline_config(client: OpenAI, df: pd.DataFrame) -> PipelineConfig:
    """Создаёт конфигурацию pipeline"""
    logger.info("Setting up pipeline configuration")
    
    # Создаём каталог вопросов
    df_questions_info = get_unique_questions_info(df)
    logger.info(f"Questions info shape: {df_questions_info.shape}")
    
    catalog = QuestionCatalog.from_df(df_questions_info)
    logger.info(f"Catalog created with {len(catalog.questions)} questions")
    
    # Конфигурации агентов
    retriever_config = RetrieverConfig()
    planner_config = PlannerConfig()
    grounder_config = GrounderConfig()
    executor_config = ExecutorConfig()
    
    # Сборка полной конфигурации
    config = PipelineConfig(
        client=client,
        retriever_config=retriever_config,
        planner_config=planner_config,
        grounder_config=grounder_config,
        executor_config=executor_config,
        df_schema=df.columns.to_list(),
        catalog=catalog
    )
    
    logger.info("Pipeline configuration ready")
    return config


def main():
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
        config = setup_pipeline_config(client, df)
        
        # Запуск pipeline
        user_query = (
            # "Какое распределение посетителей торговых сетей? "
            "Отфильтруй мне жителей Москвы"# и Московской области"
        )
        
        logger.info(f"\nUSER QUERY: {user_query}\n")
        
        result = run_pipeline(user_query, df, config)
        print(result)
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()