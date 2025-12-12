import logging
from pathlib import Path

from .config import PipelineConfig
from .executor import executor
from .grounder import grounder
from .planner import planner

# Импортируем самих агентов
from .retriever import retriever
from .state import PipelineStatus, SessionState
from .utils import save_results_to_excel, setup_logging

logger = logging.getLogger("src.engine")

class PipelineEngine:
    def __init__(self, config: PipelineConfig, base_log_dir: str = "logs"):
        self.cfg = config
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(exist_ok=True)

    def create_session(self, user_query: str) -> SessionState:
        """Создает новую сессию"""
        state = SessionState(user_query=user_query)
        
        # Генерируем путь: logs/run_YYYY-MM-DD_UUID
        timestamp = state.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = self.base_log_dir / f"run_{timestamp}_{state.id[:8]}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        state._work_dir = session_dir
        
        # Настраиваем логирование в файл внутри папки сессии
        setup_logging(log_dir=str(self.base_log_dir), run_dir=str(session_dir))
        
        state.save_state()
        logger.info(f"Session created: {state.id}")
        return state

    def load_session(self, session_path: str | Path) -> SessionState:
        """Загружает существующую сессию по пути к папке"""
        session_path = Path(session_path)
        state_path = session_path / "state.json"
        
        if not state_path.exists():
            raise FileNotFoundError(f"State file not found in {session_path}")

        # Подхватываем логирование в ту же папку
        setup_logging(log_dir=str(self.base_log_dir), run_dir=str(session_path))
        
        state = SessionState.load(state_path)
        state._work_dir = session_path
        
        # Восстанавливаем контекст (очень важно для Ретривера/Планера)
        if state.retriever_output:
            self.cfg.update_context(state.retriever_output)
            
        logger.info(f"Session loaded: {state.id} | Status: {state.status}")
        return state

    def step(self, state: SessionState) -> SessionState:
        """Выполняет ровно ОДИН шаг пайплайна."""
        try:
            # 1. Retrieval
            if state.status == PipelineStatus.CREATED:
                if not state.user_query:
                    raise ValueError("User query is empty")

                logger.info("--- [STEP 1] Starting Retrieval ---")
                state.retriever_output = retriever(state.user_query, self.cfg)
                
                # Проверка результата
                if not state.retriever_output or not state.retriever_output.results:
                    logger.error("Retriever found nothing relevant.")
                    raise ValueError("Retriever returned empty results")
                
                self.cfg.update_context(state.retriever_output)
                state.status = PipelineStatus.RETRIEVED
                state.save_state()
                return state

            # 2. Planning
            if state.status == PipelineStatus.RETRIEVED:
                if state.retriever_output is None:
                    raise ValueError("State inconsistency: Status is RETRIEVED, but retriever_output is None")
                self.cfg.update_context(state.retriever_output)

                logger.info("--- [STEP 2] Starting Planning ---")
                plan = planner(state.user_query, self.cfg)
                
                state.planner_output = plan
                state.status = PipelineStatus.PLANNED
                state.save_state()
                return state

            # 3. Grounding
            if state.status == PipelineStatus.PLANNED:
                if state.planner_output is None:
                     raise ValueError("State inconsistency: Status is PLANNED, but planner_output is None")

                logger.info("--- [STEP 3] Starting Grounding ---")
                
                grounded = grounder(state.planner_output)
                
                if not grounded.steps:
                    raise ValueError("Grounder produced 0 executable steps. Check planner output.")

                state.grounder_output = grounded
                state.status = PipelineStatus.GROUNDED
                state.save_state()
                return state

            # 4. Execution
            if state.status == PipelineStatus.GROUNDED:
                if state.grounder_output is None:
                    raise ValueError("State inconsistency: Status is GROUNDED, but grounder_output is None")
                
                if self.cfg.source_df is None:
                    raise ValueError("Source DataFrame is missing in PipelineConfig")

                logger.info("--- [STEP 4] Starting Execution ---")
                
                initial_ctx = {"dataset": self.cfg.source_df}
                result_ctx, provenance = executor(state.grounder_output, initial_ctx)
                
                if not result_ctx:
                     logger.warning("Executor finished but returned empty context.")

                out_file = state._work_dir / "results.xlsx"
                save_results_to_excel(result_ctx, provenance, str(out_file))
                
                state.execution_result_path = str(out_file)
                state.execution_provenance = provenance
                state.status = PipelineStatus.EXECUTED
                state.save_state()
                return state

            if state.status == PipelineStatus.EXECUTED:
                logger.info("Pipeline already finished.")
                return state

        except Exception as e:
            # Любая ошибка переведет статус в FAILED
            state.status = PipelineStatus.FAILED
            state.save_state()
            logger.error(f"Error during step processing: {e}", exc_info=True)
            raise e
        
        return state

    def run(self, user_query: str) -> SessionState:
        """Режим 'Полный автомат'"""
        session = self.create_session(user_query)
        
        while session.status not in [PipelineStatus.EXECUTED, PipelineStatus.FAILED]:
            self.step(session)
            
        return session


    def rewind(self, state: SessionState, target_status: PipelineStatus) -> SessionState:
        """
        Откатывает сессию к заданному статусу, удаляя результаты последующих шагов.
        """
        logger.info(f"Rewinding session {state.id} from {state.status} to {target_status}")
        
        if target_status == PipelineStatus.CREATED:
            # Хотим начать всё с нуля (новый запрос?)
            state.retriever_output = None
            state.planner_output = None
            state.grounder_output = None
            state.execution_result_path = None
            state.execution_provenance = None

        elif target_status == PipelineStatus.RETRIEVED:
            # Хотим перегенерировать ПЛАН. 
            # Оставляем ретривер, удаляем всё, что после.
            state.planner_output = None
            state.grounder_output = None
            state.execution_result_path = None
            state.execution_provenance = None

        elif target_status == PipelineStatus.PLANNED:
            # Хотим переделать GROUNDING (например, поменяли логику маппинга операций)
            # Оставляем план, удаляем граундинг и исполнение.
            state.grounder_output = None
            state.execution_result_path = None
            state.execution_provenance = None
            
        elif target_status == PipelineStatus.GROUNDED:
            # Хотим переделать EXECUTION (например, поправили баг в коде executor.py)
            # Оставляем всё, кроме результатов исполнения.
            state.execution_result_path = None
            state.execution_provenance = None
            
        else:
            raise ValueError(f"Cannot rewind to status {target_status}")

        state.status = target_status
        state.save_state()
        
        # Если откатываемся к RETRIEVED, нужно убедиться, что контекст в конфиге актуален
        if state.retriever_output:
            self.cfg.update_context(state.retriever_output)
            
        logger.info(f"Session rewound successfully. Ready to run step for {target_status}")
        return state