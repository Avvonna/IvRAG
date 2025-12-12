from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field

from .schemas import GrounderOut, PlannerOut, RetrieverOut, SaveableModel


class PipelineStatus(str, Enum):
    CREATED = "CREATED"
    RETRIEVED = "RETRIEVED"
    PLANNED = "PLANNED"
    GROUNDED = "GROUNDED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"

class SessionState(SaveableModel):
    """
    Единый объект состояния, хранящий весь контекст выполнения запроса.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Входные данные
    user_query: str
    
    # Текущий статус
    status: PipelineStatus = PipelineStatus.CREATED
    
    # Артефакты этапов
    retriever_output: Optional[RetrieverOut] = None
    planner_output: Optional[PlannerOut] = None
    grounder_output: Optional[GrounderOut] = None
    
    # Результаты исполнения (путь к файлу или краткая сводка, т.к. данные могут быть тяжелыми)
    execution_result_path: Optional[str] = None
    execution_provenance: Optional[dict] = None
    
    # Папка, где живет эта сессия (не сохраняем в JSON, задаем при загрузке)
    _work_dir: Path = Path(".")

    def update_timestamp(self):
        self.updated_at = datetime.now()

    def save_state(self):
        """Сохраняет state.json в рабочую директорию сессии"""
        self.update_timestamp()
        path = self._work_dir / "state.json"
        self.save(path)
        return path