from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

from .schemas import QuestionInfo


def _clip(xs: list[str], limit: int) -> list[str]:
    return xs[:limit] if limit and len(xs) > limit else xs

class QuestionCatalog(BaseModel):
    questions: list[QuestionInfo] = Field(default_factory=list)

    def as_value_catalog(self, limit: int = 30) -> dict[str, dict]:
        return {
            q.id: {
                "answers": _clip(q.answers, limit),
                "waves": _clip(q.waves, limit),
            }
            for q in self.questions
        }

    def filter(self, selected_questions: list[str]):
        return QuestionCatalog(
            questions=[
                q for q in self.questions 
                if q.id in selected_questions
            ]
        )

    @classmethod
    def from_df(cls, qs_info_df: pd.DataFrame) -> QuestionCatalog:
        questions = [
            QuestionInfo(
                id=getattr(row, "question"),
                waves=getattr(row, "waves"),
                answers=getattr(row, "answers"),
            )
            for row in qs_info_df.itertuples(index=False)
        ]
        return cls(questions=sorted(questions, key=lambda q: q.id))
