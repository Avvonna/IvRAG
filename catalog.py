from __future__ import annotations
import pandas as pd
from pydantic import BaseModel, Field

from schemas import QuestionInfo


class QuestionCatalog(BaseModel):
    questions: list[QuestionInfo] = Field(default_factory=list)

    def allowed_question_ids(self) -> list[str]:
        return [q.id for q in self.questions]

    def as_value_catalog(self, limit: int = 30) -> dict[str, dict]:
        def clip(xs: list[str]) -> list[str]:
            return xs[:limit] if limit and len(xs) > limit else xs
        
        return {
            q.id: {
                "answers": clip(q.answers),
                "waves": clip(q.waves),
            }
            for q in self.questions
        }

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> QuestionCatalog:
        questions = [
            QuestionInfo(
                id=getattr(row, "question"),
                waves=getattr(row, "waves"),
                answers=getattr(row, "answers"),
            )
            for row in df.itertuples(index=False)
        ]
        return cls(questions=sorted(questions, key=lambda q: q.id))