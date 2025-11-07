import logging
import time
from functools import reduce
from typing import Any, Callable, Iterable, Optional

import pandas as pd
from openai import RateLimitError
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

def _parse_formatted_question(col: pd.Series):
    """ Функция, разбивающая отформатированный вопрос на составные части """
    pattern = r"^\[(?P<tag>[^\]]+)?\]\s*(?P<q_clean>[^@|]+)(?:\s*@\s*(?P<detail>[^|]+))?(?:\s*\|\s*(?P<option>.+))?$"

    df = col.str.extract(pattern)

    return df

def get_question_parts(
    col: pd.Series,
    out: list[str] = ["type", "option"]
):
    """
    Вспомогательная функция, возвращающая тип вопроса - MIX, SINGLE, MULTI - и другие составные части
    
    Доступные значения out: 'tag', 'q_clean', 'detail', 'option', 'type'
    """

    QS = _parse_formatted_question(col)

    if "type" in out:
        m_det = QS["detail"].notna()
        m_opt = QS["option"].notna()

        # MIX, SINGLE, MULTI
        QS.loc[m_det, "type"] = "MIX"
        QS.loc[~ m_det & ~ m_opt, "type"] = "SINGLE"
        QS.loc[~ m_det & m_opt, "type"] = "MULTI"

    return QS[out]

def retry_call(
    fn: Callable[[], Any],
    retries: int = 3,
    base_delay: float = 1.0
) -> Any:
    """Повтор с откатом (учитывает rate limit и X-RateLimit-Reset)"""
    delay = base_delay
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except RateLimitError as e:
            last_exc = e
            wait = delay
            try:
                resp = getattr(e, "response", None)
                ts = resp.headers.get("X-RateLimit-Reset") if resp and hasattr(resp, "headers") else None
                if ts:
                    ts = float(ts) / 1000.0  # мс -> сек
                    wait = max(0.0, ts - time.time())
            except Exception:
                pass

            if attempt < retries:
                logger.warning(f"Rate limit (attempt {attempt}/{retries}); sleep {wait:.1f}s")
                time.sleep(wait)
                delay *= 2
            else:
                raise
        except Exception as e:
            last_exc = e
            if attempt < retries:
                logger.warning(f"{type(e).__name__} (attempt {attempt}/{retries}); sleep {delay:.1f}s")
                time.sleep(delay)
                delay *= 2
            else:
                raise last_exc

def take_first_n(x: Iterable, n=30):
    x = sorted(x)

    if len(x) <= n:
        return x
    else:
        return x[:n] + ['...']

def get_unique_questions_info(df: pd.DataFrame):
    qs_ans = df.groupby("question", observed=True).agg(
        waves=("wave", lambda x: set(x)),
        answers=("answer", lambda x: set(x))
    ).reset_index()

    qs_ans = pd.concat([
        get_question_parts(qs_ans["question"], out=['tag', 'q_clean', 'detail', 'option', 'type']),
        qs_ans.drop(columns=["question"])
    ], axis=1)

    qs_ans = qs_ans.groupby(["q_clean","type"]).agg(
        waves   = ("waves",   lambda s: sorted(reduce(set.union, s, set()))),
        answers = ("answers", lambda s: take_first_n(reduce(set.union, s, set()))),
        options = ("option",  lambda s: sorted(s.dropna().unique().tolist())),
        details = ("detail",  lambda s: sorted(s.dropna().unique().tolist())),
    ).reset_index()

    return qs_ans

def find_top_match(query, choices) -> str:
    match_ = process.extract(
        query, choices,
        scorer=fuzz.token_set_ratio,
        limit=1
    )[0]
    return match_[0]
