import logging
import time
from typing import Callable, Iterable, Optional, TypeVar

import pandas as pd
from openai import RateLimitError
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

T = TypeVar('T')

def retry_call(
    fn: Callable[[], T],
    retries: int = 3,
    base_delay: float = 1.0
) -> T:
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
                logger.warning(f"{e} (attempt {attempt}/{retries}); sleep {delay:.1f}s")
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
    qs_ans = df.groupby(["question"], observed=True).agg(
        waves   = ("wave",   set),
        answers = ("answer", set)
    ).reset_index()

    return qs_ans

def find_top_match(query, choices) -> str:
    match_ = process.extract(
        query, choices,
        scorer=fuzz.token_set_ratio,
        limit=1
    )[0]
    return match_[0]

def remove_defs_and_refs(schema: dict):
    schema = schema.copy()
    defs = schema.pop('$defs', {})

    def resolve(subschema):
        if isinstance(subschema, dict):
            ref = subschema.get('$ref', None)
            if ref:
                _def = ref.split('/')[-1]
                return resolve(defs[_def])
            return {
                _def: resolve(_ref)
                for _def, _ref in subschema.items()
            }
        if isinstance(subschema, list):
            return [resolve(ss) for ss in subschema]
        return subschema
    
    return resolve(schema)
