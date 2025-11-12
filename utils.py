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

def resolve_refs(schema: dict) -> dict:
    """
    Разворачивает все $ref в JSON Schema, делая схему полностью inline.
    Удаляет $defs после разворачивания.
    """
    import copy
    
    schema = copy.deepcopy(schema)
    defs = schema.pop('$defs', {})
    
    def replace_refs(obj):
        if isinstance(obj, dict):
            if '$ref' in obj:
                # Извлекаем имя из $ref (например, '#/$defs/PlanStep' или 'PlanStep')
                ref_path = obj['$ref']
                ref_name = ref_path.split('/')[-1] if '/' in ref_path else ref_path
                
                if ref_name in defs:
                    # Заменяем $ref на полное определение
                    resolved = copy.deepcopy(defs[ref_name])
                    # Рекурсивно обрабатываем вложенные $ref
                    return replace_refs(resolved)
                else:
                    return obj
            else:
                # Рекурсивно обрабатываем все ключи
                return {k: replace_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_refs(item) for item in obj]
        else:
            return obj
    
    return replace_refs(schema)

def clean_model(obj):
    """ Удаляет из схемы title и additionalProperties"""
    if isinstance(obj, dict):
        return {
            k: clean_model(v) 
            for k, v in obj.items() 
            if k not in ['title', 'additionalProperties']
        }
    elif isinstance(obj, list):
        return [clean_model(item) for item in obj]
    return obj
