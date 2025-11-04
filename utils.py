import logging
import time
from typing import Any, Callable, Optional, Type, TypeVar

import pandas as pd
from openai import OpenAI, RateLimitError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

def _parse_formatted_question(col: pd.Series):
    """ Функция, разбивающая отформатированный вопрос на составные части """
    pattern = r"^\[(?P<tag>[^\]]+)?\]\s*(?P<q_clean>[^@|]+)(?:\s*@\s*(?P<detail>[^|]+))?(?:\s*\|\s*(?P<option>.+))?$"

    df = col.str.extract(pattern)

    return df

def format_question_no_tag(questions: pd.Series) -> pd.Series:
    """
    Собирает строку без [TAG] в формате:
      Q [+ " @ " + DETAIL] [+ " | " + OPTION]
    Разделители добавляются только если соответствующая часть непуста.
    """
    qs_parts = _parse_formatted_question(questions)

    q = qs_parts["q_clean"].fillna("").astype("string").str.strip()
    d = qs_parts["detail"].fillna("").astype("string").str.strip()
    o = qs_parts["option"].fillna("").astype("string").str.strip()

    # если detail/option пусты — оставляем пустую строку, иначе добавляем с разделителем
    detail_part = d.where(d.eq(""), other=(" @ " + d))
    option_part = o.where(o.eq(""), other=(" | " + o))

    res = (q + detail_part + option_part).astype("string")
    res.name = questions.name  # сохраняем имя столбца
    return res

def _retry_call(
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


T = TypeVar("T", bound=BaseModel)

def _get_structured_response(
    client: OpenAI,
    model: str,
    prompt: str,
    response_model: Type[T],
    retries: int = 3,
    base_delay: float = 1.0,
    temperature: float = 0.1,
) -> T:
    """Вывод в формате pydantic схемы"""
    def _call():
        resp = client.responses.parse(
            model=model,
            input=prompt,
            text_format=response_model,
            temperature=temperature,
            store=False,
        )
        return resp.output_parsed

    return _retry_call(_call, retries=retries, base_delay=base_delay)