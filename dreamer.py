import logging

from config import PipelineConfig
from schemas import DreamerOut
from utils import retry_call

logger = logging.getLogger(__name__)


def dreamer(
    user_query: str,
    config: PipelineConfig
) -> DreamerOut:
    dc = config.dreamer_config

    prompt = _make_dreamer_prompt(user_query, config)

    def _call(prompt):
        logger.debug(f"Prompt length: {len(prompt)}")
        
        resp = config.client.chat.completions.create(
            model=dc.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=dc.temperature,
            reasoning_effort=dc.reasoning_effort
        )
        assert resp.choices[0].message, "Empty retriever message"
        return resp.choices[0].message
    
    msg = retry_call(lambda: _call(prompt), dc.retries, dc.base_delay)
    res = DreamerOut(analysis=msg.content)

    if dc.reasoning_effort:
        try:
            if msg.reasoning_details and len(msg.reasoning_details) > 0:
                res.reasoning = msg.reasoning_details[0]["text"]
        except (AttributeError, IndexError) as e:
                logger.warning(f"Error extracting reasoning: {e}")
    
    return res

def _make_dreamer_prompt(user_query: str, config: PipelineConfig) -> str:
    # TODO: протестировать различные промпты

    return f"""
В твоем распоряжении есть результаты опроса населения. Твоя задача построить возможный план анализа для ответа на вопрос пользователя

Запрос пользователя: {user_query}

Список релевантных вопросов из базы с ответами и волнами: {config.relevant_as_value_catalog()}
""".strip()
