import pandas as pd


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