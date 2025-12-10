import pandas as pd
import pytest

from src.operations import op_FILTER, op_INTERSECT, op_LOAD_DATA, op_PIVOT

# --- Fixtures ---

@pytest.fixture
def sample_data():
    """
    Создаем синтетический датасет.
    Сценарий:
    - User 1: Подходит под фильтр 1 и 2 (Wave 1)
    - User 2: Подходит только под фильтр 1 (Wave 1)
    - User 3: Подходит только под фильтр 2 (Wave 1)
    - User 4: Не подходит никуда (Wave 1)
    - User 5: Подходит под оба (Wave 2)
    """
    data = [
        # --- User 1 (Match Both) ---
        {"respondent_uid": "u1", "wave": "w1", "question": "Gender", "answer": "Male"},
        {"respondent_uid": "u1", "wave": "w1", "question": "Age", "answer": "18-24"},
        {"respondent_uid": "u1", "wave": "w1", "question": "Brand_Usage", "answer": "Yes"},

        # --- User 2 (Match Gender only) ---
        {"respondent_uid": "u2", "wave": "w1", "question": "Gender", "answer": "Male"},
        {"respondent_uid": "u2", "wave": "w1", "question": "Age", "answer": "35-44"}, # Fail filter 2
        {"respondent_uid": "u2", "wave": "w1", "question": "Brand_Usage", "answer": "No"},

        # --- User 3 (Match Age only) ---
        {"respondent_uid": "u3", "wave": "w1", "question": "Gender", "answer": "Female"}, # Fail filter 1
        {"respondent_uid": "u3", "wave": "w1", "question": "Age", "answer": "18-24"},
        {"respondent_uid": "u3", "wave": "w1", "question": "Brand_Usage", "answer": "Yes"},

        # --- User 4 (Match None) ---
        {"respondent_uid": "u4", "wave": "w1", "question": "Gender", "answer": "Female"},
        {"respondent_uid": "u4", "wave": "w1", "question": "Age", "answer": "55+"},
        {"respondent_uid": "u4", "wave": "w1", "question": "Brand_Usage", "answer": "No"},
        
        # --- User 5 (Match Both - Wave 2) ---
        {"respondent_uid": "u5", "wave": "w2", "question": "Gender", "answer": "Male"},
        {"respondent_uid": "u5", "wave": "w2", "question": "Age", "answer": "18-24"},
        {"respondent_uid": "u5", "wave": "w2", "question": "Brand_Usage", "answer": "Yes"},
    ]
    return pd.DataFrame(data)

def test_pipeline_equivalence(sample_data):
    """
    Проверяет эквивалентность двух пайплайнов:
    1. Sequential: Filter A -> Filter B -> Pivot
    2. Intersection: (Filter A) AND (Filter B) -> Pivot
    """
    
    # Параметры фильтрации
    # Фильтр 1: Мужчины
    f1_q = "Gender"
    f1_a = ["Male"]
    
    # Фильтр 2: Возраст 18-24
    f2_q = "Age"
    f2_a = ["18-24"]
    
    # Вопрос для пивота
    target_q = "Brand_Usage"

    # Загрузка данных (общая для обоих путей)
    initial_res = op_LOAD_DATA(waves=[], dataset=sample_data)
    initial_df = initial_res["dataset"]

    # ==========================================
    # ПУТЬ 1: Последовательная фильтрация
    # Filter 1 -> Filter 2 -> Pivot
    # ==========================================
    
    # Шаг 1: Фильтр 1
    p1_step1 = op_FILTER(dataset=initial_df, question=f1_q, answer_values=f1_a)
    df_seq_1 = p1_step1["filtered_dataset"]
    
    # Шаг 2: Фильтр 2 (применяем к результату шага 1)
    p1_step2 = op_FILTER(dataset=df_seq_1, question=f2_q, answer_values=f2_a)
    df_seq_final = p1_step2["filtered_dataset"]
    
    # Шаг 3: Пивот
    p1_res = op_PIVOT(dataset=df_seq_final, questions=[target_q])
    pivot_sequential = p1_res["pivot"]

    # ==========================================
    # ПУТЬ 2: Пересечение (Intersection)
    # (Filter 1, Filter 2) -> Intersect -> Pivot
    # ==========================================
    
    # Шаг 1: Независимый Фильтр 1 (из исходного)
    p2_step1 = op_FILTER(dataset=initial_df, question=f1_q, answer_values=f1_a)
    df_ind_1 = p2_step1["filtered_dataset"]
    
    # Шаг 2: Независимый Фильтр 2 (из исходного)
    p2_step2 = op_FILTER(dataset=initial_df, question=f2_q, answer_values=f2_a)
    df_ind_2 = p2_step2["filtered_dataset"]
    
    # Шаг 3: Пересечение
    p2_intersect = op_INTERSECT(datasets=[df_ind_1, df_ind_2])
    df_intersected = p2_intersect["intersected_dataset"]
    
    # Шаг 4: Пивот
    p2_res = op_PIVOT(dataset=df_intersected, questions=[target_q])
    pivot_intersection = p2_res["pivot"]

    # ==========================================
    # ПРОВЕРКИ
    # ==========================================

    print("\n--- Pipeline 1 (Sequential) Result ---")
    print(pivot_sequential)
    print("\n--- Pipeline 2 (Intersection) Result ---")
    print(pivot_intersection)

    # 1. Проверяем идентичность значений в сводной таблице
    pd.testing.assert_frame_equal(
        pivot_sequential, 
        pivot_intersection,
        obj="Pivot tables should be identical"
    )
    
    # 2. Дополнительная логическая проверка для нашего датасета
    # Ожидаем: U1 (w1) и U5 (w2) должны остаться. U2, U3, U4 исключены.
    # В пивоте для Brand_Usage="Yes": w1=1 (u1), w2=1 (u5).
    assert pivot_sequential.loc["Yes", "w1"] == 1
    assert pivot_sequential.loc["Yes", "w2"] == 1
    
    # Проверка на отсутствие ложных срабатываний
    if "No" in pivot_sequential.index:
        assert pivot_sequential.loc["No", "w1"] == 0

    print("\n✅ TEST PASSED: Operations are equivalent.")
