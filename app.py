import os

import pandas as pd
import streamlit as st
from openai import OpenAI

# Попытка импорта модулей проекта
try:
    from src.config import PipelineConfig
    from src.engine import PipelineEngine
    from src.state import PipelineStatus, SessionState
    from src.utils import load_data, setup_environment
except ImportError as e:
    st.error(f"Ошибка импорта модулей проекта: {e}")
    st.info("Убедитесь, что app.py находится в корневой директории проекта.")
    st.stop()

# --- Константы ---
PAGE_TITLE = "Analytic AI Pipeline"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
LOGS_BASE_DIR = "logs"

AVAILABLE_MODELS = [
    "x-ai/grok-4.1-fast",
    "deepseek/deepseek-chat"
]

# --- Функции форматирования (UI) ---

def render_retriever_view(session: SessionState):
    """Отображает результаты ретривера"""
    if not session.retriever_output:
        return
    
    out = session.retriever_output
    with st.expander(f"📚 1. Retrieval ({len(out.results)} found)", expanded=False):
        for i, q in enumerate(out.results, 1):
            st.markdown(f"**{i}. {q.question}**")
            st.caption(f"Reason: {q.reason}")
        if out.reasoning:
             st.markdown("---")
             st.markdown(f"**Reasoning:** {out.reasoning}")

def render_planner_view(session: SessionState):
    """Отображает результаты планировщика"""
    if not session.planner_output:
        return
        
    out = session.planner_output
    with st.expander(f"🧠 2. Plan ({len(out.steps)} steps)", expanded=False):
        for step in out.steps:
            st.markdown(f"**[{step.id}] {step.operation}**")
            st.text(f"Goal: {step.goal}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("*Inputs:*")
                st.json(step.inputs)
            with col2:
                st.markdown("*Outputs:*")
                st.code(step.outputs)
            st.divider()
        
        st.info(f"Export Variables: {out.export_variables}")

def render_grounder_view(session: SessionState):
    """Отображает результаты граундера (готовность к исполнению)"""
    if not session.grounder_output:
        return
        
    with st.expander(f"🔗 3. Grounded Plan ({len(session.grounder_output.steps)} executable)", expanded=False):
        st.success("Plan validated and linked to executable operations.")
        for step in session.grounder_output.steps:
            st.text(f"[{step.id}] {step.op_type} -> Ready")

def render_execution_view(session: SessionState):
    """Отображает результаты исполнения"""
    if not session.execution_result_path:
        return

    st.divider()
    st.header("📊 Результаты (Execution)")
    
    # Пытаемся загрузить Excel
    try:
        excel_file = pd.ExcelFile(session.execution_result_path)
        sheet_names = excel_file.sheet_names
        
        if len(sheet_names) > 0:
            # Читаем основной отчет
            df_report = pd.read_excel(session.execution_result_path, sheet_name=0)
            st.dataframe(df_report, use_container_width=True)
            st.success(f"Файл сохранен: `{session.execution_result_path}`")
            
            # Кнопка скачивания
            with open(session.execution_result_path, "rb") as f:
                st.download_button(
                    label="📥 Скачать Excel",
                    data=f,
                    file_name="results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("Excel файл пуст.")
            
    except Exception as e:
        st.error(f"Не удалось прочитать файл результатов: {e}")

# --- Вспомогательные функции ---

@st.cache_resource
def init_environment():
    return setup_environment()

@st.cache_resource
def load_dataset(db_path):
    return load_data(db_path, wave_filter=[])

def get_log_dirs(base_dir):
    if not os.path.exists(base_dir):
        return []
    # Ищем папки, начинающиеся с run_
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]
    dirs.sort(reverse=True)
    return dirs

# --- Инициализация ---
st.title(PAGE_TITLE)

try:
    env_api_key, env_db_path = init_environment()
except Exception as e:
    st.error(f"Ошибка инициализации окружения: {e}")
    st.stop()

db = None
try:
    with st.spinner("Загрузка базы данных..."):
        db = load_dataset(env_db_path)
    st.toast(f"База данных загружена: {len(db)} строк.", icon="💾")
except Exception as e:
    st.error(f"Критическая ошибка загрузки БД: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("⚙️ Настройки")

# 1. Окружение
with st.sidebar.expander("1. Окружение", expanded=False):
    api_key = st.text_input("API Key", value=env_api_key, type="password")
    base_url = st.text_input("Base URL", value="https://openrouter.ai/api/v1")
    
    # Фильтр волны
    all_waves = sorted(db["wave"].cat.categories.astype(str).tolist(), reverse=True) if "wave" in db.columns else []
    selected_waves = st.multiselect(
        "Волны (waves)", 
        options=all_waves, 
        default=[all_waves[0]] if all_waves else None
    )

# 2. Retriever
with st.sidebar.expander("2. Retriever", expanded=True):
    retriever_model = st.selectbox("Модель Retriever", options=AVAILABLE_MODELS, index=0)
    retriever_temp = st.slider("Temperature R", 0.0, 1.0, 0.5, 0.1)

# 3. Planner
with st.sidebar.expander("3. Planner", expanded=True):
    planner_model = st.selectbox("Модель Planner", options=AVAILABLE_MODELS, index=0)
    planner_temp = st.slider("Temperature P", 0.0, 1.0, 0.2, 0.1)

# 4. История
st.sidebar.divider()
st.sidebar.header("📂 История")
available_runs = get_log_dirs(LOGS_BASE_DIR)
selected_run_dir = st.sidebar.selectbox("Выбрать прошлый запуск:", options=[""] + available_runs)

# --- Main Logic ---

# 1. Сборка Config и Engine
# Мы собираем их на каждом перезапуске, чтобы настройки из Sidebar применялись сразу
client = OpenAI(base_url=base_url, api_key=api_key)

try:
    # Настраиваем конфиг
    if not selected_waves and all_waves:
        st.warning("⚠️ Не выбрана ни одна волна. Используется последняя доступная.")
        target_waves = [all_waves[0]]
    else:
        target_waves = selected_waves
    
    PPL_cfg = PipelineConfig.setup(
        df=db,
        client=client,
        question_waves=target_waves,
        retriever_params={"model": retriever_model, "temperature": retriever_temp},
        planner_params={"model": planner_model, "temperature": planner_temp}
    )
    
    engine = PipelineEngine(config=PPL_cfg, base_log_dir=LOGS_BASE_DIR)
    
except Exception as e:
    st.error(f"Ошибка настройки Pipeline: {e}")
    st.stop()

# 2. Управление состоянием (Session)

session = None

# Сценарий А: Пользователь выбрал историю
if selected_run_dir:
    session_path = os.path.join(LOGS_BASE_DIR, selected_run_dir)
    try:
        session = engine.load_session(session_path)
        st.info(f"Загружена сессия: `{selected_run_dir}` | Статус: **{session.status.value}**")
        st.markdown(f"**Запрос:** {session.user_query}")
    except Exception as e:
        st.error(f"Ошибка загрузки сессии: {e}")

# Сценарий Б: Новый запуск
else:
    default_query = "Проанализируй изменение потребительского поведения среди тех, кто активно экономит на продуктах"
    user_query = st.text_area("Введите запрос:", value=default_query, height=100)
    
    if st.button("🚀 Запустить", type="primary"):
        try:
            session = engine.create_session(user_query)
            st.rerun() # Перезагрузка, чтобы подхватить ID сессии (если бы мы хранили его в URL, но тут просто обновим UI)
        except Exception as e:
            st.error(f"Ошибка создания сессии: {e}")

# 3. Визуализация и Исполнение (Step Runner)

if session:
    # Отображаем то, что уже есть
    render_retriever_view(session)
    render_planner_view(session)
    render_grounder_view(session)
    render_execution_view(session)
    
    # Если процесс не завершен и не упал
    if session.status not in [PipelineStatus.EXECUTED, PipelineStatus.FAILED]:
        st.write("---")
        col_run, col_stop = st.columns([1, 4])
        
        # Кнопка продолжения
        btn_label = "▶️ Продолжить выполнение"
        if session.status == PipelineStatus.CREATED:
            btn_label = "▶️ Запустить поиск (Retrieval)"
        elif session.status == PipelineStatus.RETRIEVED:
            btn_label = "▶️ Запустить планирование (Planning)"
        elif session.status == PipelineStatus.PLANNED:
            btn_label = "▶️ Запустить валидацию (Grounding)"
        elif session.status == PipelineStatus.GROUNDED:
            btn_label = "▶️ Запустить выполнение (Execution)"
        
        if col_run.button(btn_label, type="primary"):
            
            # --- ЦИКЛ ВЫПОЛНЕНИЯ ---
            status_container = st.status("Выполнение Pipeline...", expanded=True)
            
            try:
                # Мы крутим цикл, пока не дойдем до конца или ошибки
                # step() выполняется синхронно
                
                while session.status not in [PipelineStatus.EXECUTED, PipelineStatus.FAILED]:
                    current_status = session.status
                    status_container.write(f"Запуск шага для статуса: {current_status}...")
                    
                    # ВЫПОЛНЕНИЕ ШАГА
                    session = engine.step(session)
                    
                    # Логирование в UI
                    if session.status == PipelineStatus.RETRIEVED:
                        status_container.write("✅ Retrieval завершен.")
                    elif session.status == PipelineStatus.PLANNED:
                        status_container.write("✅ Planning завершен.")
                    elif session.status == PipelineStatus.GROUNDED:
                        status_container.write("✅ Grounding завершен.")
                    elif session.status == PipelineStatus.EXECUTED:
                        status_container.write("✅ Execution завершен! 🎉")
            
                status_container.update(label="Готово!", state="complete", expanded=False)
                st.rerun() # Обновляем страницу, чтобы показать результаты рендерами выше
                
            except Exception as e:
                status_container.update(label="Ошибка!", state="error")
                st.error(f"Произошла ошибка: {e}")
    
    # Если упал - кнопка отката
    elif session.status == PipelineStatus.FAILED:
        st.error("Выполнение завершилось ошибкой.")
        
        # Пример: кнопка Rewind
        if st.button("⏪ Откатить к Retrieval (сбросить план)"):
            engine.rewind(session, PipelineStatus.RETRIEVED)
            st.rerun()