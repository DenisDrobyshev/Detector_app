"""
Страница со статистикой и аналитикой
"""
import streamlit as st
import pandas as pd
from utils.report_generator import ReportGenerator
import plotly.express as px

st.set_page_config(page_title="Статистика", page_icon="📊", layout="wide")

st.title("Статистика и аналитика")

# Получение данных из обоих процессоров
all_violations = []

if 'video_processor' in st.session_state:
    all_violations.extend(st.session_state['video_processor'].get_violation_history())

if 'upload_processor' in st.session_state:
    all_violations.extend(st.session_state['upload_processor'].get_violation_history())

if not all_violations:
    st.info("Нет данных для отображения. Начните мониторинг или загрузите видео.")
    st.stop()

# Подготовка данных
df_violations = pd.DataFrame([
    {
        'Время': v['timestamp'],
        'Тип': v['class_name'],
        'Уверенность': v['confidence']
    }
    for v in all_violations
])

# Общая статистика
st.subheader("Общая статистика")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Всего нарушений", len(all_violations))

with col2:
    avg_conf = df_violations['Уверенность'].mean()
    st.metric("Средняя уверенность", f"{avg_conf:.2%}")

with col3:
    most_common = df_violations['Тип'].value_counts().index[0]
    st.metric("Самое частое", most_common)

with col4:
    unique_types = df_violations['Тип'].nunique()
    st.metric("Типов нарушений", unique_types)

st.divider()

# Графики
col1, col2 = st.columns(2)

with col1:
    st.subheader("Распределение по типам")

    type_counts = df_violations['Тип'].value_counts()
    fig_pie = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Доля нарушений по типам"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Количество по типам")

    fig_bar = px.bar(
        x=type_counts.index,
        y=type_counts.values,
        labels={'x': 'Тип нарушения', 'y': 'Количество'},
        title="Частота нарушений"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Временная динамика
st.subheader("Динамика нарушений во времени")

df_violations['Час'] = df_violations['Время'].dt.hour
hourly_counts = df_violations.groupby('Час').size().reset_index(name='Количество')

fig_timeline = px.line(
    hourly_counts,
    x='Час',
    y='Количество',
    title="Нарушения по часам",
    markers=True
)
st.plotly_chart(fig_timeline, use_container_width=True)

# Таблица с подробностями
st.subheader("Подробная таблица нарушений")

df_display = df_violations.copy()
df_display['Время'] = df_display['Время'].dt.strftime('%Y-%m-%d %H:%M:%S')
df_display['Уверенность'] = df_display['Уверенность'].apply(lambda x: f"{x:.2%}")

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True
)

# Кнопки для экспорта
st.divider()
col1, col2 = st.columns(2)

# Инициализация путей
if "stats_csv_path" not in st.session_state:
    st.session_state["stats_csv_path"] = None

with col1:
    if st.button("Экспортировать в CSV"):
        report_gen = ReportGenerator()
        st.session_state["stats_csv_path"] = report_gen.create_csv_report(all_violations)

    if st.session_state["stats_csv_path"]:
        with open(st.session_state["stats_csv_path"], "rb") as f:
            st.download_button(
                "⬇Скачать CSV",
                f.read(),
                file_name="full_statistics.csv",
                mime="text/csv",
                key="stats_csv_download",
            )

with col2:
    if st.button("Очистить всю статистику"):
        if 'video_processor' in st.session_state:
            st.session_state['video_processor'].clear_history()
        if 'upload_processor' in st.session_state:
            st.session_state['upload_processor'].clear_history()
        st.success("Статистика очищена!")
        st.rerun()
