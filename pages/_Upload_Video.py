"""
Страница для загрузки и обработки видео
"""

import streamlit as st
import tempfile
import os

from utils.video_processor import VideoProcessor
from utils.report_generator import ReportGenerator

st.set_page_config(page_title="Загрузить видео", page_icon="📁", layout="wide")

st.title("Обработка видеофайла")

# ---------- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ----------

if "upload_processor" not in st.session_state:
    st.session_state["upload_processor"] = VideoProcessor()

if "upload_violations" not in st.session_state:
    st.session_state["upload_violations"] = []  # сырые нарушения по кадрам

if "upload_aggregated" not in st.session_state:
    st.session_state["upload_aggregated"] = []  # агрегированные эпизоды

if "upload_csv_path" not in st.session_state:
    st.session_state["upload_csv_path"] = None

if "upload_txt_path" not in st.session_state:
    st.session_state["upload_txt_path"] = None

if "upload_video_path" not in st.session_state:
    st.session_state["upload_video_path"] = None

processor = st.session_state["upload_processor"]

# ---------- ЗАГРУЗКА ФАЙЛА ----------

uploaded_file = st.file_uploader(
    "Выберите видеофайл",
    type=["mp4", "avi", "mov", "mkv"],
    help="Поддерживаемые форматы: MP4, AVI, MOV, MKV",
)

# Настройки
col1, col2 = st.columns(2)

with col1:
    conf_threshold = st.session_state.get("confidence", 0.5)
    st.info(f"Порог уверенности: {conf_threshold}")

with col2:
    save_output = st.checkbox("Сохранить обработанное видео", value=False)

# ---------- ОБРАБОТКА ВИДЕО ----------

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    if st.button("Начать обработку", type="primary"):
        processor.clear_history()

        progress_bar = st.progress(0)
        status_text = st.empty()
        video_placeholder = st.empty()

        output_path = None
        if save_output:
            os.makedirs("reports", exist_ok=True)
            output_path = os.path.join("reports", "processed_video.mp4")

        frame_count = 0
        total_frames = 1000  # примерное значение

        for processed_frame, violations, current_frame in processor.process_video_file(
            video_path, output_path, conf_threshold
        ):
            if current_frame % 5 == 0:
                video_placeholder.image(
                    processed_frame,
                    channels="BGR",
                )
                progress = min(current_frame / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Обработано кадров: {current_frame}")

            frame_count = current_frame

        progress_bar.progress(1.0)
        status_text.text(f"Обработка завершена! Всего кадров: {frame_count}")

        os.unlink(video_path)

        st.success("Обработка завершена!")

        # сырые нарушения по кадрам
        violations = processor.get_violation_history()
        st.session_state["upload_violations"] = violations

        report_gen = ReportGenerator()

        if violations:
            # агрегируем по времени → эпизоды (одно нарушение на тип/человека)
            aggregated = report_gen.aggregate_violations_by_time(
                violations,
                time_window_seconds=2,
            )
            st.session_state["upload_aggregated"] = aggregated

            # считаем по агрегированным
            st.subheader(f"📊 Обнаружено нарушений: {len(aggregated)}")

            # отчёты тоже по агрегированным
            st.session_state["upload_csv_path"] = report_gen.create_csv_report(aggregated)
            st.session_state["upload_txt_path"] = report_gen.create_text_report(aggregated)

            if save_output and output_path and os.path.exists(output_path):
                st.session_state["upload_video_path"] = output_path
            else:
                st.session_state["upload_video_path"] = None

        else:
            st.info("Нарушений не обнаружено")
            st.session_state["upload_aggregated"] = []
            st.session_state["upload_csv_path"] = None
            st.session_state["upload_txt_path"] = None
            st.session_state["upload_video_path"] = None

# ---------- БЛОК СКАЧИВАНИЯ И ГРАФИКА ----------

if st.session_state["upload_aggregated"]:
    st.subheader("Результаты обработки")

    col1, col2, col3 = st.columns(3)

    report_gen = ReportGenerator()

    # 1) CSV
    with col1:
        if st.session_state["upload_csv_path"]:
            with open(st.session_state["upload_csv_path"], "rb") as f:
                st.download_button(
                    "Скачать CSV отчёт",
                    f.read(),
                    file_name="video_violations_report.csv",
                    mime="text/csv",
                    key="upload_csv_download",
                )

    # 2) График статистики по агрегированным событиям (ЭПИЗОДАМ)
    with col2:
        aggregated = st.session_state["upload_aggregated"]
        fig = report_gen.create_statistics_plot(aggregated)
        if fig:
            st.pyplot(fig)

    # 3) Текстовый отчёт и видео
    with col3:
        if st.session_state["upload_txt_path"]:
            with open(st.session_state["upload_txt_path"], "rb") as f:
                st.download_button(
                    "Скачать текстовый отчёт",
                    f.read(),
                    file_name="video_violations_report.txt",
                    mime="text/plain",
                    key="upload_txt_download",
                )

        if st.session_state["upload_video_path"] and os.path.exists(st.session_state["upload_video_path"]):
            with open(st.session_state["upload_video_path"], "rb") as f:
                st.download_button(
                    "Скачать обработанное видео",
                    f.read(),
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                    key="upload_video_download",
                )
