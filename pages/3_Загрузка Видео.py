"""
Страница для загрузки и обработки видео
"""

import streamlit as st
import tempfile
import os

from utils.video_processor import VideoProcessor
from utils.report_generator import ReportGenerator

st.set_page_config(page_title="Загрузить видео", page_icon="📁", layout="wide")

st.title("📁 Обработка видеофайла")

# Инициализация
if "upload_processor" not in st.session_state:
    st.session_state["upload_processor"] = VideoProcessor()

processor = st.session_state["upload_processor"]

# Загрузка файла
uploaded_file = st.file_uploader(
    "Выберите видеофайл",
    type=["mp4", "avi", "mov", "mkv"],
    help="Поддерживаемые форматы: MP4, AVI, MOV, MKV",
)

# Настройки (оставляем как были)
col1, col2 = st.columns(2)

with col1:
    conf_threshold = st.session_state.get("confidence", 0.5)
    st.info(f"Порог уверенности: {conf_threshold}")

with col2:
    save_output = st.checkbox("Сохранить обработанное видео", value=False)

# Дополнительно: фиксированный пропуск кадров (без ползунка)
SKIP_FRAMES = 3  # детекция на каждом 3‑м кадре

if uploaded_file is not None:
    # Сохранение временного файла
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Кнопка обработки
    if st.button("🚀 Начать обработку", type="primary"):
        processor.clear_history()

        # Прогресс бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        video_placeholder = st.empty()

        # Выходной путь
        output_path = None
        if save_output:
            os.makedirs("reports", exist_ok=True)
            output_path = os.path.join("reports", "processed_video.mp4")

        # Обработка видео
        frame_count = 0
        total_frames = 1000  # примерное значение

        for processed_frame, violations, current_frame in processor.process_video_file(
            video_path, output_path, conf_threshold, skip_frames=SKIP_FRAMES
        ):
            # Обновление интерфейса каждые 5 кадров
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
        status_text.text(f"✅ Обработка завершена! Всего кадров: {frame_count}")

        # Удаление временного файла
        os.unlink(video_path)

        # Отображение результатов
        st.success("Обработка завершена!")

        violations = processor.get_violation_history()

        if violations:
            report_gen = ReportGenerator()

            # Агрегируем по времени → эпизоды
            aggregated = report_gen.aggregate_violations_by_time(
                violations,
                time_window_seconds=60,  # большое окно => одно нарушение на всё видео
            )

            # Берём ОДИН эпизод с максимальной уверенностью
            if aggregated:
                main_violation = max(aggregated, key=lambda v: v.get("confidence", 0.0))
                episodes = [main_violation]
            else:
                episodes = []

            st.subheader(f"📊 Обнаружено нарушений: {len(episodes)}")

            # Генерация отчётов по эпизодам (там уже offender_name и face_path)
            col1, col2 = st.columns(2)

            with col1:
                csv_path = report_gen.create_csv_report(episodes)
                if csv_path:
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            "📄 Скачать CSV отчет",
                            f,
                            file_name="video_violations_report.csv",
                            mime="text/csv",
                        )

            with col2:
                fig = report_gen.create_statistics_plot(episodes)
                if fig:
                    st.pyplot(fig)

            # Текстовый отчёт
            txt_path = report_gen.create_text_report(episodes)
            if txt_path:
                with open(txt_path, "rb") as f:
                    st.download_button(
                        "📝 Скачать текстовый отчёт",
                        f,
                        file_name="video_violations_report.txt",
                        mime="text/plain",
                    )

            # Сохраненное видео
            if save_output and output_path and os.path.exists(output_path):
                st.success(f"Обработанное видео сохранено: {output_path}")
                with open(output_path, "rb") as f:
                    st.download_button(
                        "📹 Скачать обработанное видео",
                        f,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )
        else:
            st.info("Нарушений не обнаружено")
