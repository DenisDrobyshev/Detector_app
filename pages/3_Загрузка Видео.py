"""
Страница для загрузки и обработки видео
"""

import os
import tempfile
import streamlit as st

from utils.video_processor import VideoProcessor
from utils.report_generator import ReportGenerator

st.set_page_config(page_title="Загрузить видео", page_icon="📁", layout="wide")
st.title("📁 Обработка видеофайла")

# -------------------------
# Init
# -------------------------
if "upload_processor" not in st.session_state:
    st.session_state["upload_processor"] = VideoProcessor()
processor = st.session_state["upload_processor"]

# Храним результаты (чтобы скачивать много раз после rerun)
for k in ("csv_bytes", "txt_bytes", "video_bytes", "last_uploaded_name"):
    if k not in st.session_state:
        st.session_state[k] = None

def clear_download_cache():
    st.session_state["csv_bytes"] = None
    st.session_state["txt_bytes"] = None
    st.session_state["video_bytes"] = None

# -------------------------
# Upload + settings
# -------------------------
uploaded_file = st.file_uploader(
    "Выберите видеофайл",
    type=["mp4", "avi", "mov", "mkv"],
    help="Поддерживаемые форматы: MP4, AVI, MOV, MKV",
)

col1, col2 = st.columns(2)
with col1:
    conf_threshold = st.session_state.get("confidence", 0.5)
    st.info(f"Порог уверенности: {conf_threshold}")
with col2:
    save_output = st.checkbox("Сохранить обработанное видео", value=False)

SKIP_FRAMES = 3  # детекция на каждом 3‑м кадре

# Если пользователь выбрал другой файл — сбрасываем старые результаты скачивания
if uploaded_file is not None:
    if st.session_state["last_uploaded_name"] != uploaded_file.name:
        st.session_state["last_uploaded_name"] = uploaded_file.name
        clear_download_cache()

# -------------------------
# Processing
# -------------------------
if uploaded_file is not None and st.button("🚀 Начать обработку", type="primary"):
    # очищаем историю и прошлые результаты
    processor.clear_history()
    clear_download_cache()

    # сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    progress_bar = st.progress(0)
    status_text = st.empty()
    video_placeholder = st.empty()

    output_path = None
    if save_output:
        os.makedirs("reports", exist_ok=True)
        output_path = os.path.join("reports", "processed_video.mp4")

    frame_count = 0
    total_frames = 1000  # если хочешь точно — надо считать кадры через cv2.VideoCapture

    for processed_frame, violations, current_frame in processor.process_video_file(
        video_path, output_path, conf_threshold, skip_frames=SKIP_FRAMES
    ):
        if current_frame % 5 == 0:
            video_placeholder.image(processed_frame, channels="BGR")
            progress_bar.progress(min(current_frame / total_frames, 1.0))
            status_text.text(f"Обработано кадров: {current_frame}")
        frame_count = current_frame

    progress_bar.progress(1.0)
    status_text.text(f"✅ Обработка завершена! Всего кадров: {frame_count}")

    # удаляем временный входной файл
    try:
        os.unlink(video_path)
    except OSError:
        pass

    st.success("Обработка завершена!")

    violations = processor.get_violation_history()
    if not violations:
        st.info("Нарушений не обнаружено")
    else:
        report_gen = ReportGenerator()

        aggregated = report_gen.aggregate_violations_by_time(
            violations,
            time_window_seconds=60,
        )

        episodes = [max(aggregated, key=lambda v: v.get("confidence", 0.0))] if aggregated else []
        st.subheader(f"📊 Обнаружено нарушений: {len(episodes)}")

        # ---- генерируем отчёты в файлы и сразу читаем в bytes для повторного скачивания
        csv_path = report_gen.create_csv_report(episodes)
        if csv_path and os.path.exists(csv_path):
            st.session_state["csv_bytes"] = open(csv_path, "rb").read()

        txt_path = report_gen.create_text_report(episodes)
        if txt_path and os.path.exists(txt_path):
            st.session_state["txt_bytes"] = open(txt_path, "rb").read()

        if save_output and output_path and os.path.exists(output_path):
            st.session_state["video_bytes"] = open(output_path, "rb").read()

# -------------------------
# Download area (ALWAYS visible if data exists)
# -------------------------
st.divider()
st.subheader("⬇️ Скачивание результатов")

d1, d2, d3 = st.columns(3)

with d1:
    if st.session_state.get("csv_bytes"):
        st.download_button(
            "📄 Скачать CSV отчет",
            data=st.session_state["csv_bytes"],
            file_name="video_violations_report.csv",
            mime="text/csv",
            key="dl_csv",
            on_click="ignore",  # чтобы не пересобирать страницу при клике [web:619]
        )
    else:
        st.caption("CSV появится после обработки.")

with d2:
    if st.session_state.get("txt_bytes"):
        st.download_button(
            "📝 Скачать текстовый отчёт",
            data=st.session_state["txt_bytes"],
            file_name="video_violations_report.txt",
            mime="text/plain",
            key="dl_txt",
            on_click="ignore",  # [web:619]
        )
    else:
        st.caption("TXT появится после обработки.")

with d3:
    if st.session_state.get("video_bytes"):
        st.download_button(
            "📹 Скачать обработанное видео",
            data=st.session_state["video_bytes"],
            file_name="processed_video.mp4",
            mime="video/mp4",
            key="dl_video",
            on_click="ignore",  # [web:619]
        )
    else:
        st.caption("Видео появится после обработки (если включить сохранение).")
