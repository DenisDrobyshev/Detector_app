"""
Страница для работы с веб-камерой
"""
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
from utils.video_processor import VideoProcessor
from utils.report_generator import ReportGenerator

st.set_page_config(page_title="Веб-камера", page_icon="📹", layout="wide")

st.title("Мониторинг через веб-камеру")

# Инициализация процессора
if 'video_processor' not in st.session_state:
    st.session_state['video_processor'] = VideoProcessor()

processor = st.session_state['video_processor']

# Настройки
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Настройки")

    detect_violations = st.checkbox("Детекция нарушений", value=True)
    recognize_faces = st.checkbox("Распознавание лиц", value=False)

    conf_threshold = st.session_state.get('confidence', 0.5)

    if st.button("Очистить историю"):
        processor.clear_history()
        st.success("История очищена!")


# Callback для обработки видео
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Обработка кадра
        processed_frame, violations = processor.process_frame(
            img,
            detect_violations=detect_violations,
            recognize_faces=recognize_faces,
            conf_threshold=conf_threshold
        )

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")


with col1:
    # Веб-камера стрим
    webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Отображение статистики
st.divider()
st.subheader("Статистика текущей сессии")

violations = processor.get_violation_history()

if violations:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Всего нарушений", len(violations))

    with col2:
        avg_conf = sum(v['confidence'] for v in violations) / len(violations)
        st.metric("Средняя уверенность", f"{avg_conf:.2%}")

    with col3:
        unique_types = len(set(v['class_name'] for v in violations))
        st.metric("Типов нарушений", unique_types)

    # Кнопка для скачивания отчета
    if st.button("Сгенерировать отчет"):
        report_gen = ReportGenerator()
        filepath = report_gen.create_csv_report(violations)
        if filepath:
            with open(filepath, 'rb') as f:
                st.download_button(
                    "Скачать CSV отчет",
                    f,
                    file_name="webcam_report.csv",
                    mime="text/csv"
                )
else:
    st.info("Нарушения пока не обнаружены")
