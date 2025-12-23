import streamlit as st
from models.face_recognition import FaceRecognizer

st.set_page_config(page_title="Регистрация студента", page_icon="🧑", layout="wide")
st.title("🧑 Регистрация студента")

if "face_recognizer" not in st.session_state:
    st.session_state["face_recognizer"] = FaceRecognizer()

rec = st.session_state["face_recognizer"]

# Статистика базы
col1, col2 = st.columns(2)
col1.metric("👥 Студентов в базе", len(rec.known_faces))

with col2:
    if st.button("🗑️ Очистить базу", type="secondary"):
        rec.clear_database()
        st.rerun()

st.divider()

# Форма регистрации
name = st.text_input("ФИО студента", help="Иванов Иван Иванович")
uploaded_img = st.file_uploader("📸 Фото лица", type=["jpg", "jpeg", "png"])

if uploaded_img is not None and name:
    import cv2
    import numpy as np

    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Фото")

    with col2:
        if st.button("✅ Добавить студента", type="primary"):
            success = rec.register_face(img, name)
            if success:
                st.success(f"✅ {name} добавлен!")
                st.rerun()
            else:
                st.error("❌ Лицо не найдено!")

# Список студентов
if rec.known_faces:
    st.subheader("📋 База студентов")
    for name in rec.known_faces.keys():
        st.write(f"• {name}")
