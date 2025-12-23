import streamlit as st

st.set_page_config(page_title="Мониторинг дисциплины", layout="wide")  # должно быть одним из первых вызовов [web:557][web:558]

# дефолты (инициализация один раз)
if "confidence" not in st.session_state:
    st.session_state["confidence"] = 0.5
if "iou" not in st.session_state:
    st.session_state["iou"] = 0.45
if "violations" not in st.session_state:
    st.session_state["violations"] = []

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.card {
  padding: 16px 18px;
  border-radius: 14px;
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}
.hero {
  padding: 18px 20px;
  border-radius: 16px;
  background: linear-gradient(135deg, #2F80ED 0%, #56CCF2 100%);
  color: white;
}
.hero h1 {margin: 0 0 6px 0; font-size: 28px;}
.hero p {margin: 0; opacity: 0.95;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Система мониторинга дисциплины</h1>
  <p>Детекция нарушений по видео (YOLO) + опционально распознавание лиц (InsightFace)</p>
</div>
""", unsafe_allow_html=True)

st.write("")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card"><b>Режимы</b><br>Веб‑камера • Загрузка видео</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card"><b>Отчёты</b><br>CSV + графики + текстовый отчёт</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card"><b>Настройки</b><br>Порог confidence и IoU в сайдбаре</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Настройки")

    st.slider(
        "Порог уверенности",
        0.0, 1.0,
        step=0.05,
        key="confidence"  # ключ -> значение будет в st.session_state["confidence"] [web:571]
    )

    st.slider(
        "Порог IoU",
        0.0, 1.0,
        step=0.05,
        key="iou"  # значение будет в st.session_state["iou"] [web:571]
    )

    st.divider()
    st.info("Используйте меню слева для выбора режима и настройки параметров.")
