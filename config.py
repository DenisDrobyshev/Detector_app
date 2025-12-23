"""
Конфигурация приложения
"""
import os

# Пути к моделям
YOLO_MODEL_PATH = r"D:\PyCharm 2025.1\PycharmProjects\monitoring_app\very_best_best.pt"

INSIGHTFACE_MODEL = "buffalo_l"

# Соответствие индексов классов из модели YOLO и их названий в приложении
VIOLATION_CLASSES = {
    0: "sleeping",
    1: "phone",
    2: "food",
    3: "bottle"
}

# Цвета для визуализации (BGR)
CLASS_COLORS = {
    "food":   (0, 165, 255),    # Оранжевый
    "sleeping": (0, 255, 0),      # Зелёный
    "phone":    (0, 0, 255),      # Красный
    "bottle":   (128, 0, 128)     # Фиолетовый
}

# Параметры детекции
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IOU = 0.35

# Параметры видео
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FPS = 60

# Путь для сохранения отчетов
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
