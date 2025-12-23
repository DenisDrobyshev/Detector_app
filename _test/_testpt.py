from ultralytics import YOLO

from pathlib import Path

model = YOLO(str(Path(__file__).resolve().parent / "best.pt"))


# Вывод информации о модели
print(model.info())
print(model.names)  # Список классов
