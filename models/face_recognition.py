"""
Класс для распознавания лиц с помощью InsightFace.
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config import INSIGHTFACE_MODEL

# ✅ КОНСТАНТА ПРЯМО ЗДЕСЬ
FACES_DB_PATH = "faces_db.npy"


class FaceRecognizer:
    def __init__(self, model_name: str = INSIGHTFACE_MODEL):
        """Инициализация модели распознавания лиц."""
        print("🔄 Инициализация FaceRecognizer...")
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces: dict[str, np.ndarray] = {}
        self.load_database()
        print(f"✅ Готово. Известных лиц: {len(self.known_faces)}")

    def detect_faces(self, frame):
        """Детекция лиц на кадре."""
        faces = self.app.get(frame)
        return faces

    def register_face(self, frame, name: str) -> bool:
        """Регистрация нового лица в базе."""
        faces = self.detect_faces(frame)
        if faces:
            embedding = faces[0].embedding
            self.known_faces[name] = embedding
            self.save_database()
            return True
        return False

    def recognize_face(self, face_embedding: np.ndarray, threshold: float = 0.5):
        """Распознавание лица по эмбеддингу."""
        if not self.known_faces:
            return None

        min_distance = float("inf")
        recognized_name = None

        for name, known_embedding in self.known_faces.items():
            distance = np.linalg.norm(face_embedding - known_embedding)
            if distance < min_distance and distance < threshold:
                min_distance = distance
                recognized_name = name

        return recognized_name

    def draw_faces(self, frame, faces):
        """Отрисовка рамок вокруг лиц."""
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        return frame

    def save_database(self, path: str = FACES_DB_PATH):
        """Сохраняет базу известных лиц в файл."""
        if not self.known_faces:
            return
        try:
            names = list(self.known_faces.keys())
            embeddings = np.stack([self.known_faces[name] for name in names])
            data = {"names": names, "embeddings": embeddings}
            np.save(path, data)
            print(f"💾 База сохранена: {len(names)} лиц")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")

    def load_database(self, path: str = FACES_DB_PATH):
        """Загружает базу известных лиц из файла."""
        self.known_faces = {}

        if not os.path.exists(path):
            print("📂 Файл базы не найден — пустая база")
            return

        try:
            data = np.load(path, allow_pickle=True).item()
            if isinstance(data, dict) and "names" in data and "embeddings" in data:
                names = data["names"]
                embeddings = data["embeddings"]
                if len(names) == len(embeddings):
                    self.known_faces = {
                        name: embedding for name, embedding in zip(names, embeddings)
                    }
                    print(f"📂 База загружена: {len(self.known_faces)} лиц")
                else:
                    raise ValueError("Несоответствие размеров")
            else:
                raise ValueError("Неверный формат файла")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки '{path}': {e}")
            self.known_faces = {}
            try:
                os.remove(path)
                print("🗑️ Повреждённый файл удалён")
            except:
                pass

    def clear_database(self):
        """Очищает базу лиц."""
        self.known_faces = {}
        if os.path.exists(FACES_DB_PATH):
            os.remove(FACES_DB_PATH)
        print("🗑️ База очищена")
