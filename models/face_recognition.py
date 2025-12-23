from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import INSIGHTFACE_MODEL


# Корень проекта: .../monitoring_app
BASE_DIR = Path(__file__).resolve().parents[1]
STUDENTS_DB_PATH = BASE_DIR / "students.pkl"


class FaceRecognizer:
    def __init__(
        self,
        model_name: str = INSIGHTFACE_MODEL,
        db_path: Path | str = STUDENTS_DB_PATH,
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
    ):
        """Инициализация модели распознавания лиц и загрузка базы."""
        self.db_path = Path(db_path)

        print("🔄 Инициализация FaceRecognizer...")
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # dict[name] = embedding(np.ndarray)
        self.known_faces: Dict[str, np.ndarray] = {}
        self.load_database()
        print(f"✅ Готово. Известных лиц: {len(self.known_faces)}")

    # ---------- ДЕТЕКЦИЯ / РЕГИСТРАЦИЯ / РАСПОЗНАВАНИЕ ----------

    def detect_faces(self, frame: np.ndarray):
        """Детекция лиц на кадре (BGR). Возвращает список insightface Face."""
        return self.app.get(frame)

    def register_face(self, frame: np.ndarray, name: str) -> bool:
        """Регистрация нового лица в базе (берём первое найденное лицо)."""
        name = (name or "").strip()
        if not name:
            return False

        faces = self.detect_faces(frame)
        if not faces:
            return False

        embedding = np.asarray(faces[0].embedding, dtype=np.float32)
        self.known_faces[name] = embedding
        self.save_database()
        return True

    def recognize_face(self, face_embedding: np.ndarray, threshold: float = 0.9) -> Optional[str]:
        """
        Распознавание по эмбеддингу.
        Сравнение по L2 расстоянию: чем меньше — тем ближе.
        threshold подбирается на данных (0.9 часто мягче, чем 0.5).
        """
        if not self.known_faces:
            return None

        emb = np.asarray(face_embedding, dtype=np.float32)

        min_distance = float("inf")
        recognized_name = None

        for name, known_emb in self.known_faces.items():
            d = float(np.linalg.norm(emb - known_emb))
            if d < min_distance:
                min_distance = d
                recognized_name = name

        if min_distance < threshold:
            return recognized_name
        return None

    def recognize_faces_on_frame(self, frame: np.ndarray, threshold: float = 0.9):
        """
        Возвращает список dict:
        [{bbox: (x1,y1,x2,y2), name: str|None, distance: float}, ...]
        """
        out = []
        faces = self.detect_faces(frame)
        for face in faces:
            bbox = tuple(face.bbox.astype(int).tolist())  # x1,y1,x2,y2
            emb = np.asarray(face.embedding, dtype=np.float32)

            name = None
            dist = None

            if self.known_faces:
                # ищем ближайшего
                best_name = None
                best_dist = float("inf")
                for n, known_emb in self.known_faces.items():
                    d = float(np.linalg.norm(emb - known_emb))
                    if d < best_dist:
                        best_dist = d
                        best_name = n
                dist = best_dist
                if best_dist < threshold:
                    name = best_name

            out.append({"bbox": bbox, "name": name, "distance": dist})
        return out

    def draw_faces(self, frame: np.ndarray, faces_info, show_unknown: bool = True) -> np.ndarray:
        """Отрисовка рамок и имён поверх кадра."""
        img = frame.copy()
        for item in faces_info:
            x1, y1, x2, y2 = item["bbox"]
            name = item.get("name")

            if name is None and not show_unknown:
                continue

            color = (0, 255, 0) if name else (0, 255, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = name if name else "Unknown"
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return img

    # ---------- БАЗА ЛИЦ ----------

    def save_database(self):
        """Сохраняет базу известных лиц в students.pkl (pickle)."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "wb") as f:
                pickle.dump(self.known_faces, f)
            print(f"💾 База сохранена: {len(self.known_faces)} лиц -> {self.db_path}")
        except Exception as e:
            print(f"❌ Ошибка сохранения базы: {e}")

    def load_database(self):
        """Загружает базу известных лиц из students.pkl (pickle)."""
        self.known_faces = {}

        if not self.db_path.exists():
            print(f"📂 Файл базы не найден — пустая база: {self.db_path}")
            return

        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                # гарантируем np.ndarray float32
                cleaned = {}
                for name, emb in data.items():
                    if not isinstance(name, str):
                        continue
                    cleaned[name] = np.asarray(emb, dtype=np.float32)
                self.known_faces = cleaned
                print(f"📂 База загружена: {len(self.known_faces)} лиц <- {self.db_path}")
            else:
                raise ValueError("Неверный формат базы (ожидался dict)")

        except Exception as e:
            print(f"⚠️ Ошибка загрузки '{self.db_path}': {e}")
            self.known_faces = {}
            # если файл поврежден — можно удалить, чтобы не ломал запуск
            try:
                os.remove(self.db_path)
                print("🗑️ Повреждённый файл базы удалён")
            except Exception:
                pass

    def clear_database(self):
        """Очищает базу лиц и удаляет файл."""
        self.known_faces = {}
        try:
            if self.db_path.exists():
                self.db_path.unlink()
        except Exception:
            pass
        print("🗑️ База очищена")
