"""
Класс для обработки видеопотоков:
- детекция нарушений (YOLO),
- опциональное распознавание лиц (InsightFace),
- накопление истории нарушений.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Generator

import cv2
import numpy as np

from models.yolo_detector import YOLODetector
from models.face_recognition import FaceRecognizer
from config import REPORTS_DIR


class VideoProcessor:
    def __init__(self):
        """Инициализация процессора видео."""
        self.yolo_detector = YOLODetector()
        self.face_recognizer = FaceRecognizer()
        self.violation_history: List[Dict[str, Any]] = []

        self.segments_dir = os.path.join(REPORTS_DIR, "segments")
        self.faces_dir = os.path.join(REPORTS_DIR, "faces")
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.faces_dir, exist_ok=True)

    # ---------- ОБРАБОТКА ОДНОГО КАДРА ----------

    def process_frame(
        self,
        frame: np.ndarray,
        detect_violations: bool = True,
        recognize_faces: bool = False,
        conf_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        violations: List[Dict[str, Any]] = []
        processed_frame = frame.copy()

        if detect_violations:
            # детекция YOLO
            results = self.yolo_detector.detect(frame, conf_threshold)
            processed_frame = self.yolo_detector.annotate_frame(frame, results)
            violations = self.yolo_detector.get_violations(results)

            if violations:
                timestamp = datetime.now()

                # лицо + распознанное имя
                face_path, offender_name = self.save_face_from_frame(frame, timestamp)

                for v in violations:
                    v["timestamp"] = timestamp
                    v["offender_name"] = offender_name
                    if face_path is not None:
                        v["face_path"] = face_path

                self.violation_history.extend(violations)

        if recognize_faces:
            faces = self.face_recognizer.detect_faces(frame)
            processed_frame = self.face_recognizer.draw_faces(processed_frame, faces)

        return processed_frame, violations

    # ---------- ОБРАБОТКА ВИДЕОФАЙЛА ----------

    def process_video_file(
        self,
        video_path: str,
        output_path: str | None = None,
        conf_threshold: float = 0.5,
    ) -> Generator[Tuple[np.ndarray, List[Dict[str, Any]], int], None, Tuple[bool, int]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield from ()
            return False, 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_index = 0
        self.clear_history()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, violations = self.process_frame(
                frame,
                detect_violations=True,
                recognize_faces=False,
                conf_threshold=conf_threshold,
            )

            if writer:
                writer.write(processed_frame)

            frame_index += 1
            yield processed_frame, violations, frame_index

        cap.release()
        if writer:
            writer.release()

        return True, len(self.violation_history)

    # ---------- УТИЛИТЫ ----------

    def get_violation_history(self) -> List[Dict[str, Any]]:
        return self.violation_history

    def clear_history(self):
        self.violation_history = []

    # ---------- СОХРАНЕНИЕ ЛИЦ И СЕГМЕНТОВ ----------

    def save_face_from_frame(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
    ) -> tuple[str | None, str]:
        """Сохраняет лицо + распознаёт студента. Возвращает путь и имя."""
        if timestamp is None:
            timestamp = datetime.now()

        faces = self.face_recognizer.detect_faces(frame)
        if not faces:
            return None, "Неизвестный"

        face = faces[0]

        # распознавание
        recognized_name = self.face_recognizer.recognize_face(face.embedding)
        final_name = recognized_name if recognized_name else "Неизвестный"

        # вырезаем лицо
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        face_img = frame[y1:y2, x1:x2]

        # сохраняем
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"face_{ts_str}.jpg"
        filepath = os.path.join(self.faces_dir, filename)
        cv2.imwrite(filepath, face_img)

        return filepath, final_name

    def save_segment(
        self,
        frames: List[np.ndarray],
        timestamp_start: datetime,
        fps: float,
    ) -> str | None:
        if not frames:
            return None

        h, w = frames[0].shape[:2]
        ts_str = timestamp_start.strftime("%Y%m%d_%H%M%S")
        filename = f"seg_{ts_str}.mp4"
        filepath = os.path.join(self.segments_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
        writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))

        for f in frames:
            writer.write(f)
        writer.release()

        return filepath
