"""
Генерация отчетов о нарушениях.
"""

import os
from datetime import datetime
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import REPORTS_DIR


class ReportGenerator:
    def __init__(self):
        """Инициализация генератора отчетов."""
        self.reports_dir = REPORTS_DIR

    # ---------- CSV ОТЧЕТ ----------

    def create_csv_report(self, violations: List[Dict[str, Any]], filename: str | None = None) -> str | None:
        """
        Создание CSV отчета по списку нарушений (желательно уже агрегированному).

        Args:
            violations: список нарушений (словарей).
            filename: имя файла (опционально).

        Returns:
            filepath: путь к созданному файлу или None, если нарушений нет.
        """
        if not violations:
            return None

        data = []
        for v in violations:
            data.append({
                'Время': v['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Тип нарушения': v['class_name'],
                'Уверенность': f"{v['confidence']:.2%}",
                'Координаты': f"{v['bbox']}",
                'Нарушитель': v.get('offender_name', 'Неизвестный'),
                'Фото лица': v.get('face_path', ''),
            })

        df = pd.DataFrame(data)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violations_report_{timestamp}.csv"

        filepath = os.path.join(self.reports_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")

        return filepath

    # ---------- ГРАФИКИ ----------

    def create_statistics_plot(self, violations: List[Dict[str, Any]]):
        """
        Создание графика статистики нарушений.

        Returns:
            fig: объект matplotlib.figure.Figure или None.
        """
        if not violations:
            return None

        violation_types = [v.get("class_name", "unknown") for v in violations]
        df = pd.DataFrame({"Тип": violation_types})
        counts = df["Тип"].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis")
        ax.set_title("Статистика нарушений", fontsize=16, fontweight="bold")
        ax.set_xlabel("Тип нарушения", fontsize=12)
        ax.set_ylabel("Количество", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def get_summary_statistics(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Получение сводной статистики по списку нарушений.
        """
        if not violations:
            return {
                "total": 0,
                "by_type": {},
                "avg_confidence": 0.0,
            }

        total = len(violations)

        by_type: Dict[str, int] = {}
        for v in violations:
            class_name = v.get("class_name", "unknown")
            by_type[class_name] = by_type.get(class_name, 0) + 1

        avg_confidence = sum(v.get("confidence", 0.0) for v in violations) / total

        return {
            "total": total,
            "by_type": by_type,
            "avg_confidence": avg_confidence,
        }

    def aggregate_violations_by_time(self, violations, time_window_seconds: int = 1):
        """
        Упрощённая агрегация нарушений:
        - группируем по типу и времени (с точностью до time_window_seconds),
        - внутри группы оставляем запись с максимальной уверенностью.
        """
        if not violations:
            return []

        grouped = {}
        for v in violations:
            ts = v.get("timestamp")
            class_name = v.get("class_name", "unknown")

            if ts is None:
                key = (class_name, None)
            else:
                rounded_ts = ts.replace(microsecond=0)
                if time_window_seconds > 1:
                    sec = (rounded_ts.second // time_window_seconds) * time_window_seconds
                    rounded_ts = rounded_ts.replace(second=sec)
                key = (class_name, rounded_ts)

            current_best = grouped.get(key)
            if current_best is None or v.get("confidence", 0.0) > current_best.get("confidence", 0.0):
                grouped[key] = v

        return list(grouped.values())

    # ---------- ТЕКСТОВЫЙ ОТЧЕТ В ФОРМАТЕ МЕТОДИЧКИ ----------

    def create_text_report(self, violations: List[Dict[str, Any]],
                           monitoring_start: datetime | None = None,
                           monitoring_end: datetime | None = None,
                           filename: str | None = None) -> str | None:
        """
        Создание текстового отчета.

        ВАЖНО: для корректного подсчёта лучше передавать уже агрегированный список нарушений.
        """
        if not violations:
            return None

        # Если на вход дали сырой список, можно дополнительно агрегировать
        aggregated = self.aggregate_violations_by_time(violations)

        violations_sorted = sorted(
            aggregated,
            key=lambda v: v.get("timestamp", datetime.min)
        )

        if monitoring_start is None:
            first_ts = violations_sorted[0].get("timestamp", datetime.now())
            monitoring_start = first_ts if isinstance(first_ts, datetime) else datetime.now()
        if monitoring_end is None:
            last_ts = violations_sorted[-1].get("timestamp", datetime.now())
            monitoring_end = last_ts if isinstance(last_ts, datetime) else datetime.now()

        date_str = monitoring_start.strftime("%Y-%m-%d")
        start_str = monitoring_start.strftime("%H:%M:%S")
        end_str = monitoring_end.strftime("%H:%M:%S")

        lines: list[str] = []
        sep = "═" * 72
        sub_sep = "─" * 72

        # Заголовок
        lines.append(sep)
        lines.append(" ОТЧЁТ О НАРУШЕНИЯХ ДИСЦИПЛИНЫ")
        lines.append(f" Дата: {date_str}")
        lines.append(f" Время мониторинга: {start_str} - {end_str}")
        lines.append(sep)
        lines.append("")

        # Основные записи
        for idx, v in enumerate(violations_sorted, start=1):
            ts = v.get("timestamp", "")
            if isinstance(ts, datetime):
                ts_str = ts.strftime("%H:%M:%S")
            else:
                ts_str = str(ts)

            class_name = v.get("class_name", "unknown")
            offender_name = v.get("offender_name", "Неизвестный")
            segment_path = v.get("segment_path", "")
            face_path = v.get("face_path", "")
            confidence = v.get("confidence", None)

            lines.append(f"№{idx}. НАРУШЕНИЕ")
            lines.append(sub_sep)
            lines.append(f" Время: {ts_str}")
            lines.append(f" Тип: {class_name}")
            lines.append(f" Нарушитель: {offender_name}")
            if segment_path:
                lines.append(f" Видеозапись: {segment_path}")
            if face_path:
                lines.append(f" Фото лица: {face_path}")
            if confidence is not None:
                lines.append(f" Уверенность: {confidence:.2%}")
            lines.append(sub_sep)
            lines.append("")

        # Итоги по агрегированным нарушениям
        stats = self.get_summary_statistics(violations_sorted)
        lines.append(sep)
        lines.append(" ИТОГО")
        lines.append(sep)
        lines.append(f" Всего нарушений: {stats['total']}")
        lines.append("")
        lines.append(" По типам:")
        for t, c in stats["by_type"].items():
            lines.append(f" - {t}: {c}")
        lines.append(sep)

        text = "\n".join(lines)

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"violations_report_{ts}.txt"

        filepath = os.path.join(self.reports_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        return filepath
