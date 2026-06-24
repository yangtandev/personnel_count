import csv
import logging
from datetime import datetime
from pathlib import Path

import cv2

from config.loader import project_path


class Recorder:
    def __init__(self, config):
        storage = config["storage"]
        self.img_dir = Path(storage.get("img_log_dir", "img_log/personnel_count"))
        self.log_dir = Path(storage.get("log_dir", "log"))
        if not self.img_dir.is_absolute():
            self.img_dir = project_path(self.img_dir)
        if not self.log_dir.is_absolute():
            self.log_dir = project_path(self.log_dir)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.log_dir / "personnel_count_events.csv"
        self.logger = logging.getLogger("personnel_count")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        handler = logging.FileHandler(self.log_dir / "personnel_count.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())
        self._ensure_csv()

    def _ensure_csv(self):
        if self.events_path.exists():
            return
        with self.events_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "timestamp",
                "camera",
                "direction",
                "event",
                "count_before",
                "count_after",
                "confidence",
                "image_path",
                "status",
            ])

    def save_image(self, camera, frame, reason):
        day_dir = self.img_dir / datetime.now().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{camera}_{datetime.now().strftime('%H%M%S_%f')}_{reason}.jpg"
        path = day_dir / filename
        cv2.imwrite(str(path), frame)
        return path

    def record_event(self, event, image_path):
        ts = datetime.now().isoformat(timespec="seconds")
        rel_path = Path(image_path)
        try:
            rel_path = rel_path.relative_to(project_path())
        except ValueError:
            pass
        with self.events_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                ts,
                event.camera,
                event.direction,
                event.event,
                event.count_before,
                event.count_after,
                f"{event.confidence:.3f}",
                str(rel_path),
                event.status,
            ])
        self.logger.info(
            "%s %s %s count %s -> %s (%s)",
            event.camera,
            event.direction,
            event.event,
            event.count_before,
            event.count_after,
            event.status,
        )

    def record_status(self, message):
        self.logger.info(message)
