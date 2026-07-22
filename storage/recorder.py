import logging
import shutil
from datetime import datetime, timedelta
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
        self.camera_names = config.get("ui", {}).get("camera_names", {})
        self.logger = logging.getLogger("personnel_count")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self._log_date = None
        self._log_handler = None
        self._set_log_date(datetime.now().date())

    def _set_log_date(self, log_date):
        if self._log_date == log_date:
            return
        if self._log_handler is not None:
            self.logger.removeHandler(self._log_handler)
            self._log_handler.close()

        self._log_date = log_date
        self._prune_daily_logs(log_date)
        self._log_handler = logging.FileHandler(
            self.log_dir / f"personnel_count_{log_date:%Y-%m-%d}.log",
            encoding="utf-8",
        )
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self._log_handler)

    def _prune_daily_logs(self, today):
        cutoff = today - timedelta(days=6)
        prefix = "personnel_count_"
        for path in self.log_dir.glob(f"{prefix}*.log"):
            try:
                log_date = datetime.strptime(path.stem[len(prefix):], "%Y-%m-%d").date()
            except ValueError:
                continue
            if log_date < cutoff:
                path.unlink()

    def _prune_daily_images(self, today):
        cutoff = today - timedelta(days=6)
        for path in self.img_dir.iterdir():
            if not path.is_dir():
                continue
            try:
                image_date = datetime.strptime(path.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if image_date < cutoff:
                shutil.rmtree(path)

    def _write_log(self, now, message, *args):
        self._set_log_date(now.date())
        self.logger.info(message, *args)

    def save_image(self, camera, frame, reason):
        now = datetime.now()
        self._prune_daily_images(now.date())
        day_dir = self.img_dir / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{camera}_{now.strftime('%H%M%S_%f')}_{reason}.jpg"
        path = day_dir / filename
        cv2.imwrite(str(path), frame)
        return path

    def record_event(self, event, image_path):
        now = datetime.now()
        camera_name = self.camera_names.get(event.camera, event.camera)
        action = "進入" if event.event == "enter" else "離開"
        delta = abs(event.count_after - event.count_before)
        if delta == 0:
            delta = 1
        self._write_log(
            now,
            "%s %s%s %s 人，目前上下設備中共 %s 人。",
            now.strftime("%Y-%m-%d %H:%M:%S"),
            camera_name,
            action,
            delta,
            event.count_after,
        )

    def record_status(self, message):
        pass

    def record_reset(self, count):
        now = datetime.now()
        self._write_log(
            now,
            "%s 觸發重置，目前上下設備中共 %s 人。",
            now.strftime("%Y-%m-%d %H:%M:%S"),
            count,
        )
