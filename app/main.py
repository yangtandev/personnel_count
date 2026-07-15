import signal
import sys
import threading
import time

import cv2
from ui.qt_compat import configure_runtime_environment

configure_runtime_environment()
from PyQt5 import QtCore, QtWidgets

from camera.capture import VideoCapture
from config.loader import load_config
from counting.zones import ZoneCounter
from detection.person import PersonDetector
from storage.recorder import Recorder
from ui.window import PersonnelCountWindow


STATUS_TEXT = {
    "starting": "啟動中",
    "camera_waiting": "等待攝影機畫面",
    "detector_error": "偵測異常",
    "waiting": "等待人員通過",
    "middle_only": "人員在通道中",
    "tracking": "追蹤人員移動中",
    "paused_multi_person": "多人同框，暫停計數",
    "incomplete_path": "路徑未完成，未計數",
    "cooldown": "已計數，等待人員離開",
    "unknown_direction": "方向不明，未計數",
    "counted_enter": "已計入進入",
    "counted_exit": "已計入離開",
    "A_to_middle": "從 A 區往中間移動",
    "B_to_middle": "從 B 區往中間移動",
    "returned_A": "折返 A 區，未計數",
    "returned_B": "折返 B 區，未計數",
    "seen_A": "人員位於 A 區",
    "seen_B": "人員位於 B 區",
}


def user_status_text(status):
    return STATUS_TEXT.get(status, "系統運作中")


class SharedState:
    def __init__(self, initial_count):
        self.lock = threading.Lock()
        self.count = int(initial_count)
        self.reset_generation = 0
        self.frames = {}
        self.status = {"top": "starting", "bottom": "starting"}

    def reset_count(self):
        with self.lock:
            self.count = 0
            self.reset_generation += 1


class CameraWorker(threading.Thread):
    def __init__(self, name, camera_url, config, detector, recorder, shared):
        super().__init__(daemon=True)
        self.name = name
        self.capture = VideoCapture(camera_url, config_data={**config.get("camera", {})})
        self.detector = detector
        self.recorder = recorder
        self.shared = shared
        self.counter = ZoneCounter(name, config)
        self.snapshot_interval = float(config["counter"].get("snapshot_interval_sec", 1.0))
        self.last_snapshot_at = 0.0
        self.last_multi_person_at = 0.0
        self.stop_event = threading.Event()
        self.reset_generation = shared.reset_generation

    def stop(self):
        self.stop_event.set()
        self.capture.terminate()

    def run(self):
        while not self.stop_event.is_set():
            frame = self.capture.read()
            if frame is None:
                self._set_status("camera_waiting")
                continue

            now = time.time()
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                self.recorder.record_status(f"{self.name} detector_error: {exc}")
                self._set_status("detector_error")
                self._publish_frame(frame)
                time.sleep(0.2)
                continue

            with self.shared.lock:
                current_count = self.shared.count
                reset_generation = self.shared.reset_generation
            if reset_generation != self.reset_generation:
                self.counter.reset()
                self.reset_generation = reset_generation

            event, status, people = self.counter.update(detections, frame.shape, now, current_count)
            annotated = self._annotate(frame.copy(), detections, people, status)

            if people and now - self.last_snapshot_at >= self.snapshot_interval:
                self.recorder.save_image(self.name, annotated, "detect")
                self.last_snapshot_at = now

            if status == "paused_multi_person":
                if now - self.last_multi_person_at >= self.snapshot_interval:
                    self.recorder.save_image(self.name, annotated, "multi_person")
                    self.recorder.record_status(f"{self.name} paused_multi_person")
                    self.last_multi_person_at = now

            if event is not None:
                with self.shared.lock:
                    should_record = reset_generation == self.shared.reset_generation
                    if should_record:
                        self.shared.count = event.count_after
                if should_record:
                    image_path = self.recorder.save_image(self.name, annotated, event.event)
                    self.recorder.record_event(event, image_path)

            self._set_status(status)
            self._publish_frame(annotated)

    def _set_status(self, status):
        with self.shared.lock:
            self.shared.status[self.name] = status

    def _publish_frame(self, frame):
        with self.shared.lock:
            self.shared.frames[self.name] = frame

    def _annotate(self, frame, detections, people, status):
        self._draw_zones(frame)
        for det in detections:
            color = (0, 180, 0) if det in people else (80, 80, 80)
            x1, y1, x2, y2 = det.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"person {det.conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def _draw_zones(self, frame):
        h, w = frame.shape[:2]
        left_label, right_label = self.counter.zone_layout()
        left_x1, left_x2, right_x1, right_x2 = self.counter.zone_regions(w)
        for label, x1, x2, color in (
            (left_label, left_x1, left_x2, self._zone_color(left_label)),
            (right_label, right_x1, right_x2, self._zone_color(right_label)),
        ):
            x1 = int(x1)
            x2 = int(x2)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.putText(frame, label, (x1 + 12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    def _zone_color(self, label):
        if label == "A":
            return (255, 120, 0)
        if label == "B":
            return (0, 220, 255)
        return (255, 255, 255)


class PersonnelCountApp:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.recorder = Recorder(self.config)
        self.detector = PersonDetector(self.config)
        self.shared = SharedState(self.config["counter"].get("initial_count", 0))
        cameras = self.config["camera"]
        self.workers = [
            CameraWorker("top", cameras["top"], self.config, self.detector, self.recorder, self.shared),
            CameraWorker("bottom", cameras["bottom"], self.config, self.detector, self.recorder, self.shared),
        ]

    def run(self):
        qt_app = QtWidgets.QApplication(sys.argv)
        ui_config = self.config.get("ui", {})
        window = PersonnelCountWindow(
            ui_config.get("window_title", "人員停留數"),
            ui_config.get("camera_names", {}),
        )
        window.reset_counter_requested.connect(self.reset_count)
        if self.config.get("ui", {}).get("fullscreen", True):
            window.showFullScreen()
        else:
            window.show()

        for worker in self.workers:
            worker.start()

        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: self._refresh(window))
        timer.start(100)

        def shutdown(*_):
            for worker in self.workers:
                worker.stop()
            qt_app.quit()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        result = qt_app.exec_()
        shutdown()
        return result

    def _refresh(self, window):
        with self.shared.lock:
            count = self.shared.count
            frames = dict(self.shared.frames)
            status = dict(self.shared.status)
        window.set_count(count)
        window.set_camera_status("top", user_status_text(status.get("top")))
        window.set_camera_status("bottom", user_status_text(status.get("bottom")))
        for name, frame in frames.items():
            window.set_frame(name, frame)

    def reset_count(self):
        self.shared.reset_count()
        self.recorder.record_status("counter reset to 0")


def run(config_path):
    return PersonnelCountApp(config_path).run()
