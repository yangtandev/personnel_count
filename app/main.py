import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ui.qt_compat import configure_runtime_environment

configure_runtime_environment()
from PyQt5 import QtCore, QtWidgets

from camera.capture import VideoCapture
from config.loader import load_config
from counting.lines import LineCounter
from detection.person import PersonDetector
from storage.recorder import Recorder
from ui.window import PersonnelCountWindow


STATUS_TEXT = {
    "starting": "啟動中",
    "camera_waiting": "等待攝影機畫面",
    "detector_error": "偵測異常",
    "waiting": "等待人員通過",
    "tracking": "追蹤人員移動中",
    "incomplete_path": "路徑未完成，未計數",
    "cooldown": "已計數，等待人員離開",
    "unknown_direction": "方向不明，未計數",
    "counted_enter": "已計入進入",
    "counted_exit": "已計入離開",
    "near_line": "人員接近計數線",
    "crossing_rejected": "未穿越有效線段，不計數",
    "tracking_unavailable": "追蹤 ID 無法使用，停止計數",
    "line_not_configured": "尚未設定計數線",
    "flow_reverse_ignored": "反向事件已由方向鎖忽略",
    "stabilizing": "確認軌跡起始側",
    "rearming": "等待軌跡重新定位",
    "crossing_pending": "確認跨線方向",
    "point_jump": "追蹤點跳動，重新定位",
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
    def __init__(self, name, camera_url, config, recorder, shared):
        super().__init__(daemon=True)
        self.name = name
        self.capture = VideoCapture(camera_url, config_data={**config.get("camera", {})})
        self.detector = PersonDetector(config)
        self.recorder = recorder
        self.shared = shared
        self.counter = LineCounter(name, config)
        self.stop_event = threading.Event()
        self.reset_generation = shared.reset_generation
        self.counter_lock = threading.Lock()

    def reload_config(self, config):
        with self.counter_lock:
            self.counter = LineCounter(self.name, config)

    def stop(self):
        self.stop_event.set()
        self.capture.terminate()
        close = getattr(self.detector, "close", None)
        if close is not None:
            close()

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
            with self.counter_lock:
                if reset_generation != self.reset_generation:
                    self.counter.reset()
                    self.reset_generation = reset_generation

                events, status, people = self.counter.update(detections, frame.shape, now, current_count)
                display_count = events[-1].count_after if events else current_count
                annotated = self._annotate(frame.copy(), people, status, display_count, events)

            for event in events:
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

    def _annotate(self, frame, people, status, count, events=()):
        self._draw_counting_line(frame)
        for track in self.counter.track_visuals():
            trail = np.array([[int(x), int(y)] for x, y in track["trail"]], dtype=np.int32)
            if len(trail) >= 2:
                cv2.polylines(frame, [trail], False, (255, 255, 0), 2)
            if len(trail):
                x, y = trail[-1]
                cv2.putText(
                    frame,
                    f'uid {track["stable_id"]} / id {track["track_id"]} {track["point_source"]} side {track["side"]} {track["state"]}',
                    (int(x) + 8, int(y) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2,
                )
        for det in people:
            color = (0, 180, 0)
            x1, y1, x2, y2 = det.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"person {det.conf:.2f}"
            if getattr(det, "track_id", None) is not None:
                label = f"id {det.track_id} {label}"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            head_box = getattr(det, "head_box", None)
            if head_box is not None:
                hx1, hy1, hx2, hy2 = head_box
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 220, 255), 2)
            point_x, point_y = self.counter.detection_point(det)
            cv2.circle(frame, (int(point_x), int(point_y)), 6, (0, 0, 255), -1)
            cv2.putText(frame, "track-point", (int(point_x) + 8, int(point_y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        event_text = ", ".join(event.event.upper() for event in events)
        status_line = f"COUNT {count} | {status}"
        if event_text:
            status_line += f" | {event_text}"
        cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 720), 50), (0, 0, 0), -1)
        cv2.putText(frame, status_line, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

    def _draw_counting_line(self, frame):
        h, w = frame.shape[:2]
        line = self.counter.counting_line(w, h)
        if line is None:
            return
        start, end = line
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        cv2.line(frame, start, end, (0, 220, 255), 4)
        cv2.circle(frame, start, 7, (0, 220, 255), -1)
        cv2.circle(frame, end, 7, (0, 220, 255), -1)
        cv2.putText(frame, "COUNT LINE", (start[0], max(30, start[1] - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)


class PersonnelCountApp:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config_mtime = self.config_path.stat().st_mtime
        self.config = load_config(config_path)
        self.recorder = Recorder(self.config)
        self.shared = SharedState(self.config["counter"].get("initial_count", 0))
        cameras = self.config["camera"]
        self.single_camera = self._same_camera(cameras.get("top"), cameras.get("bottom"))
        self.workers = [
            CameraWorker("top", cameras["top"], self.config, self.recorder, self.shared),
        ]
        if not self.single_camera:
            self.workers.append(
                CameraWorker("bottom", cameras["bottom"], self.config, self.recorder, self.shared)
            )

    def _same_camera(self, top_url, bottom_url):
        return str(top_url or "").strip() == str(bottom_url or "").strip()

    def run(self):
        qt_app = QtWidgets.QApplication(sys.argv)
        ui_config = self.config.get("ui", {})
        window = PersonnelCountWindow(
            ui_config.get("window_title", "人員停留數"),
            ui_config.get("camera_names", {}),
            single_camera=self.single_camera,
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
        self._reload_config_if_changed()
        with self.shared.lock:
            count = self.shared.count
            frames = dict(self.shared.frames)
            status = dict(self.shared.status)
        window.set_count(count)
        window.set_camera_status("top", user_status_text(status.get("top")))
        if not self.single_camera:
            window.set_camera_status("bottom", user_status_text(status.get("bottom")))
        for name, frame in frames.items():
            window.set_frame(name, frame)

    def reset_count(self):
        self.shared.reset_count()
        self.recorder.record_reset(0)

    def _reload_config_if_changed(self):
        try:
            mtime = self.config_path.stat().st_mtime
        except OSError as exc:
            self.recorder.record_status(f"config_reload_error: {exc}")
            return
        if mtime == self.config_mtime:
            return
        try:
            config = load_config(self.config_path)
            for worker in self.workers:
                worker.reload_config(config)
            self.config = config
            self.config_mtime = mtime
            self.recorder.record_status("config_reloaded")
            print(f"Reloaded config from {self.config_path}")
        except Exception as exc:
            self.recorder.record_status(f"config_reload_error: {exc}")
            print(f"Config reload failed: {exc}")


def run(config_path):
    return PersonnelCountApp(config_path).run()
