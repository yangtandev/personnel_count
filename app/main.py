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
from ui.status import user_status_text
from ui.window import PersonnelCountWindow


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
                annotated = self._annotate(frame.copy(), people)

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

    def _annotate(self, frame, people):
        self._draw_counting_geometry(frame)
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
        return frame

    def _draw_counting_geometry(self, frame):
        h, w = frame.shape[:2]
        mode, geometry = self.counter.counting_geometry(w, h)
        if geometry is None:
            return
        if mode == "corridor":
            colors = {
                "outside": (255, 120, 0),
                "transit": (0, 220, 255),
                "inside": (0, 200, 0),
            }
            labels = {"outside": "OUTSIDE", "transit": "PASSAGE", "inside": "INSIDE"}
            for name, polygon in geometry.items():
                points = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [points], colors[name])
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
                cv2.polylines(frame, [points], True, colors[name], 3)
                x, y = points[0]
                cv2.putText(frame, labels[name], (int(x), max(30, int(y) - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)
            return
        line = geometry
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
