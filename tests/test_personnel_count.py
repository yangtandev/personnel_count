import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/personnel_count_test_mpl")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/personnel_count_test_yolo")


class DummyDetection:
    def __init__(self, center_x, conf=0.9):
        self.box = (center_x - 2, 10, center_x + 2, 90)
        self.conf = conf

    @property
    def center_x(self):
        return (self.box[0] + self.box[2]) / 2

    @property
    def area(self):
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


def counter_config():
    return {
        "zones": {
            "left_ratio": 0.25,
            "right_ratio": 0.75,
            "zone_width_ratio": 0.18,
            "mode": "side",
            "labels": {
                "top": {"left": "A", "right": "B"},
                "bottom": {"left": "B", "right": "A"},
            },
        },
        "direction": {
            "top": {"A_to_B": "enter", "B_to_A": "exit"},
            "bottom": {"B_to_A": "exit", "A_to_B": "enter"},
        },
        "counter": {
            "min_person_area_ratio": 0,
            "lost_timeout_sec": 2,
            "event_cooldown_sec": 0,
            "require_middle": False,
        },
    }


class CounterTest(unittest.TestCase):
    def run_path(self, camera, xs, count=0):
        from counting.zones import ZoneCounter

        counter = ZoneCounter(camera, counter_config())
        event = None
        for index, x in enumerate(xs, start=1):
            event, _, _ = counter.update([DummyDetection(x)], (100, 100, 3), index * 0.1, count)
            if event:
                count = event.count_after
        return event, count

    def test_top_camera_directions(self):
        self.assertEqual(self.run_path("top", [25, 50, 75], 0)[1], 1)
        self.assertEqual(self.run_path("top", [75, 50, 25], 3)[1], 2)

    def test_bottom_camera_directions(self):
        event, count = self.run_path("bottom", [25, 50, 75], 0)
        self.assertEqual(count, 0)
        self.assertEqual(event.status, "blocked_negative_count")
        self.assertEqual(self.run_path("bottom", [75, 50, 25], 2)[1], 3)

    def test_fast_crossing_can_skip_middle(self):
        self.assertEqual(self.run_path("top", [25, 75], 0)[1], 1)
        self.assertEqual(self.run_path("bottom", [75, 25], 0)[1], 1)

    def test_fast_crossing_can_skip_target_band(self):
        self.assertEqual(self.run_path("top", [25, 95], 0)[1], 1)
        self.assertEqual(self.run_path("bottom", [95, 25], 0)[1], 1)

    def test_foldback_then_complete_counts_once(self):
        event, count = self.run_path("top", [25, 50, 25, 50, 75], 0)
        self.assertEqual(count, 1)
        self.assertEqual(event.event, "enter")

    def test_negative_count_blocked(self):
        event, count = self.run_path("top", [75, 50, 25], 0)
        self.assertEqual(count, 0)
        self.assertEqual(event.status, "blocked_negative_count")

    def test_multi_person_pauses(self):
        from counting.zones import ZoneCounter

        event, status, people = ZoneCounter("top", counter_config()).update(
            [DummyDetection(25), DummyDetection(75)], (100, 100, 3), 1.0, 0
        )
        self.assertIsNone(event)
        self.assertEqual(status, "paused_multi_person")
        self.assertEqual(len(people), 2)


class QtCompatTest(unittest.TestCase):
    def test_cv2_cannot_leave_qt_plugin_path_on_cv2_plugins(self):
        import cv2  # noqa: F401
        from ui.qt_compat import configure_runtime_environment

        configure_runtime_environment()
        self.assertIn("PyQt5", os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"])
        self.assertNotIn("/cv2/qt/plugins", os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"])

    def test_window_can_be_created_offscreen(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt5 import QtWidgets
        from ui.window import PersonnelCountWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PersonnelCountWindow(camera_names={"top": "井上", "bottom": "井底"})
        window.set_count(7)
        window.set_camera_status("top", "等待人員通過")
        window.set_camera_status("bottom", "多人同框，暫停計數")
        self.assertEqual(window.count_label.text(), "人員停留數：7")
        self.assertEqual(window.camera_titles["top"], "井上")
        self.assertEqual(window.camera_titles["bottom"], "井底")
        self.assertEqual(window.camera_statuses["top"], "等待人員通過")
        self.assertEqual(window.camera_statuses["bottom"], "多人同框，暫停計數")
        window.close()
        app.processEvents()

    def test_window_fonts_scale_with_size(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt5 import QtWidgets
        from ui.window import PersonnelCountWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PersonnelCountWindow()
        window.resize(1920, 1080)
        window._update_font_sizes()
        self.assertEqual(window.bar_font, 60)
        self.assertGreaterEqual(window.bar_height, 58)
        self.assertIn("font-size:83px", window.count_label.styleSheet())
        window.close()
        app.processEvents()

    def test_camera_text_is_drawn_inside_single_view(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt5 import QtWidgets
        from ui.window import PersonnelCountWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PersonnelCountWindow(camera_names={"top": "井上", "bottom": "井底"})
        window.resize(1280, 720)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        window.set_camera_status("top", "等待人員通過")
        window.set_frame("top", frame)
        self.assertFalse(window.top_view.pixmap().isNull())
        self.assertIsNone(getattr(window, "top_title", None))
        self.assertIsNone(getattr(window, "top_status", None))
        window.close()
        app.processEvents()

    def test_text_is_centered_in_visible_black_bars(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt5 import QtWidgets
        from ui.window import PersonnelCountWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PersonnelCountWindow(camera_names={"top": "井上", "bottom": "井底"})
        window.resize(1920, 1080)
        window._update_font_sizes()
        window.top_view.setFixedSize(640, 720)

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = (40, 80, 120)
        window.set_camera_status("top", "等待人員通過")
        window.set_frame("top", frame)

        pixmap = window.top_view.pixmap()
        image = pixmap.toImage().convertToFormat(4)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        pixels = np.frombuffer(ptr, np.uint8).reshape((image.height(), image.width(), 4))
        white = (pixels[:, :, 0] > 180) & (pixels[:, :, 1] > 180) & (pixels[:, :, 2] > 180)

        top_black_h = 180
        bottom_black_y = 540
        top_ys = np.where(white[:top_black_h, :])[0]
        bottom_ys = np.where(white[bottom_black_y:, :])[0]
        self.assertTrue(top_ys.size)
        self.assertTrue(bottom_ys.size)
        top_text_center = (top_ys.min() + top_ys.max() + 1) / 2
        bottom_text_center = bottom_black_y + (bottom_ys.min() + bottom_ys.max() + 1) / 2
        self.assertLess(abs(top_text_center - top_black_h / 2), 3)
        self.assertLess(abs(bottom_text_center - (bottom_black_y + (720 - bottom_black_y) / 2)), 3)
        window.close()
        app.processEvents()

    def test_centered_text_baseline_stays_inside_bar(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt5 import QtGui, QtWidgets
        from ui.window import PersonnelCountWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PersonnelCountWindow()
        window.resize(1920, 1080)
        window._update_font_sizes()
        font = QtGui.QFont()
        font.setBold(True)
        font.setPixelSize(window.bar_font)
        metrics = QtGui.QFontMetrics(font)
        baseline = (window.bar_height - metrics.height()) // 2 + metrics.ascent()
        self.assertGreater(baseline - metrics.ascent(), 0)
        self.assertLess(baseline + metrics.descent(), window.bar_height)
        window.close()
        app.processEvents()

    def test_status_text_is_user_facing(self):
        from app.main import user_status_text

        self.assertEqual(user_status_text("waiting"), "等待人員通過")
        self.assertEqual(user_status_text("middle_only"), "人員在通道中")
        self.assertEqual(user_status_text("paused_multi_person"), "多人同框，暫停計數")

    def test_zone_color_follows_label_not_position(self):
        from app.main import CameraWorker

        worker = CameraWorker.__new__(CameraWorker)
        self.assertEqual(worker._zone_color("A"), (255, 120, 0))
        self.assertEqual(worker._zone_color("B"), (0, 220, 255))


class DetectorResizeTest(unittest.TestCase):
    def test_prepare_frame_scales_1920_to_960(self):
        from detection.person import PersonDetector

        with patch("detection.person.YOLOv10"):
            detector = PersonDetector({
                "model": {
                    "path": "models/int8/best_cloth2_openvino_model",
                    "person_class_id": 1,
                    "min_conf": 0.35,
                    "iou": 0.45,
                    "inference_width": 960,
                }
            })
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized, scale_x, scale_y = detector._prepare_frame(frame)
        self.assertEqual(resized.shape[:2], (540, 960))
        self.assertEqual(scale_x, 2.0)
        self.assertEqual(scale_y, 2.0)

    def test_prepare_frame_keeps_small_frame(self):
        from detection.person import PersonDetector

        with patch("detection.person.YOLOv10"):
            detector = PersonDetector({
                "model": {
                    "path": "models/int8/best_cloth2_openvino_model",
                    "person_class_id": 1,
                    "min_conf": 0.35,
                    "iou": 0.45,
                    "inference_width": 960,
                }
            })
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        same, scale_x, scale_y = detector._prepare_frame(frame)
        self.assertIs(same, frame)
        self.assertEqual(scale_x, 1.0)
        self.assertEqual(scale_y, 1.0)


if __name__ == "__main__":
    unittest.main()
