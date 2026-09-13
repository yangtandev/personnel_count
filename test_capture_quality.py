import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camera.capture import FrameQualityGate, VideoCapture, frame_quality_issue


def textured_frame(seed=1):
    rng = np.random.default_rng(seed)
    frame = np.clip(rng.normal(110, 20, (240, 320, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (25, 25), (295, 215), (70, 105, 135), 3)
    return frame


class FrameQualityTest(unittest.TestCase):
    def test_accepts_normal_frame(self):
        self.assertIsNone(frame_quality_issue(textured_frame()))

    def test_rejects_local_green_block(self):
        frame = textured_frame()
        frame[60:130, 25:295] = (0, 220, 0)
        self.assertEqual(frame_quality_issue(frame), "green_block")

    def test_rejects_local_noise_block(self):
        frame = textured_frame()
        rng = np.random.default_rng(2)
        frame[60:130, 25:295] = rng.integers(0, 256, (70, 270, 3), dtype=np.uint8)
        self.assertEqual(frame_quality_issue(frame), "noise_block")

    def test_rejects_sudden_blur_after_normal_frames(self):
        gate = FrameQualityGate()
        frame = textured_frame()
        self.assertIsNone(gate.check(frame))
        self.assertIsNone(gate.check(frame))
        blurred = cv2.GaussianBlur(frame, (31, 31), 0)
        self.assertIsNone(gate.check(blurred))
        self.assertEqual(gate.check(blurred), "blurred_frame")

    def test_bad_frame_is_not_published(self):
        capture = object.__new__(VideoCapture)
        capture.rtsp_url = "rtsp://user:pass@example.test/live"
        capture.latest_frame = None
        capture.latest_frame_ready = threading.Condition()
        capture.quality_gate = FrameQualityGate()
        capture._last_bad_frame_log_at = 0.0
        capture._consecutive_bad_frames = 0
        capture.proc = None
        capture.config = {}

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :] = (0, 220, 0)

        self.assertFalse(capture._publish_latest_frame(frame))
        self.assertIsNone(capture.latest_frame)

    def test_restarts_when_media_clock_falls_behind(self):
        capture = object.__new__(VideoCapture)
        proc = object()
        capture.proc = proc
        capture._progress_lock = threading.Lock()
        capture._progress_anchor = None
        capture._restart_requested_for = None
        capture.stop_threads = False
        capture.config = {"camera_max_lag_sec": 2}

        with patch("camera.capture.time.monotonic", side_effect=(10.0, 14.5)):
            capture._observe_progress(proc, 1_000_000, "test")
            with patch.object(capture, "_request_restart") as restart:
                capture._observe_progress(proc, 2_000_000, "test")
        restart.assert_called_once()


if __name__ == "__main__":
    unittest.main()
