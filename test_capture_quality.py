from pathlib import Path
import threading
import unittest

import cv2
import numpy as np

from camera.capture import VideoCapture, frame_quality_issue


class FrameQualityTest(unittest.TestCase):
    def test_rejects_green_screen(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :] = (0, 220, 0)

        self.assertIn("green_screen", frame_quality_issue(frame))

    def test_rejects_horizontal_color_artifacts(self):
        frame = np.full((240, 320, 3), 110, dtype=np.uint8)
        frame[80:140, :, :] = (255, 0, 255)
        frame[140:180, ::2, :] = (0, 255, 255)
        frame[140:180, 1::2, :] = (255, 255, 0)

        self.assertIn("decode_artifacts", frame_quality_issue(frame))

    def test_accepts_plain_cctv_like_frame(self):
        frame = np.full((240, 320, 3), (120, 120, 120), dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (300, 220), (80, 100, 120), -1)
        cv2.line(frame, (0, 120), (319, 120), (40, 40, 40), 3)

        self.assertIsNone(frame_quality_issue(frame))

    def test_rejects_sample_hevc_bad_frames_when_present(self):
        root = Path(__file__).resolve().parent
        files = [
            ("hevc_JENYI_220133246219301.jpg", "decode_artifacts"),
            ("hevc_KTSM_602501991339101.jpg", "green_screen"),
        ]
        for filename, expected_issue in files:
            path = root / filename
            if not path.exists():
                continue
            with self.subTest(filename=filename):
                frame = cv2.imread(str(path))
                self.assertIsNotNone(frame)
                self.assertIn(expected_issue, frame_quality_issue(frame))

    def test_bad_frame_is_not_published_and_exposes_issue(self):
        capture = object.__new__(VideoCapture)
        capture.rtsp_url = "rtsp://user:pass@example.test/live"
        capture.latest_frame = None
        capture.latest_frame_ready = threading.Condition()
        capture._last_bad_frame_log_at = 0.0

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :] = (0, 220, 0)

        self.assertFalse(capture._publish_latest_frame(frame))
        self.assertIsNone(capture.latest_frame)


if __name__ == "__main__":
    unittest.main()
