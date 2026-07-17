from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
from ultralytics import YOLOv10

from config.loader import project_path


@dataclass(frozen=True)
class Detection:
    box: tuple
    conf: float
    cls: int

    @property
    def center_x(self):
        x1, _, x2, _ = self.box
        return (x1 + x2) / 2

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


class PersonDetector:
    def __init__(self, config):
        model_cfg = config["model"]
        model_path = Path(model_cfg["path"])
        if not model_path.is_absolute():
            model_path = project_path(model_path)
        self.model = YOLOv10(model_path, task="detect")
        self.person_class_id = int(model_cfg.get("person_class_id", 1))
        self.min_conf = float(model_cfg.get("min_conf", 0.35))
        self.iou = float(model_cfg.get("iou", 0.45))
        self.duplicate_iou = float(model_cfg.get("duplicate_iou", 0.75))
        self.inference_width = int(model_cfg.get("inference_width", 960) or 0)
        self.lock = threading.Lock()

    def detect(self, frame):
        input_frame, scale_x, scale_y = self._prepare_frame(frame)
        with self.lock:
            result = self.model(source=input_frame, conf=self.min_conf, iou=self.iou, verbose=False)[0]
        detections = []
        for det in result.boxes:
            cls_id = int(det.cls)
            if cls_id != self.person_class_id:
                continue
            conf = float(det.conf[0])
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
            box = (
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            )
            detections.append(Detection(box, conf, cls_id))
        return self._remove_duplicate_people(detections)

    def _remove_duplicate_people(self, detections):
        kept = []
        for det in sorted(detections, key=lambda item: item.conf, reverse=True):
            if all(_box_iou(det.box, other.box) <= self.duplicate_iou for other in kept):
                kept.append(det)
        return kept

    def _prepare_frame(self, frame):
        if self.inference_width <= 0:
            return frame, 1.0, 1.0
        height, width = frame.shape[:2]
        if width <= self.inference_width:
            return frame, 1.0, 1.0
        new_width = self.inference_width
        new_height = max(1, int(round(height * (new_width / width))))
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, width / new_width, height / new_height


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(1, area_a + area_b - intersection)
