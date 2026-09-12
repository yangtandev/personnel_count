from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
try:
    from ultralytics import YOLOv10
except ImportError:
    from ultralytics import YOLO as YOLOv10

try:
    import mediapipe as mp
except ImportError:
    mp = None

from config.loader import project_path
from detection.head_assignment import match_heads_to_people


@dataclass(frozen=True)
class Detection:
    box: tuple
    conf: float
    cls: int
    track_id: int = None
    point: tuple = None
    point_source: str = "person"
    head_box: tuple = None

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
        self.track_conf = float(model_cfg.get("track_conf", 0.25))
        self.iou = float(model_cfg.get("iou", 0.45))
        self.duplicate_iou = float(model_cfg.get("duplicate_iou", 0.75))
        self.inference_width = int(model_cfg.get("inference_width", 960) or 0)
        self.use_tracking = bool(model_cfg.get("use_tracking", True))
        self.tracker = model_cfg.get("tracker", "botsort.yaml")
        self.head_class_ids = {int(cls) for cls in model_cfg.get("head_class_ids", [2])}
        self.detect_class_ids = sorted({self.person_class_id, *self.head_class_ids})
        self.use_face_detection = bool(model_cfg.get("use_face_detection", True))
        self.lock = threading.Lock()
        self.tracking_failed = False
        self.face_detector = None
        self.face_mesh = None
        if mp is not None and self.use_face_detection:
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=float(model_cfg.get("face_min_conf", 0.35)),
            )
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=int(model_cfg.get("max_faces", 5)),
                refine_landmarks=True,
                min_detection_confidence=float(model_cfg.get("face_min_conf", 0.35)),
                min_tracking_confidence=0.5,
            )

    def detect(self, frame):
        input_frame, scale_x, scale_y = self._prepare_frame(frame)
        with self.lock:
            if self.use_tracking and not self.tracking_failed:
                try:
                    result = self.model.track(
                        source=input_frame,
                        persist=True,
                        classes=self.detect_class_ids,
                        conf=self.track_conf,
                        iou=self.iou,
                        tracker=self.tracker,
                        verbose=False,
                    )[0]
                except ModuleNotFoundError:
                    self.tracking_failed = True
                    result = self.model(
                        source=input_frame,
                        classes=self.detect_class_ids,
                        conf=self.min_conf,
                        iou=self.iou,
                        verbose=False,
                    )[0]
            else:
                result = self.model(
                    source=input_frame,
                    classes=self.detect_class_ids,
                    conf=self.min_conf,
                    iou=self.iou,
                    verbose=False,
                )[0]
        head_boxes = self._head_boxes(result, input_frame, scale_x, scale_y)
        detections = []
        track_ids = [] if result.boxes.id is None else result.boxes.id.cpu().numpy().astype(int).tolist()
        for index, det in enumerate(result.boxes):
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
            track_id = track_ids[index] if index < len(track_ids) else None
            detections.append(Detection(box, conf, cls_id, track_id))

        detections = self._remove_duplicate_people(detections)
        matched_heads = match_heads_to_people(head_boxes, [item.box for item in detections])
        return [
            Detection(
                item.box,
                item.conf,
                item.cls,
                item.track_id,
                _box_center(head["box"]) if head else None,
                head["source"] if head else "person",
                head["box"] if head else None,
            )
            for item, head in zip(detections, matched_heads)
        ]

    def close(self):
        if self.face_detector is not None:
            try:
                self.face_detector.close()
            except ValueError:
                pass
            self.face_detector = None
        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except ValueError:
                pass
            self.face_mesh = None

    def _head_boxes(self, result, frame, scale_x, scale_y):
        boxes = []
        for det in result.boxes:
            if int(det.cls) not in self.head_class_ids:
                continue
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
            boxes.append(
                {
                    "box": (
                        int(round(x1 * scale_x)),
                        int(round(y1 * scale_y)),
                        int(round(x2 * scale_x)),
                        int(round(y2 * scale_y)),
                    ),
                    "conf": float(det.conf[0]),
                    "source": "head",
                }
            )

        if self.face_detector is None and self.face_mesh is None:
            return boxes

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.face_detector is not None:
            face_result = self.face_detector.process(rgb)
            if face_result.detections:
                for det in face_result.detections:
                    rel = det.location_data.relative_bounding_box
                    x1 = rel.xmin * w
                    y1 = rel.ymin * h
                    x2 = x1 + rel.width * w
                    y2 = y1 + rel.height * h
                    boxes.append(
                        {
                            "box": _scale_box(_clamp_box((x1, y1, x2, y2), w, h), scale_x, scale_y),
                            "conf": float(det.score[0]),
                            "source": "face",
                        }
                    )
        if self.face_mesh is not None:
            mesh_result = self.face_mesh.process(rgb)
            if mesh_result.multi_face_landmarks:
                for landmarks in mesh_result.multi_face_landmarks:
                    xs = [point.x * w for point in landmarks.landmark]
                    ys = [point.y * h for point in landmarks.landmark]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    bw, bh = x2 - x1, y2 - y1
                    box = (x1 - bw * 0.2, y1 - bh * 0.35, x2 + bw * 0.2, y2 + bh * 0.1)
                    boxes.append(
                        {
                            "box": _scale_box(_clamp_box(box, w, h), scale_x, scale_y),
                            "conf": 1.0,
                            "source": "face",
                        }
                    )
        return boxes

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


def _box_center(box):
    x1, y1, x2, y2 = box
    return (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))


def _clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = min(width, max(x1 + 1, int(x2)))
    y2 = min(height, max(y1 + 1, int(y2)))
    return x1, y1, x2, y2


def _scale_box(box, scale_x, scale_y):
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )
