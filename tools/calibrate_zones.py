import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from camera.capture import VideoCapture
from config.loader import DEFAULT_CONFIG_PATH, load_config


COLORS = {
    "A": (255, 120, 0),
    "B": (0, 220, 255),
}
LABELS = ("A", "B")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture one frame and draw A/B counting zones.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="config path")
    parser.add_argument("--camera", default="top", choices=("top", "bottom"), help="camera name")
    parser.add_argument("--output", help="output config path. default: overwrite --config")
    parser.add_argument("--image", help="use an image file instead of grabbing camera frame")
    parser.add_argument("--timeout", type=float, default=20.0, help="camera frame timeout seconds")
    return parser.parse_args()


def grab_frame(config, camera_name, timeout):
    capture = VideoCapture(config["camera"][camera_name], config_data={**config.get("camera", {})})
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            frame = capture.read()
            if frame is not None:
                return frame
    finally:
        capture.terminate()
    raise TimeoutError(f"no frame from {camera_name} within {timeout:g}s")


def normalized(points, width, height):
    return [[round(x / width, 4), round(y / height, 4)] for x, y in points]


def denormalized(points, width, height):
    denorm = []
    for x, y in points or []:
        if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
            x *= width
            y *= height
        denorm.append((int(round(x)), int(round(y))))
    return denorm


def draw_preview(frame, zone_points, active_label):
    preview = frame.copy()
    overlay = preview.copy()
    for label in LABELS:
        points = zone_points[label]
        color = COLORS[label]
        if len(points) >= 3:
            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(preview, [polygon], True, color, 3)
        for index, point in enumerate(points):
            cv2.circle(preview, point, 6 if label == active_label else 4, color, -1)
            if index:
                cv2.line(preview, points[index - 1], point, color, 2)
        if points:
            cv2.putText(preview, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.addWeighted(overlay, 0.18, preview, 0.82, 0, preview)
    cv2.putText(
        preview,
        f"Drawing {active_label} | A/B switch | left-click add | right-click or U undo | R reset | S save | Q quit",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return preview


def existing_zone_points(config, camera_name, frame_shape):
    height, width = frame_shape[:2]
    regions = config.get("zones", {}).get("regions", {})
    camera_regions = regions.get(camera_name, regions) if isinstance(regions, dict) else {}
    if not isinstance(camera_regions, dict):
        camera_regions = {}
    return {label: denormalized(camera_regions.get(label), width, height) for label in LABELS}


def edit_zones(frame, zone_points):
    active_label = "A"
    window = "calibrate_zones"

    def on_mouse(event, x, y, *_):
        points = zone_points[active_label]
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        cv2.imshow(window, draw_preview(frame, zone_points, active_label))
        key = cv2.waitKey(30) & 0xFF
        points = zone_points[active_label]
        if key in (ord("a"), ord("A")):
            active_label = "A"
        elif key in (ord("b"), ord("B")):
            active_label = "B"
        elif key in (ord("u"), ord("U"), 8) and points:
            points.pop()
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window)
            return None
        elif key in (ord("s"), ord("S")):
            missing = [label for label in LABELS if len(zone_points[label]) < 3]
            if missing:
                print(f"Need at least 3 points for {', '.join(missing)}")
                continue
            cv2.destroyWindow(window)
            return zone_points


def save_zones(config, config_path, output_path, camera_name, zone_points, frame_shape):
    height, width = frame_shape[:2]
    zone_config = config.setdefault("zones", {})
    for key in ("left_width_ratio", "right_width_ratio", "zone_width_ratio", "left_ratio", "right_ratio", "mode", "labels"):
        zone_config.pop(key, None)
    config.get("counter", {}).pop("require_middle", None)

    regions = zone_config.setdefault("regions", {})
    regions[camera_name] = {label: normalized(zone_points[label], width, height) for label in LABELS}
    zone_config.setdefault("zone_point_y_ratio", 0.35)

    path = Path(output_path or config_path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=4)
        fh.write("\n")
    return path


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"cannot read image: {args.image}")
    else:
        frame = grab_frame(config, args.camera, args.timeout)

    zone_points = edit_zones(frame, existing_zone_points(config, args.camera, frame.shape))
    if zone_points is None:
        print("Canceled. Config unchanged.")
        return 1

    path = save_zones(config, args.config, args.output, args.camera, zone_points, frame.shape)
    print(f"Saved {args.camera} A/B zones to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
