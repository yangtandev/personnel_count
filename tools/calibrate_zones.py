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
    "B": (0, 220, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Capture one frame and draw the B counting zone.")
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


def draw_preview(frame, points):
    preview = frame.copy()
    overlay = preview.copy()
    color = COLORS["B"]
    if len(points) >= 3:
        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(preview, [polygon], True, color, 3)
    for index, point in enumerate(points):
        cv2.circle(preview, point, 5, color, -1)
        if index:
            cv2.line(preview, points[index - 1], point, color, 2)
    if points:
        cv2.putText(preview, "B", points[0], cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.addWeighted(overlay, 0.18, preview, 0.82, 0, preview)
    cv2.putText(
        preview,
        "Draw B zone | left-click add | right-click or U undo | R reset | S save | Q quit",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return preview


def edit_zone(frame):
    points = []
    window = "calibrate_zones"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        cv2.imshow(window, draw_preview(frame, points))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("u"), ord("U"), 8) and points:
            points.pop()
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window)
            return None
        elif key in (ord("s"), ord("S")):
            if len(points) < 3:
                print("Need at least 3 points for B")
                continue
            cv2.destroyWindow(window)
            return points


def save_zone(config, config_path, output_path, camera_name, points, frame_shape):
    height, width = frame_shape[:2]
    zone_config = config.setdefault("zones", {})
    for key in ("left_width_ratio", "right_width_ratio", "zone_width_ratio", "left_ratio", "right_ratio", "mode", "labels"):
        zone_config.pop(key, None)
    config.get("counter", {}).pop("require_middle", None)

    regions = zone_config.setdefault("regions", {})
    regions[camera_name] = {"B": normalized(points, width, height)}
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

    points = edit_zone(frame)
    if points is None:
        print("Canceled. Config unchanged.")
        return 1

    path = save_zone(config, args.config, args.output, args.camera, points, frame.shape)
    print(f"Saved {args.camera} B zone to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
