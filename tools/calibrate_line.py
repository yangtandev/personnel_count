import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from camera.capture import VideoCapture
from config.loader import DEFAULT_CONFIG_PATH, load_config
from counting.calibration import suggest_counting_line


def parse_args():
    parser = argparse.ArgumentParser(description="Capture one frame and draw a counting line.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--camera", default="top", choices=("top", "bottom"))
    parser.add_argument("--output", help="default: overwrite --config")
    parser.add_argument("--image", help="use an image instead of grabbing the camera")
    parser.add_argument("--auto-video", help="suggest line geometry from tracked movement in a video")
    parser.add_argument("--auto-seconds", type=float, default=60.0)
    parser.add_argument("--save-auto", action="store_true", help="save suggestion without GUI confirmation")
    parser.add_argument("--timeout", type=float, default=20.0)
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


def denormalized(points, width, height):
    result = []
    for x, y in points or []:
        if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
            x *= width
            y *= height
        result.append((round(x), round(y)))
    return result


def draw_preview(frame, points):
    preview = frame.copy()
    for point in points:
        cv2.circle(preview, point, 7, (0, 220, 255), -1)
    if len(points) == 2:
        start, end = points
        cv2.arrowedLine(preview, start, end, (0, 220, 255), 4, tipLength=0.04)
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        offset = (-dy / length * 45, dx / length * 45)
        plus = (round(midpoint[0] + offset[0]), round(midpoint[1] + offset[1]))
        minus = (round(midpoint[0] - offset[0]), round(midpoint[1] - offset[1]))
        cv2.putText(preview, "+", plus, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(preview, "-", minus, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(
        preview,
        "Left-click start/end | U/right-click undo | R reset | S save | Q quit",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )
    return preview


def edit_line(frame, points):
    window = "calibrate_line"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) == 2:
                points.clear()
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
            if len(points) != 2:
                print("Need exactly 2 points")
                continue
            cv2.destroyWindow(window)
            return points


def analyze_video(config, video_path, seconds):
    from detection.person import PersonDetector

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    detector_config = json.loads(json.dumps(config))
    detector_config["model"]["use_face_detection"] = False
    detector = PersonDetector(detector_config)
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    tracks = {}
    preview = None
    frame_index = 0
    try:
        while frame_index / fps < seconds:
            ok, frame = capture.read()
            if not ok:
                break
            preview = frame
            for detection in detector.detect(frame):
                if detection.track_id is None:
                    continue
                if detection.point is not None:
                    point = detection.point
                else:
                    x1, y1, x2, y2 = detection.box
                    point = ((x1 + x2) / 2, y1 + (y2 - y1) * 0.15)
                tracks.setdefault(detection.track_id, []).append(point)
            frame_index += 1
    finally:
        capture.release()
        detector.close()
    if preview is None:
        raise ValueError("video has no decodable frames")
    height, width = preview.shape[:2]
    points = suggest_counting_line(list(tracks.values()), width, height)
    return preview, points


def main():
    args = parse_args()
    config = load_config(args.config)
    points = None
    if args.auto_video:
        frame, points = analyze_video(config, args.auto_video, args.auto_seconds)
    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"cannot read image: {args.image}")
    else:
        frame = grab_frame(config, args.camera, args.timeout)

    height, width = frame.shape[:2]
    existing = config.get("crossing", {}).get("lines", {}).get(args.camera, [])
    if points is None:
        points = denormalized(existing, width, height)
    if not (args.auto_video and args.save_auto):
        points = edit_line(frame, points)
    if points is None:
        print("Canceled. Config unchanged.")
        return 1

    crossing = config.setdefault("crossing", {})
    lines = crossing.setdefault("lines", {})
    lines[args.camera] = [[round(x / width, 4), round(y / height, 4)] for x, y in points]
    path = Path(args.output or args.config)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)
        file.write("\n")
    print(f"Saved {args.camera} counting line to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
