import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.loader import load_config, project_path
from counting.lines import LineCounter
from detection.person import PersonDetector


DATASETS = {
    "top_0831-0903": ("top", "top_0831-0903.mp4"),
    "top_0910": ("top", "top_0910.mp4"),
    "bottom_0910": ("bottom", "bottom_0910.mp4"),
}

EXPECTED_RUNS = {
    "top_0831-0903": [("enter", 3), ("exit", 3), ("enter", 3), ("exit", 3)],
    "top_0910": [("enter", 9), ("exit", 9), ("enter", 3), ("exit", 3)],
    "bottom_0910": [("exit", 10), ("enter", 12)],
}


def event_runs(events):
    runs = []
    for event in events:
        name = event["event"]
        if runs and runs[-1][0] == name:
            runs[-1] = (name, runs[-1][1] + 1)
        else:
            runs.append((name, 1))
    return runs


def validate_events(dataset, events):
    actual = event_runs(events)
    expected = EXPECTED_RUNS[dataset]
    if actual != expected:
        raise AssertionError(f"{dataset}: event runs {actual}, expected {expected}")


def annotate(frame, counter, detections):
    mode, geometry = counter.counting_geometry(frame.shape[1], frame.shape[0])
    if mode == "corridor" and geometry is not None:
        colors = {
            "outside": (255, 120, 0),
            "transit": (0, 220, 255),
            "inside": (0, 200, 0),
        }
        for name, polygon in geometry.items():
            points = np.array([[round(x), round(y)] for x, y in polygon], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], colors[name])
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.polylines(frame, [points], True, colors[name], 3)
            x, y = points[0]
            cv2.putText(frame, name.upper(), (x, max(30, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)
    elif geometry is not None:
        line = geometry
        start = tuple(round(value) for value in line[0])
        end = tuple(round(value) for value in line[1])
        cv2.line(frame, start, end, (0, 220, 255), 4)
        cv2.circle(frame, start, 7, (0, 220, 255), -1)
        cv2.circle(frame, end, 7, (0, 220, 255), -1)

    for track in counter.track_visuals():
        trail = np.array([[round(x), round(y)] for x, y in track["trail"]], dtype=np.int32)
        if len(trail) >= 2:
            cv2.polylines(frame, [trail], False, (255, 255, 0), 3)
        if len(trail):
            x, y = trail[-1]
            cv2.putText(
                frame,
                f'uid {track["stable_id"]} / id {track["track_id"]} {track["point_source"]} side {track["side"]} {track["state"]}',
                (x + 8, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        tracked = detection.track_id is not None
        color = (0, 200, 0) if tracked else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"id {detection.track_id} person {detection.conf:.2f}" if tracked else "NO TRACK ID"
        cv2.putText(frame, label, (x1, max(66, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        point = counter.detection_point(detection)
        cv2.circle(frame, tuple(round(value) for value in point), 6, (0, 0, 255), -1)

    return frame


def run_dataset(name, camera_name, source_name, config, output_fps, start_time=0.0, end_time=None, output_name=None):
    source = project_path(source_name)
    if not source.exists():
        raise FileNotFoundError(source)
    print(json.dumps({"dataset": name, "stage": "initializing counter"}), flush=True)
    counter = LineCounter(camera_name, config)
    _, geometry = counter.counting_geometry(1920, 1080)
    if geometry is None:
        raise RuntimeError(f"counting geometry not configured: {camera_name}")

    detector_config = copy.deepcopy(config)
    detector_config["model"]["use_face_detection"] = False
    print(json.dumps({"dataset": name, "stage": "loading detector"}), flush=True)
    detector = PersonDetector(detector_config)
    print(json.dumps({"dataset": name, "stage": "opening video"}), flush=True)
    cap = cv2.VideoCapture(str(source))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, round(source_fps / output_fps))
    output_size = (1280, 720)
    ui_output_size = (1280, 820)
    temp_output = project_path(f".{name}_regression_mp4v.tmp.mp4")
    ui_temp_output = project_path(f".{name}_regression_ui_mp4v.tmp.mp4")
    output_stem = output_name or f"{name}_regression"
    final_output = project_path(f"{output_stem}_h264.mp4")
    writer = cv2.VideoWriter(
        str(temp_output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output: {temp_output}")

    count = int(config["counter"].get("initial_count", 0))
    all_events = []
    frame_states = []
    next_progress = 0.0
    try:
        while True:
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % stride:
                continue
            source_time = frame_no / source_fps
            if source_time < start_time:
                continue
            if end_time is not None and source_time > end_time:
                break
            if source_time >= next_progress:
                print(
                    json.dumps({"dataset": name, "time": round(source_time, 1), "count": count}),
                    flush=True,
                )
                next_progress += 30.0
            detections = detector.detect(frame)
            events, status, people = counter.update(detections, frame.shape, source_time, count)
            if events:
                count = events[-1].count_after
                all_events.extend(
                    {
                        "time": round(source_time, 3),
                        "track_id": event.track_id,
                        "stable_id": event.stable_id,
                        "direction": event.direction,
                        "event": event.event,
                        "count": event.count_after,
                        "status": event.status,
                    }
                    for event in events
                )
            annotated = annotate(frame.copy(), counter, people)
            writer.write(cv2.resize(annotated, output_size, interpolation=cv2.INTER_AREA))
            frame_states.append((count, status))
    finally:
        cap.release()
        writer.release()
        detector.close()

    if detector.tracking_failed:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError("tracking unavailable; install lapx before running regression")

    from ui.regression_renderer import RegressionWindowRenderer

    renderer = RegressionWindowRenderer(config, camera_name, ui_output_size)
    ui_writer = cv2.VideoWriter(
        str(ui_temp_output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, ui_output_size
    )
    ui_cap = cv2.VideoCapture(str(temp_output))
    if not ui_writer.isOpened() or not ui_cap.isOpened():
        ui_cap.release()
        ui_writer.release()
        renderer.close()
        raise RuntimeError(f"cannot render UI regression output: {name}")
    try:
        for count, status in frame_states:
            ok, frame = ui_cap.read()
            if not ok:
                raise RuntimeError(f"annotated regression frame missing: {name}")
            ui_writer.write(renderer.render(frame, count, status))
    finally:
        ui_cap.release()
        ui_writer.release()
        renderer.close()

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(ui_temp_output),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(final_output),
        ],
        check=True,
    )
    temp_output.unlink()
    ui_temp_output.unlink()
    report = {
        "dataset": name,
        "camera": camera_name,
        "source": source.name,
        "output": final_output.name,
        "final_count": count,
        "events": all_events,
    }
    project_path(f"{output_stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if start_time <= 0 and end_time is None:
        validate_events(name, all_events)
    print(json.dumps(report, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--start", type=float, default=0.0, help="source start time in seconds")
    parser.add_argument("--end", type=float, help="source end time in seconds")
    parser.add_argument("--output-name", help="output stem, without extension")
    parser.add_argument("--corridor", help="JSON corridor geometry, used for the selected camera")
    args = parser.parse_args()
    unknown_datasets = sorted(set(args.datasets) - DATASETS.keys())
    if unknown_datasets:
        parser.error(f"unknown dataset(s): {', '.join(unknown_datasets)}")

    config = load_config(args.config)
    reports = []
    if args.output_name and len(args.datasets) != 1:
        parser.error("--output-name requires exactly one dataset")
    if args.corridor and len(args.datasets) != 1:
        parser.error("--corridor requires exactly one dataset")
    if args.corridor:
        camera_name, _ = DATASETS[args.datasets[0]]
        config.setdefault("crossing", {})["counting_mode"] = "corridor"
        config["crossing"].setdefault("corridors", {})[camera_name] = json.loads(
            Path(args.corridor).read_text(encoding="utf-8")
        )
    for name in args.datasets or DATASETS:
        camera_name, source_name = DATASETS[name]
        reports.append(
            run_dataset(
                name,
                camera_name,
                source_name,
                config,
                args.fps,
                args.start,
                args.end,
                args.output_name,
            )
        )
    print(json.dumps({"datasets": len(reports), "events": sum(len(item["events"]) for item in reports)}))


if __name__ == "__main__":
    main()
