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


def annotate(frame, counter, detections, status, count, source_name, source_time, event_text):
    line = counter.counting_line(frame.shape[1], frame.shape[0])
    if line is not None:
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

    lines = [
        f"COUNT {count}   STATUS {status}",
        f"SOURCE {source_name} {source_time:06.2f}s   LOCK {'ON' if counter.flow_direction_lock else 'OFF'}",
        "+ to - / - to +: see camera direction mapping",
    ]
    if event_text:
        lines.append(f"EVENT {event_text}")
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (min(frame.shape[1] - 8, 900), 20 + len(lines) * 32), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for index, text in enumerate(lines):
        color = (0, 255, 255) if text.startswith("EVENT") else (255, 255, 255)
        cv2.putText(frame, text, (20, 38 + index * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    return frame


def run_dataset(name, camera_name, source_name, config, output_fps):
    source = project_path(source_name)
    if not source.exists():
        raise FileNotFoundError(source)
    counter = LineCounter(camera_name, config)
    if counter.counting_line(1920, 1080) is None:
        raise RuntimeError(f"counting line not configured: {camera_name}")

    detector_config = copy.deepcopy(config)
    detector_config["model"]["use_face_detection"] = False
    detector = PersonDetector(detector_config)
    cap = cv2.VideoCapture(str(source))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, round(source_fps / output_fps))
    output_size = (1280, 720)
    temp_output = project_path(f".{name}_regression_mp4v.tmp.mp4")
    final_output = project_path(f"{name}_regression_h264.mp4")
    writer = cv2.VideoWriter(
        str(temp_output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output: {temp_output}")

    count = int(config["counter"].get("initial_count", 0))
    all_events = []
    banner = ""
    banner_until = -1.0
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
                banner = " + ".join(event.event.upper() for event in events)
                banner_until = source_time + 1.0
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
            annotated = annotate(
                frame.copy(),
                counter,
                people,
                status,
                count,
                source.name,
                source_time,
                banner if source_time <= banner_until else "",
            )
            writer.write(cv2.resize(annotated, output_size, interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
        writer.release()
        detector.close()

    if detector.tracking_failed:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError("tracking unavailable; install lapx before running regression")

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(temp_output),
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
    report = {
        "dataset": name,
        "camera": camera_name,
        "source": source.name,
        "output": final_output.name,
        "final_count": count,
        "events": all_events,
    }
    project_path(f"{name}_regression.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validate_events(name, all_events)
    print(json.dumps(report, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=DATASETS)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    config = load_config(args.config)
    reports = []
    for name in args.datasets or DATASETS:
        camera_name, source_name = DATASETS[name]
        reports.append(run_dataset(name, camera_name, source_name, config, args.fps))
    print(json.dumps({"datasets": len(reports), "events": sum(len(item["events"]) for item in reports)}))


if __name__ == "__main__":
    main()
