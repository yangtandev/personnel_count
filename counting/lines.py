from dataclasses import dataclass, field
from math import hypot


@dataclass(frozen=True)
class CountEvent:
    camera: str
    direction: str
    event: str
    count_before: int
    count_after: int
    confidence: float
    status: str


@dataclass
class _Track:
    side: int
    anchor_point: tuple
    point: tuple
    last_seen_at: float
    seen_frames: int = 1
    cooldown_until: float = 0.0
    counted: bool = False
    trail: list = field(default_factory=list)


class LineCounter:
    def __init__(self, camera_name, config):
        self.camera_name = camera_name
        counter_cfg = config["counter"]
        crossing_cfg = config.get("crossing", {})
        self.line_points = crossing_cfg.get("lines", {}).get(camera_name)
        self.direction_map = config["direction"][camera_name]
        self.min_area_ratio = float(counter_cfg.get("min_person_area_ratio", 0.02))
        self.lost_timeout_sec = float(counter_cfg.get("lost_timeout_sec", 2.0))
        self.cooldown_sec = float(counter_cfg.get("event_cooldown_sec", 1.5))
        self.one_event_per_track = bool(counter_cfg.get("one_event_per_track", False))
        self.flow_direction_lock = bool(counter_cfg.get("flow_direction_lock", True))
        self.hysteresis_ratio = float(crossing_cfg.get("hysteresis_ratio", 0.012))
        self.segment_margin_ratio = float(crossing_cfg.get("segment_margin_ratio", 0.05))
        self.min_track_frames = int(crossing_cfg.get("min_track_frames", 3))
        self.trail_length = int(crossing_cfg.get("trail_length", 30))
        ratio_cfg = crossing_cfg.get("point_y_ratio", 0.15)
        if isinstance(ratio_cfg, dict):
            ratio_cfg = ratio_cfg.get(camera_name, ratio_cfg.get("default", 0.15))
        self.point_y_ratio = float(ratio_cfg)
        self.tracks = {}
        self.flow_direction = None
        self.status = "waiting"

    def reset(self, status="waiting"):
        self.tracks.clear()
        self.flow_direction = None
        self.status = status

    def counting_line(self, width, height):
        if not isinstance(self.line_points, (list, tuple)) or len(self.line_points) != 2:
            return None
        points = []
        for point in self.line_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None
            x, y = float(point[0]), float(point[1])
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                x *= width
                y *= height
            points.append((x, y))
        if points[0] == points[1]:
            return None
        return tuple(points)

    def detection_point(self, detection):
        x1, y1, x2, y2 = detection.box
        return (
            (x1 + x2) / 2,
            y1 + max(0, y2 - y1) * self.point_y_ratio,
        )

    def update(self, detections, frame_shape, now, current_count):
        height, width = frame_shape[:2]
        line = self.counting_line(width, height)
        min_area = width * height * self.min_area_ratio
        people = [det for det in detections if det.area >= min_area]

        self._drop_lost_tracks(now)
        if not self.tracks:
            self.flow_direction = None
        if line is None:
            self.status = "line_not_configured"
            return [], self.status, people
        if not people:
            self.status = "tracking" if self.tracks else "waiting"
            return [], self.status, people

        events = []
        count = current_count
        has_untracked_person = False
        for person in people:
            external_id = getattr(person, "track_id", None)
            if external_id is None:
                has_untracked_person = True
                continue

            track_id = f"det:{external_id}"
            point = self.detection_point(person)
            side = self._side(point, line, min(width, height) * self.hysteresis_ratio)
            track = self.tracks.get(track_id)
            if track is None:
                self.tracks[track_id] = _Track(
                    side=side,
                    anchor_point=point,
                    point=point,
                    last_seen_at=now,
                    trail=[point],
                )
                self.status = "near_line" if side == 0 else "tracking"
                continue

            track.last_seen_at = now
            track.seen_frames += 1
            track.point = point
            track.trail.append(point)
            if len(track.trail) > self.trail_length:
                del track.trail[:-self.trail_length]

            if side == 0:
                self.status = "near_line"
                continue
            if track.side == 0:
                track.side = side
                track.anchor_point = point
                self.status = "tracking"
                continue
            if side == track.side:
                track.anchor_point = point
                self.status = "tracking"
                continue

            direction = "positive_to_negative" if track.side > side else "negative_to_positive"
            if track.seen_frames < self.min_track_frames or not self._crosses_segment(
                track.anchor_point, point, line
            ):
                track.side = side
                track.anchor_point = point
                self.status = "crossing_rejected"
                continue
            if self.one_event_per_track and track.counted:
                self.status = "counted_track"
                continue
            if now < track.cooldown_until:
                self.status = "cooldown"
                continue

            mapped_event = self.direction_map.get(direction)
            if mapped_event not in {"enter", "exit"}:
                track.side = side
                track.anchor_point = point
                self.status = "unknown_direction"
                continue
            if self.flow_direction_lock and self.flow_direction and direction != self.flow_direction:
                track.side = side
                track.anchor_point = point
                self.status = "flow_reverse_ignored"
                continue

            before = count
            after = before + (1 if mapped_event == "enter" else -1)
            status = "ok"
            if after < 0:
                after = 0
                status = "blocked_negative_count"
            events.append(
                CountEvent(
                    camera=self.camera_name,
                    direction=direction,
                    event=mapped_event,
                    count_before=before,
                    count_after=after,
                    confidence=person.conf,
                    status=status,
                )
            )
            count = after
            track.side = side
            track.anchor_point = point
            track.cooldown_until = now + self.cooldown_sec
            track.counted = True
            self.flow_direction = direction
            self.status = f"counted_{mapped_event}"

        if events:
            self.status = f"counted_{events[-1].event}"
        elif has_untracked_person:
            self.status = "tracking_unavailable"
        return events, self.status, people

    def track_visuals(self):
        visuals = []
        for track_id, track in self.tracks.items():
            visuals.append(
                {
                    "track_id": track_id.removeprefix("det:"),
                    "side": {1: "+", -1: "-", 0: "line"}[track.side],
                    "trail": tuple(track.trail),
                }
            )
        return visuals

    def _drop_lost_tracks(self, now):
        lost = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track.last_seen_at > self.lost_timeout_sec
        ]
        for track_id in lost:
            del self.tracks[track_id]

    @staticmethod
    def _side(point, line, margin):
        (x1, y1), (x2, y2) = line
        distance = ((x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)) / hypot(
            x2 - x1, y2 - y1
        )
        if distance > margin:
            return 1
        if distance < -margin:
            return -1
        return 0

    def _crosses_segment(self, previous, current, line):
        (x1, y1), (x2, y2) = line
        previous_cross = (x2 - x1) * (previous[1] - y1) - (y2 - y1) * (previous[0] - x1)
        current_cross = (x2 - x1) * (current[1] - y1) - (y2 - y1) * (current[0] - x1)
        denominator = previous_cross - current_cross
        if denominator == 0:
            return False
        fraction = previous_cross / denominator
        cross_x = previous[0] + (current[0] - previous[0]) * fraction
        cross_y = previous[1] + (current[1] - previous[1]) * fraction
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        projection = ((cross_x - x1) * (x2 - x1) + (cross_y - y1) * (y2 - y1)) / line_length_sq
        return -self.segment_margin_ratio <= projection <= 1 + self.segment_margin_ratio
