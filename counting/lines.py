from dataclasses import dataclass, field
from math import hypot
from statistics import median


@dataclass(frozen=True)
class CountEvent:
    camera: str
    direction: str
    event: str
    count_before: int
    count_after: int
    confidence: float
    status: str
    track_id: int = None
    stable_id: int = None


@dataclass
class _Track:
    stable_id: int
    external_id: int
    side: int
    anchor_point: tuple
    point: tuple
    raw_point: tuple
    box_height: float
    point_source: str
    created_at: float
    last_seen_at: float
    seen_frames: int = 1
    cooldown_until: float = 0.0
    counted: bool = False
    last_event: str = None
    armed: bool = False
    candidate_side: int = 0
    candidate_since: float = 0.0
    candidate_frames: int = 0
    velocity: tuple = (0.0, 0.0)
    samples: list = field(default_factory=list)
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
        self.flow_direction_lock = bool(counter_cfg.get("flow_direction_lock", False))
        self.hysteresis_ratio = float(crossing_cfg.get("hysteresis_ratio", 0.012))
        self.segment_margin_ratio = float(crossing_cfg.get("segment_margin_ratio", 0.05))
        self.min_track_frames = int(crossing_cfg.get("min_track_frames", 3))
        self.trail_length = int(crossing_cfg.get("trail_length", 30))
        self.point_filter_window = max(1, int(crossing_cfg.get("point_filter_window", 5)))
        self.side_confirm_frames = max(1, int(crossing_cfg.get("side_confirm_frames", 3)))
        self.side_confirm_sec = max(0.0, float(crossing_cfg.get("side_confirm_sec", 0.12)))
        self.confirmation_distance_ratio = max(
            0.0, float(crossing_cfg.get("confirmation_distance_ratio", 0.06))
        )
        self.motion_confirmation_distance_ratio = max(
            0.0, float(crossing_cfg.get("motion_confirmation_distance_ratio", 0.05))
        )
        self.handoff_timeout_sec = max(
            0.0, float(crossing_cfg.get("handoff_timeout_sec", 1.0))
        )
        self.association_max_distance_ratio = max(
            0.1, float(crossing_cfg.get("association_max_distance_ratio", 0.25))
        )
        self.unambiguous_association_max_distance_ratio = max(
            self.association_max_distance_ratio,
            float(crossing_cfg.get("unambiguous_association_max_distance_ratio", 0.75)),
        )
        self.same_id_bonus = max(0.0, float(crossing_cfg.get("same_id_bonus", 0.1)))
        self.use_detection_point = bool(crossing_cfg.get("use_detection_point", True))
        ratio_cfg = crossing_cfg.get("point_y_ratio", 0.15)
        if isinstance(ratio_cfg, dict):
            ratio_cfg = ratio_cfg.get(camera_name, ratio_cfg.get("default", 0.15))
        self.point_y_ratio = float(ratio_cfg)
        self.tracks = {}
        self.next_stable_id = 1
        self.flow_direction = None
        self.status = "waiting"

    def reset(self, status="waiting"):
        self.tracks.clear()
        self.next_stable_id = 1
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
        point = getattr(detection, "point", None)
        if self.use_detection_point and point is not None:
            return tuple(point)
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

        tracked_people = [person for person in people if getattr(person, "track_id", None) is not None]
        assignments = self._associate(tracked_people, now)
        events = []
        count = current_count
        statuses = []
        for person, track, created, handed_off in assignments:
            event, status = self._update_track(
                track, person, line, width, height, now, count, created, handed_off
            )
            statuses.append(status)
            if event is not None:
                events.append(event)
                count = event.count_after

        if events:
            self.status = f"counted_{events[-1].event}"
        elif len(tracked_people) != len(people):
            self.status = "tracking_unavailable"
        elif statuses:
            self.status = statuses[-1]
        return events, self.status, people

    def track_visuals(self):
        visuals = []
        for track in self.tracks.values():
            visuals.append(
                {
                    "track_id": track.external_id,
                    "stable_id": track.stable_id,
                    "side": {1: "+", -1: "-", 0: "line"}[track.side],
                    "state": "armed" if track.armed else "stabilizing",
                    "point_source": track.point_source,
                    "trail": tuple(track.trail),
                }
            )
        return visuals

    def _associate(self, people, now):
        observations = []
        for person in people:
            point = self.detection_point(person)
            _, y1, _, y2 = person.box
            observations.append(
                {
                    "person": person,
                    "external_id": person.track_id,
                    "point": point,
                    "height": max(1.0, float(y2 - y1)),
                }
            )

        pairs = []
        extended_pairs = []
        for observation_index, observation in enumerate(observations):
            for stable_id, track in self.tracks.items():
                elapsed = max(0.0, now - track.last_seen_at)
                if elapsed > self.handoff_timeout_sec:
                    continue
                predicted = (
                    track.point[0] + track.velocity[0] * elapsed,
                    track.point[1] + track.velocity[1] * elapsed,
                )
                distance = hypot(
                    observation["point"][0] - predicted[0],
                    observation["point"][1] - predicted[1],
                )
                local_scale = max(observation["height"], track.box_height, 1.0)
                normalized = distance / local_scale
                if normalized > self.unambiguous_association_max_distance_ratio:
                    continue
                same_id = observation["external_id"] == track.external_id
                cost = normalized - (self.same_id_bonus if same_id else 0.0)
                pair = (cost, observation_index, stable_id, same_id)
                if same_id:
                    extended_pairs.append(pair)
                if normalized <= self.association_max_distance_ratio:
                    pairs.append(pair)

        matched_observations = set()
        matched_tracks = set()
        matches = {}
        for _, observation_index, stable_id, same_id in sorted(pairs):
            if observation_index in matched_observations or stable_id in matched_tracks:
                continue
            matched_observations.add(observation_index)
            matched_tracks.add(stable_id)
            matches[observation_index] = (self.tracks[stable_id], not same_id)

        remaining = [
            pair for pair in extended_pairs
            if pair[1] not in matched_observations and pair[2] not in matched_tracks
        ]
        observation_options = {}
        track_options = {}
        for pair in remaining:
            observation_options.setdefault(pair[1], []).append(pair)
            track_options.setdefault(pair[2], []).append(pair)
        unambiguous = [
            options[0]
            for observation_index, options in observation_options.items()
            if len(options) == 1 and len(track_options[options[0][2]]) == 1
        ]
        for _, observation_index, stable_id, same_id in sorted(unambiguous):
            matched_observations.add(observation_index)
            matched_tracks.add(stable_id)
            matches[observation_index] = (self.tracks[stable_id], not same_id)

        assignments = []
        for index, observation in enumerate(observations):
            match = matches.get(index)
            if match is None:
                stable_id = self.next_stable_id
                self.next_stable_id += 1
                track = _Track(
                    stable_id=stable_id,
                    external_id=observation["external_id"],
                    side=0,
                    anchor_point=observation["point"],
                    point=observation["point"],
                    raw_point=observation["point"],
                    box_height=observation["height"],
                    point_source=getattr(observation["person"], "point_source", "person"),
                    created_at=now,
                    last_seen_at=now,
                    samples=[observation["point"]],
                    trail=[observation["point"]],
                )
                self.tracks[stable_id] = track
                assignments.append((observation["person"], track, True, False))
            else:
                track, handed_off = match
                assignments.append((observation["person"], track, False, handed_off))
        return assignments

    def _update_track(self, track, person, line, width, height, now, count, created, handed_off):
        raw_point = self.detection_point(person)
        _, y1, _, y2 = person.box
        box_height = max(1.0, float(y2 - y1))
        point_source = getattr(person, "point_source", "person")
        same_external_id = track.external_id == person.track_id
        same_point_source = track.point_source == point_source

        if created:
            side = self._side(raw_point, line, min(width, height) * self.hysteresis_ratio)
            if (
                side != 0
                and self.min_track_frames <= 1
                and self.side_confirm_frames <= 1
                and self.side_confirm_sec <= 0
            ):
                track.side = side
                track.armed = True
                return None, "tracking"
            track.candidate_side = side
            track.candidate_since = now
            track.candidate_frames = 1 if side else 0
            return None, "near_line" if side == 0 else "stabilizing"

        elapsed = max(1e-6, now - track.last_seen_at)
        instant_velocity = (
            (raw_point[0] - track.raw_point[0]) / elapsed,
            (raw_point[1] - track.raw_point[1]) / elapsed,
        )
        track.velocity = (
            track.velocity[0] * 0.7 + instant_velocity[0] * 0.3,
            track.velocity[1] * 0.7 + instant_velocity[1] * 0.3,
        )
        track.external_id = person.track_id
        track.raw_point = raw_point
        track.box_height = track.box_height * 0.8 + box_height * 0.2
        track.point_source = point_source
        track.last_seen_at = now
        track.seen_frames += 1
        track.samples.append(raw_point)
        if len(track.samples) > self.point_filter_window:
            del track.samples[:-self.point_filter_window]
        point = (
            median(sample[0] for sample in track.samples),
            median(sample[1] for sample in track.samples),
        )
        track.point = point
        self._append_trail(track, point)

        margin = min(width, height) * self.hysteresis_ratio
        side = self._side(point, line, margin)
        motion_confirmed = False
        if track.armed and not handed_off and same_external_id and same_point_source:
            motion_side = self._motion_crossing_side(track, raw_point, line, margin)
            if motion_side:
                point = raw_point
                side = motion_side
                motion_confirmed = True
        if side == 0:
            self._clear_candidate(track)
            return None, "near_line"

        if not track.armed:
            self._record_candidate_side(track, side, now)
            if self._candidate_confirmed(track, now) and track.seen_frames >= self.min_track_frames:
                track.side = side
                track.anchor_point = point
                track.armed = True
                self._clear_candidate(track)
                return None, "track_handoff" if handed_off else "tracking"
            return None, "track_handoff" if handed_off else "stabilizing"

        if side == track.side:
            track.anchor_point = point
            self._clear_candidate(track)
            return None, "track_handoff" if handed_off else "tracking"

        self._record_candidate_side(track, side, now)
        if not motion_confirmed and not self._candidate_confirmed(track, now):
            return None, "crossing_pending"

        distance_ratio = (
            self.motion_confirmation_distance_ratio
            if motion_confirmed
            else self.confirmation_distance_ratio
        )
        confirmation_distance = max(margin, track.box_height * distance_ratio)
        if abs(self._signed_distance(point, line)) < confirmation_distance:
            return None, "crossing_pending"

        direction = "positive_to_negative" if track.side > side else "negative_to_positive"
        if not self._crosses_segment(track.anchor_point, point, line):
            self._reanchor(track, side, point)
            return None, "crossing_rejected"
        if self.one_event_per_track and track.counted:
            return None, "counted_track"
        if now < track.cooldown_until:
            self._reanchor(track, side, point)
            return None, "cooldown"

        mapped_event = self.direction_map.get(direction)
        if mapped_event not in {"enter", "exit"}:
            self._reanchor(track, side, point)
            return None, "unknown_direction"
        if self.flow_direction_lock and self.flow_direction and direction != self.flow_direction:
            self._reanchor(track, side, point)
            return None, "flow_reverse_ignored"
        if track.last_event == mapped_event:
            self._reanchor(track, side, point)
            return None, "duplicate_direction_ignored"

        before = count
        after = before + (1 if mapped_event == "enter" else -1)
        status = "ok"
        if after < 0:
            after = 0
            status = "blocked_negative_count"
        event = CountEvent(
            camera=self.camera_name,
            direction=direction,
            event=mapped_event,
            count_before=before,
            count_after=after,
            confidence=person.conf,
            status=status,
            track_id=person.track_id,
            stable_id=track.stable_id,
        )
        track.cooldown_until = now + self.cooldown_sec
        track.counted = True
        track.last_event = mapped_event
        if motion_confirmed:
            track.point = point
            track.samples = [point]
            self._append_trail(track, point)
        self._reanchor(track, side, point)
        self.flow_direction = direction
        return event, f"counted_{mapped_event}"

    def _candidate_confirmed(self, track, now):
        return (
            track.candidate_frames >= self.side_confirm_frames
            and now - track.candidate_since >= self.side_confirm_sec
        )

    def _motion_crossing_side(self, track, raw_point, line, margin):
        side = self._side(raw_point, line, margin)
        if side == 0 or side == track.side or len(track.samples) < self.point_filter_window:
            return 0
        distances = [
            self._signed_distance(point, line)
            for point in track.samples[-self.point_filter_window :]
        ]
        if any(
            side * (current - previous) <= 0
            for previous, current in zip(distances, distances[1:])
        ):
            return 0
        required = max(margin, track.box_height * self.motion_confirmation_distance_ratio)
        return side if abs(distances[-1]) >= required else 0

    @staticmethod
    def _record_candidate_side(track, side, now):
        if track.candidate_side == side:
            track.candidate_frames += 1
        else:
            track.candidate_side = side
            track.candidate_since = now
            track.candidate_frames = 1

    @staticmethod
    def _clear_candidate(track):
        track.candidate_side = 0
        track.candidate_since = 0.0
        track.candidate_frames = 0

    def _reanchor(self, track, side, point):
        track.side = side
        track.anchor_point = point
        track.armed = True
        self._clear_candidate(track)

    def _append_trail(self, track, point):
        track.trail.append(point)
        if len(track.trail) > self.trail_length:
            del track.trail[:-self.trail_length]

    def _drop_lost_tracks(self, now):
        lost = [
            stable_id
            for stable_id, track in self.tracks.items()
            if now - track.last_seen_at > self.lost_timeout_sec
        ]
        for stable_id in lost:
            del self.tracks[stable_id]

    @staticmethod
    def _signed_distance(point, line):
        (x1, y1), (x2, y2) = line
        return ((x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)) / hypot(
            x2 - x1, y2 - y1
        )

    @classmethod
    def _side(cls, point, line, margin):
        distance = cls._signed_distance(point, line)
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
