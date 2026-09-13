from dataclasses import dataclass, field
from math import hypot, sqrt
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
    corridor_zone: str = "unknown"
    corridor_origin: str = None
    corridor_candidate_zone: str = None
    corridor_candidate_since: float = 0.0
    corridor_candidate_frames: int = 0
    corridor_baseline: bool = False
    corridor_handoff_frames: int = 0
    corridor_pending_transition: tuple = None
    corridor_terminal_since: float = 0.0
    corridor_transit_start: tuple = None
    velocity: tuple = (0.0, 0.0)
    appearance: tuple = None
    samples: list = field(default_factory=list)
    trail: list = field(default_factory=list)


class LineCounter:
    def __init__(self, camera_name, config):
        self.camera_name = camera_name
        counter_cfg = config["counter"]
        crossing_cfg = config.get("crossing", {})
        self.counting_mode = str(crossing_cfg.get("counting_mode", "line")).lower()
        if self.counting_mode not in {"line", "corridor"}:
            self.counting_mode = "line"
        self.line_points = crossing_cfg.get("lines", {}).get(camera_name)
        self.corridor_config = crossing_cfg.get("corridors", {}).get(camera_name)
        corridor_timing = self.corridor_config if isinstance(self.corridor_config, dict) else {}
        self.direction_map = config["direction"][camera_name]
        self.min_area_ratio = float(counter_cfg.get("min_person_area_ratio", 0.02))
        self.lost_timeout_sec = float(
            corridor_timing.get("lost_timeout_sec", counter_cfg.get("lost_timeout_sec", 2.0))
        )
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
        self.zone_confirm_frames = max(
            1,
            int(
                corridor_timing.get(
                    "zone_confirm_frames",
                    crossing_cfg.get("zone_confirm_frames", self.side_confirm_frames),
                )
            ),
        )
        self.zone_confirm_sec = max(
            0.0,
            float(
                corridor_timing.get(
                    "zone_confirm_sec",
                    crossing_cfg.get("zone_confirm_sec", self.side_confirm_sec),
                )
            ),
        )
        self.initial_baseline_sec = max(
            0.0,
            float(
                corridor_timing.get(
                    "initial_baseline_sec",
                    crossing_cfg.get("initial_baseline_sec", 1.0),
                )
            ),
        )
        self.allow_transit_reentry = bool(
            corridor_timing.get(
                "allow_transit_reentry",
                crossing_cfg.get("allow_transit_reentry", False),
            )
        )
        self.corridor_handoff_confirm_frames = max(
            1,
            int(
                corridor_timing.get(
                    "corridor_handoff_confirm_frames",
                    crossing_cfg.get("corridor_handoff_confirm_frames", self.side_confirm_frames),
                )
            ),
        )
        self.corridor_reverse_min_terminal_sec = max(
            0.0,
            float(
                corridor_timing.get(
                    "corridor_reverse_min_terminal_sec",
                    crossing_cfg.get("corridor_reverse_min_terminal_sec", 0.0),
                )
            ),
        )
        self.journey_reacquire_timeout_sec = max(
            0.0,
            float(
                corridor_timing.get(
                    "journey_reacquire_timeout_sec",
                    crossing_cfg.get("journey_reacquire_timeout_sec", 0.0),
                )
            ),
        )
        self.transit_reentry_min_progress_ratio = max(
            0.0,
            float(
                corridor_timing.get(
                    "transit_reentry_min_progress_ratio",
                    crossing_cfg.get("transit_reentry_min_progress_ratio", 0.0),
                )
            ),
        )
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
        self.journey_reacquire_max_distance_ratio = max(
            self.association_max_distance_ratio,
            float(
                corridor_timing.get(
                    "journey_reacquire_max_distance_ratio",
                    crossing_cfg.get("journey_reacquire_max_distance_ratio", 8.0),
                )
            ),
        )
        self.same_id_bonus = max(0.0, float(crossing_cfg.get("same_id_bonus", 0.1)))
        self.appearance_memory_enabled = bool(
            crossing_cfg.get("appearance_memory_enabled", True)
        )
        self.appearance_weight = max(
            0.0, float(crossing_cfg.get("appearance_weight", 0.35))
        )
        self.appearance_max_distance = min(
            1.0, max(0.0, float(crossing_cfg.get("appearance_max_distance", 0.25)))
        )
        self.appearance_handoff_timeout_sec = max(
            self.handoff_timeout_sec,
            float(crossing_cfg.get("appearance_handoff_timeout_sec", 2.0)),
        )
        self.appearance_association_max_distance_ratio = max(
            self.association_max_distance_ratio,
            float(
                corridor_timing.get(
                    "appearance_association_max_distance_ratio",
                    crossing_cfg.get("appearance_association_max_distance_ratio", 1.0),
                )
            ),
        )
        self.appearance_handoff_timeout_sec = max(
            self.appearance_handoff_timeout_sec,
            float(corridor_timing.get("appearance_handoff_timeout_sec", 0.0)),
        )
        self.reverse_anchor_sec = max(
            0.0, float(counter_cfg.get("reverse_anchor_sec", 0.0))
        )
        self.use_detection_point = bool(crossing_cfg.get("use_detection_point", True))
        ratio_cfg = crossing_cfg.get("point_y_ratio", 0.15)
        if isinstance(ratio_cfg, dict):
            ratio_cfg = ratio_cfg.get(camera_name, ratio_cfg.get("default", 0.15))
        self.point_y_ratio = float(ratio_cfg)
        self.tracks = {}
        self.dormant_tracks = {}
        self.next_stable_id = 1
        self.flow_direction = None
        self.status = "waiting"
        self.started_at = None

    def reset(self, status="waiting"):
        self.tracks.clear()
        self.dormant_tracks.clear()
        self.next_stable_id = 1
        self.flow_direction = None
        self.status = status
        self.started_at = None

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

    def counting_corridor(self, width, height):
        """Return normalized corridor polygons in pixel coordinates, or None if invalid."""
        if not isinstance(self.corridor_config, dict):
            return None
        geometry = {}
        for name in ("outside", "transit", "inside"):
            polygon = self._scale_polygon(self.corridor_config.get(name), width, height)
            if polygon is None:
                return None
            geometry[name] = polygon
        return geometry

    def counting_geometry(self, width, height):
        if self.counting_mode == "corridor":
            return "corridor", self.counting_corridor(width, height)
        return "line", self.counting_line(width, height)

    @staticmethod
    def _scale_polygon(points, width, height):
        if not isinstance(points, (list, tuple)) or len(points) < 3:
            return None
        scaled = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None
            x, y = float(point[0]), float(point[1])
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                x *= width
                y *= height
            scaled.append((x, y))
        return tuple(scaled)

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
        if self.started_at is None:
            self.started_at = now
        height, width = frame_shape[:2]
        line = self.counting_line(width, height)
        corridor = self.counting_corridor(width, height)
        min_area = width * height * self.min_area_ratio
        people = [det for det in detections if det.area >= min_area]

        self._drop_lost_tracks(now)
        if not self.tracks:
            self.flow_direction = None
        if self.counting_mode == "line" and line is None:
            self.status = "line_not_configured"
            return [], self.status, people
        if self.counting_mode == "corridor" and corridor is None:
            self.status = "corridor_not_configured"
            return [], self.status, people
        if not people:
            self.status = "tracking" if self.tracks else "waiting"
            return [], self.status, people

        tracked_people = [person for person in people if getattr(person, "track_id", None) is not None]
        assignments = self._associate(tracked_people, now, corridor)
        events = []
        count = current_count
        statuses = []
        for person, track, created, handed_off in assignments:
            if self.counting_mode == "corridor":
                event, status = self._update_corridor_track(
                    track, person, corridor, now, count, created, handed_off
                )
            else:
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
            if self.counting_mode == "corridor":
                side = track.corridor_zone
                state = (
                    f"{track.corridor_origin}->transit"
                    if track.corridor_origin
                    else side
                )
            else:
                side = {1: "+", -1: "-", 0: "line"}[track.side]
                state = "armed" if track.armed else "stabilizing"
            visuals.append(
                {
                    "track_id": track.external_id,
                    "stable_id": track.stable_id,
                    "side": side,
                    "state": state,
                    "point_source": track.point_source,
                    "trail": tuple(track.trail),
                }
            )
        return visuals

    def _associate(self, people, now, corridor=None):
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
                    "appearance": (
                        self._appearance(getattr(person, "appearance", None))
                        if self.appearance_memory_enabled
                        else None
                    ),
                    "corridor_zone": (
                        self._corridor_zone(point, corridor)
                        if corridor is not None
                        else None
                    ),
                }
            )

        # Some trackers temporarily assign one raw ID to several detections in
        # the same frame.  That ID cannot be evidence of identity continuity:
        # using it as a bonus lets one person steal another person's journey.
        raw_id_counts = {}
        for observation in observations:
            raw_id = observation["external_id"]
            if raw_id is not None:
                raw_id_counts[raw_id] = raw_id_counts.get(raw_id, 0) + 1

        candidate_tracks = [(stable_id, track, False) for stable_id, track in self.tracks.items()]
        if self.counting_mode == "corridor" and self.journey_reacquire_timeout_sec > 0:
            candidate_tracks.extend(
                (stable_id, track, True)
                for stable_id, track in self.dormant_tracks.items()
                if now - track.last_seen_at <= self.journey_reacquire_timeout_sec
            )

        pairs = []
        extended_pairs = []
        for observation_index, observation in enumerate(observations):
            for stable_id, track, dormant in candidate_tracks:
                elapsed = max(0.0, now - track.last_seen_at)
                if dormant:
                    if observation["corridor_zone"] != "transit":
                        continue
                    if track.corridor_zone not in {"outside", "inside"}:
                        continue
                    max_elapsed = self.journey_reacquire_timeout_sec
                else:
                    max_elapsed = self.appearance_handoff_timeout_sec
                if elapsed > max_elapsed:
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
                same_id = (
                    observation["external_id"] is not None
                    and observation["external_id"] == track.external_id
                )
                raw_id_is_unique = raw_id_counts.get(observation["external_id"], 0) == 1
                trusted_same_id = (
                    same_id
                    and raw_id_is_unique
                    and normalized <= self.association_max_distance_ratio
                )
                appearance_distance = self._appearance_distance(
                    observation["appearance"], track.appearance
                )
                appearance_match = (
                    appearance_distance is not None
                    and appearance_distance <= self.appearance_max_distance
                )
                appearance_mismatch = (
                    appearance_distance is not None
                    and not appearance_match
                )
                same_raw_reacquire = dormant and same_id and raw_id_is_unique
                appearance_reacquire = dormant and appearance_match
                if dormant and not (same_raw_reacquire or appearance_reacquire):
                    continue
                if appearance_mismatch and not trusted_same_id:
                    continue
                if appearance_mismatch:
                    appearance_distance = None
                regular_match = (
                    elapsed <= self.handoff_timeout_sec
                    and normalized <= self.unambiguous_association_max_distance_ratio
                )
                appearance_handoff = (
                    appearance_match
                    and elapsed <= self.appearance_handoff_timeout_sec
                    and normalized <= self.appearance_association_max_distance_ratio
                )
                journey_reacquire = (
                    dormant
                    and (same_raw_reacquire or appearance_reacquire)
                    and normalized <= self.journey_reacquire_max_distance_ratio
                )
                if not regular_match and not appearance_handoff and not journey_reacquire:
                    continue
                cost = normalized
                if appearance_distance is not None:
                    cost += appearance_distance * self.appearance_weight
                if same_id and raw_id_is_unique:
                    cost -= self.same_id_bonus
                pair = (cost, observation_index, stable_id, same_id, dormant)
                if (same_id and raw_id_is_unique) or appearance_handoff or journey_reacquire:
                    extended_pairs.append(pair)
                if (
                    normalized <= self.association_max_distance_ratio
                    or appearance_handoff
                    or journey_reacquire
                ):
                    pairs.append(pair)

        matched_observations = set()
        matched_tracks = set()
        matches = {}
        for _, observation_index, stable_id, same_id, dormant in sorted(pairs):
            if observation_index in matched_observations or stable_id in matched_tracks:
                continue
            matched_observations.add(observation_index)
            matched_tracks.add(stable_id)
            track = self.dormant_tracks[stable_id] if dormant else self.tracks[stable_id]
            matches[observation_index] = (track, not same_id, dormant)

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
        for _, observation_index, stable_id, same_id, dormant in sorted(unambiguous):
            matched_observations.add(observation_index)
            matched_tracks.add(stable_id)
            track = self.dormant_tracks[stable_id] if dormant else self.tracks[stable_id]
            matches[observation_index] = (track, not same_id, dormant)

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
                    appearance=observation["appearance"],
                    created_at=now,
                    last_seen_at=now,
                    samples=[observation["point"]],
                    trail=[observation["point"]],
                )
                self.tracks[stable_id] = track
                assignments.append((observation["person"], track, True, False))
            else:
                track, handed_off, dormant = match
                if dormant:
                    del self.dormant_tracks[track.stable_id]
                    self.tracks[track.stable_id] = track
                    self._reactivate_corridor_track(track, observation, now)
                assignments.append((observation["person"], track, False, handed_off))
        return assignments

    def _reactivate_corridor_track(self, track, observation, now):
        """Keep the journey state, but restart motion prediction after a gap."""
        point = observation["point"]
        track.external_id = observation["external_id"]
        track.raw_point = point
        track.point = point
        track.box_height = observation["height"]
        track.point_source = getattr(observation["person"], "point_source", "person")
        track.velocity = (0.0, 0.0)
        track.last_seen_at = now
        track.samples = [point]
        self._append_trail(track, point)

    def _update_corridor_track(self, track, person, corridor, now, count, created, handed_off):
        """Count only a completed outside -> transit -> inside (or reverse) journey."""
        raw_point = self.detection_point(person)
        _, y1, _, y2 = person.box
        box_height = max(1.0, float(y2 - y1))
        point_source = getattr(person, "point_source", "person")
        previous_raw_point = track.raw_point
        same_point_source = track.point_source == point_source

        if created:
            zone = self._corridor_zone(raw_point, corridor)
            track.corridor_zone = zone
            track.corridor_baseline = (
                self.initial_baseline_sec > 0
                and zone == "inside"
                and now - self.started_at <= self.initial_baseline_sec
            )
            track.corridor_candidate_zone = zone
            track.corridor_candidate_since = now
            track.corridor_candidate_frames = 1
            if zone in {"outside", "inside"}:
                track.corridor_terminal_since = now
            if (
                zone == "transit"
                and self.allow_transit_reentry
                and now - self.started_at > self.initial_baseline_sec
            ):
                track.corridor_origin = "transit_reentry"
                track.corridor_transit_start = raw_point
            return None, "corridor_transit_unarmed" if zone == "transit" else "tracking"

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
        if self.appearance_memory_enabled:
            track.appearance = self._merge_appearance(
                track.appearance, self._appearance(getattr(person, "appearance", None))
            )
        track.last_seen_at = now
        track.seen_frames += 1
        track.corridor_handoff_frames = (
            1 if handed_off else track.corridor_handoff_frames + 1
        )
        track.samples.append(raw_point)
        if len(track.samples) > self.point_filter_window:
            del track.samples[:-self.point_filter_window]
        track.point = (
            median(sample[0] for sample in track.samples),
            median(sample[1] for sample in track.samples),
        )
        self._append_trail(track, track.point)

        # A head point may disappear briefly and fall back to the person-box
        # point (or the reverse).  That is not an identity handoff.  Ignore the
        # switch frame for zone state so it cannot erase an armed passage; if
        # the new source persists, its next sample participates normally.
        if not same_point_source and not handed_off:
            return None, "corridor_point_source_changed"

        zone = self._corridor_zone(raw_point, corridor)
        if zone == "unknown":
            return None, "corridor_between_zones"
        if not self._record_corridor_candidate(track, zone, now):
            return None, "corridor_pending"
        previous_zone = track.corridor_zone
        if zone == previous_zone:
            pending = track.corridor_pending_transition
            if (
                pending is not None
                and pending[1] == zone
                and track.corridor_handoff_frames >= self.corridor_handoff_confirm_frames
            ):
                track.corridor_pending_transition = None
                return self._complete_corridor_path(
                    track, person, pending[0], pending[1], now, count
                )
            return None, "tracking"
        track.corridor_pending_transition = None

        # A detector can skip every frame in a narrow transit path.  Accept the
        # direct terminal change only when the swept segment actually intersects
        # the configured transit polygon.
        direct_terminal_change = (
            previous_zone in {"outside", "inside"}
            and zone in {"outside", "inside"}
            and previous_zone != zone
            and self._segment_touches_polygon(previous_raw_point, raw_point, corridor["transit"])
        )
        if direct_terminal_change:
            return self._complete_or_defer_corridor_path(
                track, person, previous_zone, zone, now, count
            )

        if zone == "transit":
            track.corridor_origin = previous_zone if previous_zone in {"outside", "inside"} else None
            track.corridor_transit_start = None
            track.corridor_zone = "transit"
            return None, "corridor_in_transit" if track.corridor_origin else "corridor_transit_unarmed"

        if previous_zone == "transit":
            origin = track.corridor_origin
            track.corridor_zone = zone
            track.corridor_origin = None
            if origin in {"outside", "inside"} and origin != zone:
                return self._complete_or_defer_corridor_path(
                    track, person, origin, zone, now, count
                )
            if origin == "transit_reentry":
                if not self._has_transit_reentry_progress(track, raw_point, zone, corridor):
                    track.corridor_terminal_since = now
                    track.corridor_transit_start = None
                    return None, "corridor_transit_reentry_unconfirmed"
                inferred_origin = "inside" if zone == "outside" else "outside"
                track.corridor_transit_start = None
                return self._complete_or_defer_corridor_path(
                    track, person, inferred_origin, zone, now, count
                )
            return None, "corridor_turnback" if origin == zone else "tracking"

        track.corridor_zone = zone
        track.corridor_origin = None
        return None, "tracking"

    def _complete_or_defer_corridor_path(self, track, person, origin, destination, now, count):
        if track.corridor_handoff_frames < self.corridor_handoff_confirm_frames:
            track.corridor_pending_transition = (origin, destination)
            return None, "corridor_handoff_pending"
        return self._complete_corridor_path(track, person, origin, destination, now, count)

    def _complete_corridor_path(self, track, person, origin, destination, now, count):
        track.corridor_zone = destination
        track.corridor_origin = None
        if track.corridor_baseline:
            return None, "corridor_baseline"
        direction = f"{origin}_to_{destination}"
        events = self.corridor_config.get("events", {}) if self.corridor_config else {}
        mapped_event = events.get(direction)
        if mapped_event not in {"enter", "exit"}:
            return None, "unknown_direction"
        if (
            track.last_event is not None
            and mapped_event != track.last_event
            and now - track.corridor_terminal_since < self.corridor_reverse_min_terminal_sec
        ):
            track.corridor_terminal_since = now
            return None, "corridor_reverse_unsettled"
        if self.one_event_per_track and track.counted:
            return None, "counted_track"
        if now < track.cooldown_until:
            return None, "cooldown"
        if track.last_event == mapped_event:
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
        track.corridor_terminal_since = now
        return event, f"counted_{mapped_event}"

    def _record_corridor_candidate(self, track, zone, now):
        if track.corridor_candidate_zone == zone:
            track.corridor_candidate_frames += 1
        else:
            track.corridor_candidate_zone = zone
            track.corridor_candidate_since = now
            track.corridor_candidate_frames = 1
        return (
            track.corridor_candidate_frames >= self.zone_confirm_frames
            and now - track.corridor_candidate_since >= self.zone_confirm_sec
        )

    @staticmethod
    def _corridor_zone(point, corridor):
        for zone in ("outside", "inside", "transit"):
            if LineCounter._point_in_polygon(point, corridor[zone]):
                return zone
        return "unknown"

    def _has_transit_reentry_progress(self, track, point, destination, corridor):
        if self.transit_reentry_min_progress_ratio <= 0:
            return True
        start = track.corridor_transit_start
        if start is None:
            return False
        origin = "inside" if destination == "outside" else "outside"
        origin_center = self._polygon_center(corridor[origin])
        destination_center = self._polygon_center(corridor[destination])
        axis_x = destination_center[0] - origin_center[0]
        axis_y = destination_center[1] - origin_center[1]
        squared_length = axis_x * axis_x + axis_y * axis_y
        if squared_length <= 1e-9:
            return False
        progress = (
            (point[0] - start[0]) * axis_x + (point[1] - start[1]) * axis_y
        ) / squared_length
        return progress >= self.transit_reentry_min_progress_ratio

    @staticmethod
    def _polygon_center(polygon):
        return (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )

    @staticmethod
    def _point_in_polygon(point, polygon):
        x, y = point
        inside = False
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            if LineCounter._point_on_segment(point, first, second):
                return True
            x1, y1 = first
            x2, y2 = second
            if (y1 > y) != (y2 > y):
                cross_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < cross_x:
                    inside = not inside
        return inside

    @staticmethod
    def _point_on_segment(point, first, second, epsilon=1e-6):
        px, py = point
        x1, y1 = first
        x2, y2 = second
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) > epsilon:
            return False
        return (
            min(x1, x2) - epsilon <= px <= max(x1, x2) + epsilon
            and min(y1, y2) - epsilon <= py <= max(y1, y2) + epsilon
        )

    @classmethod
    def _segment_touches_polygon(cls, first, second, polygon):
        if cls._point_in_polygon(first, polygon) or cls._point_in_polygon(second, polygon):
            return True
        return any(
            cls._segments_intersect(first, second, edge_start, edge_end)
            for edge_start, edge_end in zip(polygon, polygon[1:] + polygon[:1])
        )

    @staticmethod
    def _segments_intersect(a, b, c, d):
        def orientation(first, second, third):
            value = (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])
            if abs(value) < 1e-9:
                return 0
            return 1 if value > 0 else -1

        ab_c, ab_d = orientation(a, b, c), orientation(a, b, d)
        cd_a, cd_b = orientation(c, d, a), orientation(c, d, b)
        if ab_c != ab_d and cd_a != cd_b:
            return True
        return (
            (ab_c == 0 and LineCounter._point_on_segment(c, a, b))
            or (ab_d == 0 and LineCounter._point_on_segment(d, a, b))
            or (cd_a == 0 and LineCounter._point_on_segment(a, c, d))
            or (cd_b == 0 and LineCounter._point_on_segment(b, c, d))
        )

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

        motion_snapshot = (
            track.raw_point,
            track.point,
            track.velocity,
            track.box_height,
            track.point_source,
            track.last_seen_at,
            track.seen_frames,
            list(track.samples),
            list(track.trail),
        )
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
        if self.appearance_memory_enabled:
            track.appearance = self._merge_appearance(
                track.appearance, self._appearance(getattr(person, "appearance", None))
            )
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
        if (
            track.last_event is not None
            and self.reverse_anchor_sec > 0
            and now - track.candidate_since < self.reverse_anchor_sec
        ):
            self._restore_motion(track, motion_snapshot)
            return None, "reverse_pending"

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

    @staticmethod
    def _restore_motion(track, snapshot):
        (
            track.raw_point,
            track.point,
            track.velocity,
            track.box_height,
            track.point_source,
            track.last_seen_at,
            track.seen_frames,
            track.samples,
            track.trail,
        ) = snapshot

    def _drop_lost_tracks(self, now):
        lost = [
            stable_id
            for stable_id, track in self.tracks.items()
            if now - track.last_seen_at > self.lost_timeout_sec
        ]
        for stable_id in lost:
            track = self.tracks.pop(stable_id)
            if (
                self.counting_mode == "corridor"
                and self.journey_reacquire_timeout_sec > 0
                and track.corridor_zone in {"outside", "inside"}
            ):
                self.dormant_tracks[stable_id] = track
        stale_dormant = [
            stable_id
            for stable_id, track in self.dormant_tracks.items()
            if now - track.last_seen_at > self.journey_reacquire_timeout_sec
        ]
        for stable_id in stale_dormant:
            del self.dormant_tracks[stable_id]

    @staticmethod
    def _appearance(value):
        if value is None:
            return None
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if not values or not all(item == item for item in values):
            return None
        magnitude = sqrt(sum(item * item for item in values))
        if magnitude <= 1e-9:
            return None
        return tuple(item / magnitude for item in values)

    @staticmethod
    def _appearance_distance(first, second):
        if first is None or second is None or len(first) != len(second):
            return None
        similarity = sum(left * right for left, right in zip(first, second))
        return max(0.0, min(1.0, 1.0 - similarity))

    @staticmethod
    def _merge_appearance(previous, current):
        if current is None:
            return previous
        if previous is None or len(previous) != len(current):
            return current
        return LineCounter._appearance(
            tuple(old * 0.8 + new * 0.2 for old, new in zip(previous, current))
        )

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
