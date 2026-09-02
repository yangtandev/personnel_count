from dataclasses import dataclass


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
    anchor_zone: str
    point: tuple
    last_seen_at: float
    cooldown_until: float = 0.0
    counted: bool = False
    ignored_reverse_zone: str = None
    ignored_reverse_since: float = 0.0


class ZoneCounter:
    def __init__(self, camera_name, config):
        self.camera_name = camera_name
        self.zones = config["zones"]
        self.direction_map = config["direction"][camera_name]
        counter_cfg = config["counter"]
        self.min_area_ratio = float(counter_cfg.get("min_person_area_ratio", 0.02))
        self.lost_timeout_sec = float(counter_cfg.get("lost_timeout_sec", 2.0))
        self.cooldown_sec = float(counter_cfg.get("event_cooldown_sec", 1.5))
        self.one_event_per_track = bool(counter_cfg.get("one_event_per_track", False))
        self.flow_direction_lock = bool(counter_cfg.get("flow_direction_lock", False))
        self.reverse_anchor_sec = float(counter_cfg.get("reverse_anchor_sec", 0.5))
        ratio_cfg = self.zones.get("zone_point_y_ratio", 0.05)
        if isinstance(ratio_cfg, dict):
            ratio_cfg = ratio_cfg.get(self.camera_name, ratio_cfg.get("default", 0.05))
        self.zone_point_y_ratio = float(ratio_cfg)
        self.use_detection_point = bool(self.zones.get("use_detection_point", False))
        self.tracks = {}
        self.next_track_id = 1
        self.flow_direction = None
        self.status = "waiting"

    def zone_layout(self):
        labels = self.zones.get("labels", {}).get(self.camera_name, {})
        return labels.get("default", "A"), labels.get("target", "B")

    def zone_polygons(self, width, height):
        regions = self.zones.get("regions", {})
        if not isinstance(regions, dict):
            return []

        camera_regions = regions.get(self.camera_name, regions)
        if not isinstance(camera_regions, dict):
            return []

        default_label, target_label = self.zone_layout()
        polygons = []
        for label in (default_label, target_label):
            polygon = self._scale_polygon(camera_regions.get(label), width, height)
            if len(polygon) >= 3:
                polygons.append((label, polygon))
        return polygons

    def detection_point(self, detection):
        if self.use_detection_point and getattr(detection, "point", None) is not None:
            return detection.point
        x1, y1, x2, y2 = detection.box
        return (
            (x1 + x2) / 2,
            y1 + max(0, y2 - y1) * self.zone_point_y_ratio,
        )

    def zone_for_point(self, x, y, width, height):
        default_label, target_label = self.zone_layout()
        polygons = dict(self.zone_polygons(width, height))
        for label in (target_label, default_label):
            polygon = polygons.get(label)
            if polygon and _point_in_polygon(x, y, polygon):
                return label

        if default_label not in polygons and target_label in polygons:
            return default_label
        return None

    def _scale_polygon(self, points, width, height):
        if not isinstance(points, list):
            return []
        polygon = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            x, y = float(point[0]), float(point[1])
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
                x *= width
                y *= height
            polygon.append((x, y))
        return polygon

    def reset(self, status="waiting"):
        self.tracks.clear()
        self.next_track_id = 1
        self.flow_direction = None
        self.status = status

    def update(self, detections, frame_shape, now, current_count):
        height, width = frame_shape[:2]
        min_area = width * height * self.min_area_ratio
        people = []
        candidate_data = []
        for det in detections:
            if det.area < min_area:
                continue
            point = self.detection_point(det)
            zone = self.zone_for_point(point[0], point[1], width, height)
            if zone is not None:
                people.append(det)
            candidate_data.append((det, zone, point))

        had_tracks = bool(self.tracks)
        self._drop_lost_tracks(now)
        if not self.tracks:
            self.flow_direction = None
        if not candidate_data:
            if had_tracks and not self.tracks:
                self.status = "incomplete_path"
            else:
                self.status = "tracking" if self.tracks else "waiting"
            return [], self.status, people

        events = []
        count = current_count
        unmatched_track_ids = set(self.tracks)
        max_match_distance = max(width, height) * 0.5

        for person, zone, point in sorted(candidate_data, key=lambda item: item[2][0]):
            track_id = self._track_id_for(person, point, unmatched_track_ids, max_match_distance)
            if track_id is None or track_id not in self.tracks:
                if zone is None:
                    continue
                if track_id is None:
                    track_id = self.next_track_id
                    self.next_track_id += 1
                self.tracks[track_id] = _Track(zone, point, now)
                self.status = f"seen_{zone}"
                continue

            unmatched_track_ids.discard(track_id)
            track = self.tracks[track_id]
            track.last_seen_at = now
            track.point = point
            if zone is None:
                self.status = "tracking"
                continue

            if zone == track.anchor_zone:
                track.ignored_reverse_zone = None
                self.status = f"seen_{zone}"
                continue

            if self.one_event_per_track and track.counted:
                self.status = "counted_track"
                continue

            if now < track.cooldown_until:
                self.status = "cooldown"
                continue

            direction = f"{track.anchor_zone}_to_{zone}"
            mapped_event = self.direction_map.get(direction)
            if mapped_event not in {"enter", "exit"}:
                track.anchor_zone = zone
                self.status = "unknown_direction"
                continue

            if self.flow_direction_lock and self.flow_direction and direction != self.flow_direction:
                if track.ignored_reverse_zone != zone:
                    track.ignored_reverse_zone = zone
                    track.ignored_reverse_since = now
                elif now - track.ignored_reverse_since >= self.reverse_anchor_sec:
                    track.anchor_zone = zone
                    track.ignored_reverse_zone = None
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
            track.anchor_zone = zone
            track.cooldown_until = now + self.cooldown_sec
            track.counted = True
            self.flow_direction = direction
            self.status = f"counted_{mapped_event}"

        return events, self.status, people

    def _track_id_for(self, detection, point, unmatched_track_ids, max_match_distance):
        external_id = getattr(detection, "track_id", None)
        if external_id is not None:
            return f"det:{external_id}"
        return self._match_track(point, unmatched_track_ids, max_match_distance)

    def _drop_lost_tracks(self, now):
        lost = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track.last_seen_at > self.lost_timeout_sec
        ]
        for track_id in lost:
            del self.tracks[track_id]

    def _match_track(self, point, track_ids, max_distance):
        best_track_id = None
        best_distance = max_distance * max_distance
        for track_id in track_ids:
            track_point = self.tracks[track_id].point
            distance = (point[0] - track_point[0]) ** 2 + (point[1] - track_point[1]) ** 2
            if distance <= best_distance:
                best_track_id = track_id
                best_distance = distance
        return best_track_id


def _point_in_polygon(x, y, polygon):
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if _point_on_segment(x, y, previous_x, previous_y, current_x, current_y):
            return True
        crosses_y = (current_y > y) != (previous_y > y)
        if crosses_y:
            slope_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < slope_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _point_on_segment(px, py, ax, ay, bx, by):
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-9:
        return False
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)
