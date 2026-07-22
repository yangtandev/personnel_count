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


class ZoneCounter:
    def __init__(self, camera_name, config):
        self.camera_name = camera_name
        self.zones = config["zones"]
        self.direction_map = config["direction"][camera_name]
        counter_cfg = config["counter"]
        self.min_area_ratio = float(counter_cfg.get("min_person_area_ratio", 0.02))
        self.lost_timeout_sec = float(counter_cfg.get("lost_timeout_sec", 2.0))
        self.cooldown_sec = float(counter_cfg.get("event_cooldown_sec", 1.5))
        self.zone_point_y_ratio = float(self.zones.get("zone_point_y_ratio", 0.35))
        self.anchor_zone = None
        self.last_seen_at = None
        self.cooldown_until = 0.0
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

        _, target_label = self.zone_layout()
        polygon = self._scale_polygon(camera_regions.get(target_label, camera_regions.get("B")), width, height)
        if len(polygon) < 3:
            return []
        return [(target_label, polygon)]

    def detection_point(self, detection):
        x1, y1, x2, y2 = detection.box
        return (
            (x1 + x2) / 2,
            y1 + max(0, y2 - y1) * self.zone_point_y_ratio,
        )

    def zone_for_point(self, x, y, width, height):
        default_label, target_label = self.zone_layout()
        for _, polygon in self.zone_polygons(width, height):
            if _point_in_polygon(x, y, polygon):
                return target_label
        return default_label

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
        self.anchor_zone = None
        self.last_seen_at = None
        self.status = status

    def update(self, detections, frame_shape, now, current_count):
        height, width = frame_shape[:2]
        min_area = width * height * self.min_area_ratio
        people = [det for det in detections if det.area >= min_area]

        if len(people) > 1:
            self.reset("paused_multi_person")
            return None, self.status, people

        if not people:
            if self.last_seen_at is not None and now - self.last_seen_at > self.lost_timeout_sec:
                self.reset("incomplete_path")
            else:
                self.status = "waiting" if self.anchor_zone is None else "tracking"
            return None, self.status, people

        person = people[0]
        self.last_seen_at = now
        point_x, point_y = self.detection_point(person)
        zone = self.zone_for_point(point_x, point_y, width, height)

        if now < self.cooldown_until:
            self.status = "cooldown"
            return None, self.status, people

        if self.anchor_zone is None:
            self.anchor_zone = zone
            self.status = f"seen_{zone}"
            return None, self.status, people

        if zone == self.anchor_zone:
            self.status = f"seen_{zone}"
            return None, self.status, people

        direction = f"{self.anchor_zone}_to_{zone}"
        mapped_event = self.direction_map.get(direction)
        if mapped_event not in {"enter", "exit"}:
            self.anchor_zone = zone
            self.status = "unknown_direction"
            return None, self.status, people

        before = current_count
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
        )
        self.anchor_zone = zone
        self.cooldown_until = now + self.cooldown_sec
        self.status = f"counted_{mapped_event}"
        return event, self.status, people


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
