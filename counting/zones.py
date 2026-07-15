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
        self.require_middle = bool(counter_cfg.get("require_middle", False))
        self.anchor_zone = None
        self.saw_middle = False
        self.last_seen_at = None
        self.cooldown_until = 0.0
        self.status = "waiting"

    def zone_layout(self):
        labels = self.zones.get("labels", {}).get(self.camera_name, {})
        return (
            labels.get("left", "A"),
            labels.get("right", "B"),
        )

    def zone_regions(self, width):
        if "left_width_ratio" in self.zones or "right_width_ratio" in self.zones:
            left_width = float(self.zones.get("left_width_ratio", 0.34))
            right_width = float(self.zones.get("right_width_ratio", 0.34))
            if left_width + right_width > 1:
                left_width = right_width = 0.5
            return (
                0,
                width * left_width,
                width * (1 - right_width),
                width,
            )

        zone_w = width * float(self.zones.get("zone_width_ratio", 0.12))
        left_center = width * float(self.zones.get("left_ratio", 0.25))
        right_center = width * float(self.zones.get("right_ratio", 0.75))
        half_zone = zone_w / 2
        if self.zones.get("mode", "side") == "band":
            return (
                max(0, left_center - half_zone),
                min(width, left_center + half_zone),
                max(0, right_center - half_zone),
                min(width, right_center + half_zone),
            )
        return (
            0,
            min(width, left_center + half_zone),
            max(0, right_center - half_zone),
            width,
        )

    def zone_for_x(self, x, width):
        left_label, right_label = self.zone_layout()
        left_x1, left_x2, right_x1, right_x2 = self.zone_regions(width)
        if left_x1 <= x <= left_x2:
            return left_label
        if right_x1 <= x <= right_x2:
            return right_label
        return "middle"

    def reset(self, status="waiting"):
        self.anchor_zone = None
        self.saw_middle = False
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
        zone = self.zone_for_x(person.center_x, width)

        if now < self.cooldown_until:
            self.status = "cooldown"
            return None, self.status, people

        if zone == "middle":
            if self.anchor_zone is not None:
                self.saw_middle = True
                self.status = f"{self.anchor_zone}_to_middle"
            else:
                self.status = "middle_only"
            return None, self.status, people

        if self.anchor_zone is None:
            self.anchor_zone = zone
            self.saw_middle = False
            self.status = f"seen_{zone}"
            return None, self.status, people

        if zone == self.anchor_zone:
            if self.saw_middle:
                self.saw_middle = False
                self.status = f"returned_{zone}"
            else:
                self.status = f"seen_{zone}"
            return None, self.status, people

        direction = f"{self.anchor_zone}_to_{zone}"
        if self.require_middle and not self.saw_middle:
            self.anchor_zone = zone
            self.status = f"seen_{zone}"
            return None, self.status, people

        mapped_event = self.direction_map.get(direction)
        if mapped_event not in {"enter", "exit"}:
            self.anchor_zone = zone
            self.saw_middle = False
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
        self.saw_middle = False
        self.cooldown_until = now + self.cooldown_sec
        self.status = f"counted_{mapped_event}"
        return event, self.status, people
