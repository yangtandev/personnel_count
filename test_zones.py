import unittest
from dataclasses import dataclass

from counting.lines import LineCounter


@dataclass(frozen=True)
class Detection:
    box: tuple
    conf: float = 0.9
    track_id: int = None

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def detection(center_x, center_y=50, track_id=None):
    return Detection((center_x - 5, center_y - 10, center_x + 5, center_y + 10), track_id=track_id)


def make_counter(flow_direction_lock=False, line=None):
    return LineCounter(
        "top",
        {
            "counter": {
                "min_person_area_ratio": 0.0,
                "lost_timeout_sec": 1.0,
                "event_cooldown_sec": 0.1,
                "one_event_per_track": False,
                "flow_direction_lock": flow_direction_lock,
            },
            "crossing": {
                "lines": {"top": line or [(0.5, 0.0), (0.5, 1.0)]},
                "point_y_ratio": 0.5,
                "hysteresis_ratio": 0.02,
                "segment_margin_ratio": 0.0,
                "min_track_frames": 2,
                "trail_length": 3,
            },
            "direction": {
                "top": {
                    "positive_to_negative": "enter",
                    "negative_to_positive": "exit",
                }
            },
        },
    )


class LineCounterTest(unittest.TestCase):
    def test_counts_one_tracked_person_crossing(self):
        counter = make_counter()
        counter.update([detection(35, track_id=1)], (100, 100, 3), 0.0, 0)
        events, status, people = counter.update(
            [detection(65, track_id=1)], (100, 100, 3), 0.2, 0
        )

        self.assertEqual([event.event for event in events], ["enter"])
        self.assertEqual(events[0].count_after, 1)
        self.assertEqual(status, "counted_enter")
        self.assertEqual(len(people), 1)

    def test_tracks_simultaneous_opposite_crossings_independently(self):
        counter = make_counter()
        counter.update(
            [detection(35, track_id=1), detection(65, track_id=2)],
            (100, 100, 3),
            0.0,
            5,
        )
        events, _, _ = counter.update(
            [detection(65, track_id=1), detection(35, track_id=2)],
            (100, 100, 3),
            0.2,
            5,
        )

        self.assertEqual([event.event for event in events], ["enter", "exit"])
        self.assertEqual([event.count_after for event in events], [6, 5])

    def test_does_not_mix_different_track_ids(self):
        counter = make_counter()
        counter.update([detection(65, track_id=1)], (100, 100, 3), 0.0, 0)
        events, _, _ = counter.update(
            [detection(65, track_id=1), detection(35, track_id=2)],
            (100, 100, 3),
            0.2,
            0,
        )

        self.assertEqual(events, [])

    def test_untracked_detections_never_count(self):
        counter = make_counter()
        counter.update([detection(65)], (100, 100, 3), 0.0, 3)
        events, status, _ = counter.update([detection(35)], (100, 100, 3), 0.2, 3)

        self.assertEqual(events, [])
        self.assertEqual(status, "tracking_unavailable")
        self.assertEqual(counter.tracks, {})

    def test_flow_lock_ignores_opposite_crossing(self):
        counter = make_counter(flow_direction_lock=True)
        counter.update(
            [detection(35, track_id=1), detection(65, track_id=2)],
            (100, 100, 3),
            0.0,
            5,
        )
        events, status, _ = counter.update(
            [detection(65, track_id=1), detection(35, track_id=2)],
            (100, 100, 3),
            0.2,
            5,
        )

        self.assertEqual([event.event for event in events], ["enter"])
        self.assertEqual(status, "counted_enter")

    def test_hysteresis_ignores_line_jitter(self):
        counter = make_counter()
        counter.update([detection(40, track_id=1)], (100, 100, 3), 0.0, 0)
        for index, center_x in enumerate((49, 51, 48, 40), start=1):
            events, _, _ = counter.update(
                [detection(center_x, track_id=1)],
                (100, 100, 3),
                index * 0.1,
                0,
            )
            self.assertEqual(events, [])

    def test_crossing_outside_finite_line_is_rejected(self):
        counter = make_counter(line=[(50, 20), (50, 80)])
        counter.update([detection(35, 95, track_id=1)], (100, 100, 3), 0.0, 0)
        events, status, _ = counter.update(
            [detection(65, 95, track_id=1)], (100, 100, 3), 0.2, 0
        )

        self.assertEqual(events, [])
        self.assertEqual(status, "crossing_rejected")

    def test_exit_does_not_make_count_negative(self):
        counter = make_counter()
        counter.update([detection(65, track_id=1)], (100, 100, 3), 0.0, 0)
        events, _, _ = counter.update(
            [detection(35, track_id=1)], (100, 100, 3), 0.2, 0
        )

        self.assertEqual([event.event for event in events], ["exit"])
        self.assertEqual(events[0].count_after, 0)
        self.assertEqual(events[0].status, "blocked_negative_count")

    def test_trail_is_bounded(self):
        counter = make_counter()
        for index, center_x in enumerate((30, 32, 34, 36, 38)):
            counter.update(
                [detection(center_x, track_id=1)],
                (100, 100, 3),
                index * 0.1,
                0,
            )

        self.assertEqual(len(counter.track_visuals()[0]["trail"]), 3)

    def test_reports_missing_line(self):
        counter = make_counter()
        counter.line_points = None
        events, status, people = counter.update(
            [detection(35, track_id=1)], (100, 100, 3), 0.0, 0
        )

        self.assertEqual(events, [])
        self.assertEqual(status, "line_not_configured")
        self.assertEqual(len(people), 1)


if __name__ == "__main__":
    unittest.main()
