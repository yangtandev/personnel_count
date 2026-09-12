import unittest
from dataclasses import dataclass

from counting.lines import LineCounter
from counting.calibration import suggest_counting_line
from detection.head_assignment import match_heads_to_people


@dataclass(frozen=True)
class Detection:
    box: tuple
    conf: float = 0.9
    track_id: int = None
    point: tuple = None
    point_source: str = "person"

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def detection(center_x, center_y=50, track_id=None):
    return Detection((center_x - 5, center_y - 10, center_x + 5, center_y + 10), track_id=track_id)


def make_counter(flow_direction_lock=False, line=None, crossing=None, reverse_anchor_sec=0.0):
    crossing_config = {
        "lines": {"top": line or [(0.5, 0.0), (0.5, 1.0)]},
        "point_y_ratio": 0.5,
        "hysteresis_ratio": 0.02,
        "segment_margin_ratio": 0.0,
        "min_track_frames": 1,
        "trail_length": 3,
        "point_filter_window": 1,
        "side_confirm_frames": 1,
        "side_confirm_sec": 0.0,
        "confirmation_distance_ratio": 0.0,
        "motion_confirmation_distance_ratio": 10.0,
        "handoff_timeout_sec": 1.0,
        "association_max_distance_ratio": 2.0,
        "unambiguous_association_max_distance_ratio": 2.0,
        "same_id_bonus": 0.2,
    }
    crossing_config.update(crossing or {})
    return LineCounter(
        "top",
        {
            "counter": {
                "min_person_area_ratio": 0.0,
                "lost_timeout_sec": 1.0,
                "event_cooldown_sec": 0.1,
                "one_event_per_track": False,
                "flow_direction_lock": flow_direction_lock,
                "reverse_anchor_sec": reverse_anchor_sec,
            },
            "crossing": crossing_config,
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
        counter.update(
            [detection(45, track_id=1), detection(55, track_id=2)],
            (100, 100, 3),
            0.1,
            5,
        )
        events, _, _ = counter.update(
            [detection(55, track_id=1), detection(45, track_id=2)],
            (100, 100, 3),
            0.2,
            5,
        )

        self.assertEqual([event.event for event in events], ["enter", "exit"])
        self.assertEqual([event.count_after for event in events], [6, 5])

    def test_same_direction_is_not_counted_twice_without_opposite_event(self):
        counter = make_counter(
            line=[(50, 20), (50, 80)],
            crossing={"association_max_distance_ratio": 10.0},
        )
        counter.update([detection(35, 50, track_id=1)], (100, 100, 3), 0.0, 0)
        events, _, _ = counter.update(
            [detection(65, 50, track_id=1)], (100, 100, 3), 0.2, 0
        )
        self.assertEqual([event.event for event in events], ["enter"])

        counter.update([detection(65, 95, track_id=1)], (100, 100, 3), 0.4, 1)
        counter.update([detection(35, 95, track_id=1)], (100, 100, 3), 0.6, 1)
        counter.update([detection(35, 50, track_id=1)], (100, 100, 3), 0.8, 1)
        events, status, _ = counter.update(
            [detection(65, 50, track_id=1)], (100, 100, 3), 1.0, 1
        )

        self.assertEqual(events, [])
        self.assertEqual(status, "duplicate_direction_ignored")

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
        counter.update(
            [detection(45, track_id=1), detection(55, track_id=2)],
            (100, 100, 3),
            0.1,
            5,
        )
        events, status, _ = counter.update(
            [detection(55, track_id=1), detection(45, track_id=2)],
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

    def test_requires_stable_frames_on_both_sides(self):
        counter = make_counter(crossing={"side_confirm_frames": 3})
        count = 0
        for index, center_x in enumerate((35, 36, 37, 65, 36)):
            events, _, _ = counter.update(
                [detection(center_x, track_id=1)], (100, 100, 3), index * 0.1, count
            )
            self.assertEqual(events, [])
        for index, center_x in enumerate((65, 66, 67), start=5):
            events, _, _ = counter.update(
                [detection(center_x, track_id=1)], (100, 100, 3), index * 0.1, count
            )
        self.assertEqual([event.event for event in events], ["enter"])

    def test_median_filter_ignores_one_frame_outlier(self):
        counter = make_counter(
            crossing={"point_filter_window": 5, "side_confirm_frames": 2}
        )
        for index, center_x in enumerate((35, 35, 35, 35, 35, 65, 35)):
            events, _, _ = counter.update(
                [detection(center_x, track_id=1)], (100, 100, 3), index * 0.1, 0
            )
            self.assertEqual(events, [])

    def test_one_frame_outlier_does_not_create_event(self):
        counter = make_counter(
            crossing={"point_filter_window": 5, "side_confirm_frames": 2},
        )
        for index, center_x in enumerate((35, 35, 35, 35, 65, 35)):
            events, _, _ = counter.update(
                [detection(center_x, track_id=1)], (100, 100, 3), index * 0.1, 0
            )
            self.assertEqual(events, [])

    def test_consistent_motion_confirms_crossing_before_track_disappears(self):
        counter = make_counter(
            crossing={
                "point_filter_window": 5,
                "side_confirm_frames": 3,
                "confirmation_distance_ratio": 0.1,
                "motion_confirmation_distance_ratio": 0.05,
            },
        )
        events = []
        count = 0
        for index, center_x in enumerate((30, 34, 38, 42, 46, 56)):
            events, _, _ = counter.update(
                [detection(center_x, track_id=1)], (100, 100, 3), index * 0.1, count
            )
            if events:
                count = events[-1].count_after

        self.assertEqual([event.event for event in events], ["enter"])

    def test_fast_crossing_rejects_point_source_change(self):
        counter = make_counter(
            crossing={
                "point_filter_window": 5,
                "side_confirm_frames": 3,
                "motion_confirmation_distance_ratio": 0.05,
            },
        )
        for index, center_x in enumerate((30, 34, 38, 42, 46)):
            counter.update(
                [detection(center_x, track_id=1)], (100, 100, 3), index * 0.1, 0
            )
        changed = Detection((51, 40, 61, 60), track_id=1, point_source="head")
        events, _, _ = counter.update([changed], (100, 100, 3), 0.5, 0)

        self.assertEqual(events, [])

    def test_handoff_keeps_crossing_state_when_tracker_id_changes(self):
        counter = make_counter()
        counter.update([detection(35, track_id=1)], (100, 100, 3), 0.0, 0)
        events, _, _ = counter.update(
            [detection(65, track_id=2)], (100, 100, 3), 0.1, 0
        )

        self.assertEqual([event.event for event in events], ["enter"])
        self.assertEqual(events[0].track_id, 2)
        self.assertEqual(events[0].stable_id, 1)

    def test_association_distance_scales_with_person_height(self):
        counter = make_counter(
            crossing={
                "association_max_distance_ratio": 0.75,
                "unambiguous_association_max_distance_ratio": 0.75,
            }
        )
        tall_left = Detection((10, 0, 60, 100), track_id=1)
        tall_right = Detection((40, 0, 90, 100), track_id=2)
        counter.update([tall_left], (100, 100, 3), 0.0, 0)
        counter.update([tall_right], (100, 100, 3), 0.1, 0)

        self.assertEqual(len(counter.tracks), 1)

        counter = make_counter(
            crossing={
                "association_max_distance_ratio": 0.75,
                "unambiguous_association_max_distance_ratio": 0.75,
            }
        )
        small_left = Detection((10, 40, 20, 60), track_id=1)
        small_right = Detection((40, 40, 50, 60), track_id=2)
        counter.update([small_left], (100, 100, 3), 0.0, 0)
        counter.update([small_right], (100, 100, 3), 0.1, 0)

        self.assertEqual(len(counter.tracks), 2)

    def test_same_raw_id_large_move_keeps_stable_uid(self):
        counter = make_counter(
            crossing={
                "association_max_distance_ratio": 0.25,
                "unambiguous_association_max_distance_ratio": 0.75,
            }
        )
        counter.update([detection(30, track_id=1)], (100, 100, 3), 0.0, 0)
        counter.update([detection(40, track_id=1)], (100, 100, 3), 0.1, 0)

        self.assertEqual(len(counter.tracks), 1)
        self.assertEqual(counter.track_visuals()[0]["stable_id"], 1)

    def test_large_move_with_new_raw_id_does_not_steal_stable_uid(self):
        counter = make_counter(
            crossing={
                "association_max_distance_ratio": 0.25,
                "unambiguous_association_max_distance_ratio": 0.75,
            }
        )
        counter.update([detection(30, track_id=1)], (100, 100, 3), 0.0, 0)
        counter.update([detection(40, track_id=2)], (100, 100, 3), 0.1, 0)

        self.assertEqual(len(counter.tracks), 2)

    def test_ambiguous_large_moves_do_not_guess_handoffs(self):
        counter = make_counter(
            crossing={
                "association_max_distance_ratio": 0.25,
                "unambiguous_association_max_distance_ratio": 0.75,
            }
        )
        counter.update(
            [detection(30, track_id=1), detection(50, track_id=2)],
            (100, 100, 3), 0.0, 0,
        )
        counter.update(
            [detection(40, track_id=3), detection(42, track_id=4)],
            (100, 100, 3), 0.1, 0,
        )

        self.assertEqual(len(counter.tracks), 4)

    def test_sustained_crossing_after_outlier_is_counted(self):
        counter = make_counter(
            crossing={"point_filter_window": 5, "side_confirm_frames": 2},
        )
        count = 0
        for index, center_x in enumerate((35, 35, 35, 35, 65, 35, 65, 65, 65, 65)):
            events, _, _ = counter.update(
                [detection(center_x, track_id=1)], (100, 100, 3), index * 0.1, count
            )
            if events:
                count = events[-1].count_after

        self.assertEqual(count, 1)


class HeadAssignmentTest(unittest.TestCase):
    def test_one_head_is_never_shared_by_overlapping_people(self):
        head = {"box": (85, 50, 105, 70), "conf": 0.9, "source": "head"}
        people = [(60, 40, 120, 200), (85, 40, 145, 200)]

        matches = match_heads_to_people([head], people)

        self.assertEqual(sum(match is head for match in matches), 1)

    def test_two_heads_are_assigned_to_nearest_people(self):
        left = {"box": (65, 50, 85, 70), "conf": 0.9, "source": "head"}
        right = {"box": (115, 50, 135, 70), "conf": 0.9, "source": "head"}
        people = [(50, 40, 100, 200), (100, 40, 150, 200)]

        matches = match_heads_to_people([right, left], people)

        self.assertIs(matches[0], left)
        self.assertIs(matches[1], right)


class CalibrationTest(unittest.TestCase):
    def test_suggested_line_is_perpendicular_to_observed_flow(self):
        trajectories = [
            [(20, 10), (20, 30), (20, 60), (20, 90)],
            [(80, 90), (80, 60), (80, 30), (80, 10)],
        ]

        start, end = suggest_counting_line(trajectories, 100, 100)

        self.assertGreater(abs(end[0] - start[0]), abs(end[1] - start[1]) * 3)
        self.assertTrue(20 <= (start[1] + end[1]) / 2 <= 80)


if __name__ == "__main__":
    unittest.main()
