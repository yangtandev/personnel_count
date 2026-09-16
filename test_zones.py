import unittest
from dataclasses import dataclass

import numpy as np

from detection.appearance import appearance_descriptor
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
    appearance: tuple = None

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def detection(center_x, center_y=50, track_id=None, appearance=None, point_source="person"):
    return Detection(
        (center_x - 5, center_y - 10, center_x + 5, center_y + 10),
        track_id=track_id,
        appearance=appearance,
        point_source=point_source,
    )


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
        "appearance_memory_enabled": True,
        "appearance_weight": 2.0,
        "appearance_max_distance": 0.2,
        "appearance_handoff_timeout_sec": 2.0,
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


def make_corridor_counter(crossing=None):
    """A simple left-to-right passage used to specify corridor behaviour."""
    corridor = {
        "outside": [(0.0, 0.0), (0.35, 0.0), (0.35, 1.0), (0.0, 1.0)],
        "transit": [(0.35, 0.0), (0.65, 0.0), (0.65, 1.0), (0.35, 1.0)],
        "inside": [(0.65, 0.0), (1.0, 0.0), (1.0, 1.0), (0.65, 1.0)],
        "events": {
            "outside_to_inside": "enter",
            "inside_to_outside": "exit",
        },
    }
    return make_counter(
        crossing={
            "counting_mode": "corridor",
            "corridors": {"top": corridor},
            "zone_confirm_frames": 1,
            "zone_confirm_sec": 0.0,
            "initial_baseline_sec": 0.0,
            **(crossing or {}),
        }
    )


class LineCounterTest(unittest.TestCase):
    def test_corridor_counts_only_completed_outside_to_inside_path(self):
        counter = make_corridor_counter()
        self.assertEqual(counter.counting_mode, "corridor")
        count = 0
        for now, x in ((0.0, 20), (0.1, 50), (0.2, 80)):
            events, _, _ = counter.update([detection(x, track_id=1)], (100, 100, 3), now, count)
            if events:
                count = events[-1].count_after

        self.assertEqual(count, 1)

    def test_corridor_turnback_does_not_count(self):
        counter = make_corridor_counter()
        events = []
        for now, x in ((0.0, 20), (0.1, 50), (0.2, 20)):
            new_events, _, _ = counter.update(
                [detection(x, track_id=1)], (100, 100, 3), now, 0
            )
            events.extend(new_events)

        self.assertEqual(events, [])

    def test_corridor_rejects_reverse_before_terminal_dwell(self):
        counter = make_corridor_counter(
            {
                "corridor_reverse_min_terminal_sec": 0.5,
                "association_max_distance_ratio": 10.0,
                "unambiguous_association_max_distance_ratio": 10.0,
            }
        )
        observed = []
        for now, x in ((0.0, 20), (0.1, 50), (0.2, 80), (0.3, 50), (0.4, 20)):
            events, status, _ = counter.update([detection(x, track_id=1)], (100, 100, 3), now, 0)
            observed.extend(events)

        self.assertEqual([event.event for event in observed], ["enter"])
        self.assertEqual(status, "corridor_reverse_unsettled")

    def test_corridor_allows_reverse_after_terminal_dwell(self):
        counter = make_corridor_counter(
            {
                "corridor_reverse_min_terminal_sec": 0.5,
                "association_max_distance_ratio": 10.0,
                "unambiguous_association_max_distance_ratio": 10.0,
            }
        )
        count = 0
        observed = []
        for now, x in ((0.0, 20), (0.1, 50), (0.2, 80), (0.8, 80), (0.9, 50), (1.0, 20)):
            events, _, _ = counter.update([detection(x, track_id=1)], (100, 100, 3), now, count)
            observed.extend(events)
            if events:
                count = events[-1].count_after

        self.assertEqual([event.event for event in observed], ["enter", "exit"])
        self.assertEqual(count, 0)

    def test_corridor_tracks_simultaneous_opposite_paths_independently(self):
        counter = make_corridor_counter()
        count = 5
        frames = (
            (0.0, (20, 1), (80, 2)),
            (0.1, (50, 1), (50, 2)),
            (0.2, (80, 1), (20, 2)),
        )
        observed = []
        for now, first, second in frames:
            events, _, _ = counter.update(
                [detection(*first, track_id=first[1]), detection(*second, track_id=second[1])],
                (100, 100, 3), now, count,
            )
            observed.extend(event.event for event in events)
            if events:
                count = events[-1].count_after

        self.assertEqual(observed, ["enter", "exit"])
        self.assertEqual(count, 5)

    def test_corridor_keeps_people_separate_when_tracker_reuses_one_raw_id(self):
        """A duplicate raw ID must not steal the other person's journey."""
        counter = make_corridor_counter()
        red = (1.0, 0.0, 0.0)
        blue = (0.0, 0.0, 1.0)
        counter.update(
            [
                detection(20, track_id=1, appearance=red),
                detection(80, track_id=2, appearance=blue),
            ],
            (100, 100, 3),
            0.0,
            5,
        )
        # The tracker incorrectly gives both overlapping people raw ID 1.
        counter.update(
            [
                detection(50, track_id=1, appearance=blue),
                detection(50, track_id=1, appearance=red),
            ],
            (100, 100, 3),
            0.1,
            5,
        )
        events, _, _ = counter.update(
            [
                detection(20, track_id=1, appearance=blue),
                detection(80, track_id=1, appearance=red),
            ],
            (100, 100, 3),
            0.2,
            5,
        )

        self.assertEqual([event.event for event in events], ["exit", "enter"])
        self.assertEqual([event.stable_id for event in events], [2, 1])

    def test_corridor_counts_swept_path_when_detector_misses_transit_frame(self):
        counter = make_corridor_counter({"association_max_distance_ratio": 10.0})
        counter.update([detection(20, track_id=1)], (100, 100, 3), 0.0, 0)
        events, _, _ = counter.update([detection(80, track_id=1)], (100, 100, 3), 0.4, 0)

        self.assertEqual([event.event for event in events], ["enter"])

    def test_corridor_id_handoff_in_transit_keeps_logical_person(self):
        counter = make_corridor_counter(
            {
                "handoff_timeout_sec": 0.25,
                "appearance_handoff_timeout_sec": 2.0,
                "association_max_distance_ratio": 0.25,
                "unambiguous_association_max_distance_ratio": 0.75,
                "appearance_association_max_distance_ratio": 2.0,
            }
        )
        appearance = (1.0, 0.0, 0.0)
        counter.update([detection(20, track_id=1, appearance=appearance)], (100, 100, 3), 0.0, 0)
        counter.update([detection(50, track_id=1, appearance=appearance)], (100, 100, 3), 0.1, 0)
        events, _, _ = counter.update(
            [detection(80, track_id=9, appearance=appearance)], (100, 100, 3), 0.6, 0
        )

        self.assertEqual([event.event for event in events], ["enter"])
        self.assertEqual(events[0].stable_id, 1)

    def test_corridor_reconnects_a_lost_journey_that_reappears_in_transit(self):
        counter = make_corridor_counter(
            {
                "journey_reacquire_timeout_sec": 2.0,
                "journey_reacquire_max_distance_ratio": 20.0,
            }
        )
        appearance = (1.0, 0.0, 0.0)
        count = 0
        for now, x in ((0.0, 20), (0.1, 50), (0.2, 80)):
            events, _, _ = counter.update(
                [detection(x, track_id=1, appearance=appearance)], (100, 100, 3), now, count
            )
            if events:
                count = events[-1].count_after

        counter.update([], (100, 100, 3), 1.5, count)
        counter.update(
            [detection(50, track_id=1, appearance=appearance)], (100, 100, 3), 1.6, count
        )
        events, _, _ = counter.update(
            [detection(20, track_id=1, appearance=appearance)], (100, 100, 3), 1.7, count
        )

        self.assertEqual([event.event for event in events], ["exit"])
        self.assertEqual(events[0].stable_id, 1)

    def test_corridor_does_not_count_person_first_seen_in_transit(self):
        counter = make_corridor_counter()
        counter.update([detection(50, track_id=1)], (100, 100, 3), 0.0, 0)
        events, _, _ = counter.update([detection(80, track_id=1)], (100, 100, 3), 0.1, 0)

        self.assertEqual(events, [])

    def test_corridor_ignores_initial_person_already_inside(self):
        counter = make_corridor_counter({"initial_baseline_sec": 1.0})
        observed = []
        for now, x in ((0.0, 80), (0.1, 50), (0.2, 20), (0.3, 50), (0.4, 80)):
            events, _, _ = counter.update([detection(x, track_id=1)], (100, 100, 3), now, 0)
            observed.extend(events)

        self.assertEqual(observed, [])

    def test_corridor_counts_later_reacquisition_that_starts_in_transit(self):
        counter = make_corridor_counter(
            {"initial_baseline_sec": 1.0, "allow_transit_reentry": True}
        )
        counter.update([], (100, 100, 3), 0.0, 0)
        counter.update([detection(50, track_id=7)], (100, 100, 3), 2.0, 0)
        events, _, _ = counter.update([detection(80, track_id=7)], (100, 100, 3), 2.1, 0)

        self.assertEqual([event.event for event in events], ["enter"])

    def test_corridor_reentry_requires_meaningful_progress_from_transit(self):
        counter = make_corridor_counter(
            {
                "allow_transit_reentry": True,
                "transit_reentry_min_progress_ratio": 0.2,
            }
        )
        counter.update([], (100, 100, 3), 0.0, 0)
        counter.update([detection(62, track_id=7)], (100, 100, 3), 2.0, 0)
        events, _, _ = counter.update([detection(66, track_id=7)], (100, 100, 3), 2.1, 0)

        self.assertEqual(events, [])

    def test_corridor_reentry_counts_after_meaningful_progress_from_transit(self):
        counter = make_corridor_counter(
            {
                "allow_transit_reentry": True,
                "transit_reentry_min_progress_ratio": 0.2,
            }
        )
        counter.update([], (100, 100, 3), 0.0, 0)
        counter.update([detection(50, track_id=7)], (100, 100, 3), 2.0, 0)
        events, _, _ = counter.update([detection(80, track_id=7)], (100, 100, 3), 2.1, 0)

        self.assertEqual([event.event for event in events], ["enter"])

    def test_corridor_defers_event_until_a_handed_off_uid_is_stable(self):
        counter = make_corridor_counter({"corridor_handoff_confirm_frames": 2})
        appearance = (1.0, 0.0, 0.0)
        counter.update([detection(20, track_id=1, appearance=appearance)], (100, 100, 3), 0.0, 0)
        counter.update([detection(50, track_id=1, appearance=appearance)], (100, 100, 3), 0.1, 0)
        events, _, _ = counter.update(
            [detection(80, track_id=9, appearance=appearance)], (100, 100, 3), 0.2, 0
        )
        self.assertEqual(events, [])

        events, _, _ = counter.update(
            [detection(80, track_id=9, appearance=appearance)], (100, 100, 3), 0.3, 0
        )
        self.assertEqual([event.event for event in events], ["enter"])

    def test_corridor_ignores_only_the_point_source_switch_frame(self):
        counter = make_corridor_counter()
        counter.update([detection(20, track_id=1)], (100, 100, 3), 0.0, 0)
        counter.update([detection(50, track_id=1)], (100, 100, 3), 0.1, 0)
        events, _, _ = counter.update(
            [detection(80, track_id=1, point_source="head")], (100, 100, 3), 0.2, 0
        )
        self.assertEqual(events, [])

        events, _, _ = counter.update(
            [detection(80, track_id=1, point_source="head")], (100, 100, 3), 0.3, 0
        )
        self.assertEqual([event.event for event in events], ["enter"])

    def test_corridor_keeps_swept_path_across_point_source_switch(self):
        counter = make_corridor_counter()
        counter.update([detection(80, track_id=1)], (100, 100, 3), 0.0, 1)

        events, _, _ = counter.update(
            [detection(20, track_id=1, point_source="head")], (100, 100, 3), 0.1, 1
        )
        self.assertEqual(events, [])

        events, _, _ = counter.update(
            [detection(20, track_id=1, point_source="head")], (100, 100, 3), 0.2, 1
        )
        self.assertEqual([event.event for event in events], ["exit"])
        self.assertEqual(events[0].count_after, 0)

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

    def test_appearance_keeps_event_owner_when_nearby_people_exchange_raw_ids(self):
        """A raw-ID switch must not let the nearby person's crossing own the event."""
        counter = make_counter(
            crossing={
                "association_max_distance_ratio": 2.0,
                "unambiguous_association_max_distance_ratio": 2.0,
            }
        )
        person_a = (1.0, 0.0, 0.0)
        person_b = (0.0, 1.0, 0.0)
        counter.update(
            [
                detection(46, track_id=1, appearance=person_a),
                detection(54, track_id=2, appearance=person_b),
            ],
            (100, 100, 3),
            0.0,
            10,
        )
        events, _, _ = counter.update(
            [
                detection(54, track_id=31, appearance=person_a),
                detection(46, track_id=32, appearance=person_b),
            ],
            (100, 100, 3),
            0.1,
            10,
        )

        self.assertEqual(
            [(event.track_id, event.event) for event in events],
            [(31, "enter"), (32, "exit")],
        )

    def test_same_appearance_reconnects_after_short_gap_and_preserves_count_history(self):
        """A replacement raw ID must inherit a recent person's crossing state."""
        counter = make_counter(
            reverse_anchor_sec=0.5,
            crossing={
                "handoff_timeout_sec": 0.25,
                "appearance_handoff_timeout_sec": 2.0,
                "association_max_distance_ratio": 0.25,
                "unambiguous_association_max_distance_ratio": 0.75,
                "appearance_association_max_distance_ratio": 2.0,
            }
        )
        person_a = (1.0, 0.0, 0.0)
        count = 0
        for now, x in ((0.0, 35), (0.1, 40), (0.2, 45)):
            counter.update([detection(x, track_id=1, appearance=person_a)], (100, 100, 3), now, count)
        events, _, _ = counter.update(
            [detection(65, track_id=9, appearance=person_a)],
            (100, 100, 3),
            0.8,
            count,
        )

        self.assertEqual([event.event for event in events], ["enter"])
        self.assertEqual(events[0].stable_id, 1)
        self.assertEqual(events[0].track_id, 9)

        counter.update(
            [detection(35, track_id=21, appearance=person_a)],
            (100, 100, 3),
            0.9,
            1,
        )
        events, _, _ = counter.update(
            [detection(65, track_id=21, appearance=person_a)],
            (100, 100, 3),
            1.1,
            1,
        )

        self.assertEqual(events, [])
        self.assertEqual(len(counter.tracks), 1)
        self.assertEqual(counter.track_visuals()[0]["stable_id"], 1)

    def test_appearance_reconnects_across_a_missing_crossing_frame(self):
        """A short detector gap at the line must not lose a valid individual crossing."""
        counter = make_counter(
            crossing={
                "handoff_timeout_sec": 0.25,
                "appearance_handoff_timeout_sec": 2.0,
                "association_max_distance_ratio": 0.25,
                "unambiguous_association_max_distance_ratio": 0.75,
            }
        )
        person_a = (0.0, 0.0, 1.0)
        for now, x in ((0.0, 35), (0.1, 40), (0.2, 45)):
            counter.update([detection(x, track_id=1, appearance=person_a)], (100, 100, 3), now, 0)

        events, _, _ = counter.update(
            [detection(65, track_id=8, appearance=person_a)],
            (100, 100, 3),
            0.8,
            0,
        )

        self.assertEqual([event.event for event in events], ["enter"])
        self.assertEqual(events[0].stable_id, 1)

    def test_appearance_handoff_can_be_disabled_for_safe_rollback(self):
        counter = make_counter(
            crossing={
                "appearance_memory_enabled": False,
                "handoff_timeout_sec": 0.25,
                "appearance_handoff_timeout_sec": 2.0,
                "association_max_distance_ratio": 0.25,
                "unambiguous_association_max_distance_ratio": 0.75,
            }
        )
        person_a = (0.0, 0.0, 1.0)
        for now, x in ((0.0, 35), (0.1, 40), (0.2, 45)):
            counter.update([detection(x, track_id=1, appearance=person_a)], (100, 100, 3), now, 0)

        events, _, _ = counter.update(
            [detection(65, track_id=8, appearance=person_a)],
            (100, 100, 3),
            0.8,
            0,
        )

        self.assertEqual(events, [])
        self.assertEqual(len(counter.tracks), 2)

    def test_nearby_same_raw_id_survives_an_appearance_change(self):
        """A tight same-ID track remains continuous when its crop changes at overlap."""
        counter = make_counter(
            crossing={
                "association_max_distance_ratio": 0.5,
                "unambiguous_association_max_distance_ratio": 0.75,
            }
        )
        counter.update(
            [detection(35, track_id=1, appearance=(1.0, 0.0, 0.0))],
            (100, 100, 3),
            0.0,
            0,
        )
        counter.update(
            [detection(40, track_id=1, appearance=(0.0, 1.0, 0.0))],
            (100, 100, 3),
            0.1,
            0,
        )

        self.assertEqual(len(counter.tracks), 1)
        self.assertEqual(counter.track_visuals()[0]["stable_id"], 1)


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


class AppearanceDescriptorTest(unittest.TestCase):
    def test_body_colour_descriptor_is_stable_and_distinguishes_colours(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:80, 10:40] = (220, 30, 30)
        frame[20:80, 60:90] = (30, 30, 220)

        blue = appearance_descriptor(frame, (10, 20, 40, 80))
        blue_again = appearance_descriptor(frame, (10, 20, 40, 80))
        red = appearance_descriptor(frame, (60, 20, 90, 80))

        self.assertIsNotNone(blue)
        self.assertGreater(sum(left * right for left, right in zip(blue, blue_again)), 0.99)
        self.assertLess(sum(left * right for left, right in zip(blue, red)), 0.2)


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
