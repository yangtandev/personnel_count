import unittest
from dataclasses import dataclass

from counting.zones import ZoneCounter


@dataclass(frozen=True)
class Detection:
    box: tuple
    conf: float = 0.9
    track_id: int = None
    point: tuple = None

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def make_counter(one_event_per_track=False, flow_direction_lock=False, use_detection_point=False):
    return ZoneCounter(
        "top",
        {
            "counter": {
                "min_person_area_ratio": 0.0,
                "lost_timeout_sec": 1.0,
                "event_cooldown_sec": 0.1,
                "one_event_per_track": one_event_per_track,
                "flow_direction_lock": flow_direction_lock,
                "reverse_anchor_sec": 0.5,
            },
            "zones": {
                "zone_point_y_ratio": 0.5,
                "use_detection_point": use_detection_point,
                "regions": {
                    "top": {
                        "A": [(0, 0), (49, 0), (49, 100), (0, 100)],
                        "B": [(50, 0), (100, 0), (100, 100), (50, 100)],
                    }
                },
            },
            "direction": {"top": {"A_to_B": "enter", "B_to_A": "exit"}},
        },
    )


class ZoneCounterTest(unittest.TestCase):
    def test_ignores_people_outside_ab_zones(self):
        counter = make_counter()

        events, status, people = counter.update(
            [Detection((110, 10, 130, 30)), Detection((10, 10, 30, 30))],
            (100, 100, 3),
            0.0,
            0,
        )

        self.assertEqual(events, [])
        self.assertEqual(status, "seen_A")
        self.assertEqual(people, [Detection((10, 10, 30, 30))])

        events, status, people = counter.update(
            [Detection((110, 10, 130, 30)), Detection((60, 10, 80, 30))],
            (100, 100, 3),
            0.2,
            0,
        )

        self.assertEqual([event.event for event in events], ["enter"])
        self.assertEqual(status, "counted_enter")
        self.assertEqual(people, [Detection((60, 10, 80, 30))])

    def test_tracks_two_people_in_ab_zones_independently(self):
        counter = make_counter()

        counter.update(
            [Detection((10, 10, 30, 30)), Detection((10, 60, 30, 80))],
            (100, 100, 3),
            0.0,
            0,
        )
        events, status, people = counter.update(
            [Detection((60, 10, 80, 30)), Detection((60, 60, 80, 80))],
            (100, 100, 3),
            0.2,
            0,
        )

        self.assertEqual([event.event for event in events], ["enter", "enter"])
        self.assertEqual([event.count_after for event in events], [1, 2])
        self.assertEqual(status, "counted_enter")
        self.assertEqual(len(people), 2)

    def test_multiple_people_do_not_pause_counting(self):
        counter = make_counter()

        counter.update(
            [Detection((10, 10, 30, 30)), Detection((10, 60, 30, 80))],
            (100, 100, 3),
            0.0,
            0,
        )

        self.assertNotEqual(counter.status, "paused_multi_person")

    def test_external_track_ids_do_not_mix_people_between_zones(self):
        counter = make_counter()

        counter.update([Detection((60, 10, 80, 30), track_id=1)], (100, 100, 3), 0.0, 0)
        events, status, people = counter.update(
            [Detection((60, 10, 80, 30), track_id=1), Detection((10, 10, 30, 30), track_id=2)],
            (100, 100, 3),
            0.2,
            0,
        )

        self.assertEqual(events, [])
        self.assertIn(status, {"seen_A", "seen_B"})
        self.assertEqual(len(people), 2)

    def test_external_track_id_counts_only_same_person_transition(self):
        counter = make_counter()

        counter.update([Detection((10, 10, 30, 30), track_id=2)], (100, 100, 3), 0.0, 0)
        events, status, people = counter.update(
            [Detection((60, 10, 80, 30), track_id=2)],
            (100, 100, 3),
            0.2,
            0,
        )

        self.assertEqual([event.event for event in events], ["enter"])
        self.assertEqual(status, "counted_enter")
        self.assertEqual(people, [Detection((60, 10, 80, 30), track_id=2)])

    def test_can_use_detection_point_when_enabled(self):
        counter = make_counter(use_detection_point=True)

        events, status, people = counter.update(
            [Detection((60, 10, 80, 30), point=(20, 20))],
            (100, 100, 3),
            0.0,
            0,
        )

        self.assertEqual(events, [])
        self.assertEqual(status, "seen_A")
        self.assertEqual(len(people), 1)

    def test_counted_track_does_not_trigger_again_before_lost(self):
        counter = make_counter(one_event_per_track=True)

        counter.update([Detection((10, 10, 30, 30), track_id=2)], (100, 100, 3), 0.0, 0)
        events, _, _ = counter.update([Detection((60, 10, 80, 30), track_id=2)], (100, 100, 3), 0.2, 0)
        events_after_count, status, _ = counter.update(
            [Detection((10, 10, 30, 30), track_id=2)],
            (100, 100, 3),
            0.4,
            events[-1].count_after,
        )

        self.assertEqual([event.event for event in events], ["enter"])
        self.assertEqual(events_after_count, [])
        self.assertEqual(status, "counted_track")

    def test_flow_direction_lock_ignores_reverse_jitter(self):
        counter = make_counter(flow_direction_lock=True)

        counter.update([Detection((60, 10, 80, 30), track_id=1)], (100, 100, 3), 0.0, 3)
        exit_events, _, _ = counter.update([Detection((10, 10, 30, 30), track_id=1)], (100, 100, 3), 0.2, 3)
        reverse_events, status, _ = counter.update(
            [Detection((60, 10, 80, 30), track_id=1)],
            (100, 100, 3),
            0.4,
            exit_events[-1].count_after,
        )
        counter.update([Detection((60, 60, 80, 80), track_id=2)], (100, 100, 3), 0.6, exit_events[-1].count_after)
        next_exit_events, _, _ = counter.update(
            [Detection((10, 60, 30, 80), track_id=2)],
            (100, 100, 3),
            0.8,
            exit_events[-1].count_after,
        )

        self.assertEqual([event.event for event in exit_events], ["exit"])
        self.assertEqual(reverse_events, [])
        self.assertEqual(status, "flow_reverse_ignored")
        self.assertEqual([event.event for event in next_exit_events], ["exit"])

    def test_flow_direction_lock_can_reanchor_reused_track(self):
        counter = make_counter(flow_direction_lock=True)

        counter.update([Detection((60, 10, 80, 30), track_id=1)], (100, 100, 3), 0.0, 3)
        first_exit, _, _ = counter.update([Detection((10, 10, 30, 30), track_id=1)], (100, 100, 3), 0.2, 3)
        ignored_once, _, _ = counter.update([Detection((60, 10, 80, 30), track_id=1)], (100, 100, 3), 0.4, 2)
        ignored_twice, status, _ = counter.update([Detection((60, 10, 80, 30), track_id=1)], (100, 100, 3), 1.0, 2)
        second_exit, _, _ = counter.update([Detection((10, 10, 30, 30), track_id=1)], (100, 100, 3), 1.2, 2)

        self.assertEqual([event.event for event in first_exit], ["exit"])
        self.assertEqual(ignored_once, [])
        self.assertEqual(ignored_twice, [])
        self.assertEqual(status, "flow_reverse_ignored")
        self.assertEqual([event.event for event in second_exit], ["exit"])


if __name__ == "__main__":
    unittest.main()
