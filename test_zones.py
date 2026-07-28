import unittest
from dataclasses import dataclass

from counting.zones import ZoneCounter


@dataclass(frozen=True)
class Detection:
    box: tuple
    conf: float = 0.9

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def make_counter():
    return ZoneCounter(
        "top",
        {
            "counter": {
                "min_person_area_ratio": 0.0,
                "lost_timeout_sec": 1.0,
                "event_cooldown_sec": 0.1,
            },
            "zones": {
                "zone_point_y_ratio": 0.5,
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


if __name__ == "__main__":
    unittest.main()
