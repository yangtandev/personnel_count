from math import hypot

import numpy as np


def suggest_counting_line(trajectories, width, height):
    """Suggest a finite line perpendicular to the dominant observed walking axis."""
    diagonal = hypot(width, height)
    usable = []
    vectors = []
    for points in trajectories:
        if len(points) < 3:
            continue
        array = np.asarray(points, dtype=float)
        displacement = array[-1] - array[0]
        distance = float(np.linalg.norm(displacement))
        if distance < diagonal * 0.04:
            continue
        usable.append(array)
        vectors.append(displacement / distance)
    if len(usable) < 2:
        raise ValueError("need at least two moving trajectories")

    covariance = sum(np.outer(vector, vector) for vector in vectors)
    _, axes = np.linalg.eigh(covariance)
    flow_axis = axes[:, -1]
    line_axis = np.array((-flow_axis[1], flow_axis[0]))
    all_points = np.concatenate(usable)
    flow_values = all_points @ flow_axis
    candidates = np.linspace(
        np.percentile(flow_values, 25), np.percentile(flow_values, 75), 25
    )
    occupancy_radius = diagonal * 0.025
    scored = []
    for candidate in candidates:
        coverage = sum(
            float((track @ flow_axis).min()) <= candidate <= float((track @ flow_axis).max())
            for track in usable
        )
        occupancy = int(np.count_nonzero(np.abs(flow_values - candidate) <= occupancy_radius))
        scored.append((coverage, -occupancy, candidate))
    best_coverage = max(item[0] for item in scored)
    eligible = [item for item in scored if item[0] >= best_coverage * 0.8]
    _, _, level = max(eligible)

    crossing_points = []
    for track in usable:
        projected = track @ flow_axis
        for index in range(1, len(track)):
            before, after = projected[index - 1], projected[index]
            if (before - level) * (after - level) > 0 or before == after:
                continue
            fraction = (level - before) / (after - before)
            crossing_points.append(track[index - 1] + (track[index] - track[index - 1]) * fraction)
            break
    if len(crossing_points) < 2:
        crossing_points = all_points[np.abs(flow_values - level) <= occupancy_radius]
    crossing_points = np.asarray(crossing_points)
    lateral = crossing_points @ line_axis
    center = flow_axis * level + line_axis * float(np.median(lateral))
    low, high = np.percentile(lateral, (2, 98))
    padding = diagonal * 0.05
    half_length = max((high - low) / 2 + padding, diagonal * 0.15)
    center += line_axis * ((low + high) / 2 - float(np.median(lateral)))
    start = center - line_axis * half_length
    end = center + line_axis * half_length
    return [
        (round(float(np.clip(start[0], 0, width - 1))), round(float(np.clip(start[1], 0, height - 1)))),
        (round(float(np.clip(end[0], 0, width - 1))), round(float(np.clip(end[1], 0, height - 1)))),
    ]
