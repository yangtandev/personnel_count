def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _head_person_score(head, person_box):
    x1, y1, x2, y2 = person_box
    person_w = max(1, x2 - x1)
    person_h = max(1, y2 - y1)
    hx1, hy1, hx2, hy2 = head["box"]
    head_w = max(1, hx2 - hx1)
    head_h = max(1, hy2 - hy1)
    if head["source"] == "face" and (head_w > person_w * 0.75 or head_h > person_h * 0.45):
        return None
    cx, cy = _box_center(head["box"])
    if not (x1 - 20 <= cx <= x2 + 20 and y1 - 10 <= cy <= y1 + person_h * 0.55):
        return None
    source_score = 2.0 if head["source"] == "head" else 0.0
    expected_x = (x1 + x2) / 2
    expected_y = y1 + person_h * 0.15
    position_penalty = abs(cx - expected_x) / person_w + abs(cy - expected_y) / person_h
    return source_score + float(head["conf"]) - position_penalty


def match_heads_to_people(heads, person_boxes):
    """Assign each head to at most one person, including overlapping person boxes."""
    candidates = []
    for head_index, head in enumerate(heads):
        for person_index, person_box in enumerate(person_boxes):
            score = _head_person_score(head, person_box)
            if score is not None:
                candidates.append((score, head_index, person_index))

    matched_heads = set()
    matched_people = set()
    result = [None] * len(person_boxes)
    for _, head_index, person_index in sorted(candidates, reverse=True):
        if head_index in matched_heads or person_index in matched_people:
            continue
        matched_heads.add(head_index)
        matched_people.add(person_index)
        result[person_index] = heads[head_index]
    return result
