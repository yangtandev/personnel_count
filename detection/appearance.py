"""Small, privacy-preserving appearance descriptors for short tracker handoffs."""

import cv2
import numpy as np


def appearance_descriptor(frame, box):
    """Return an L2-normalized HSV body-colour histogram, or ``None`` when unusable.

    This intentionally describes clothing colours only.  It is not a face embedding and
    is kept only in the in-memory, short-lived track state.
    """
    if frame is None or getattr(frame, "ndim", 0) != 3:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None

    # Ignore the detector-box edges and upper face/head region where background and
    # head assignment changes are most likely to contaminate the descriptor.
    inner_x1 = x1 + int((x2 - x1) * 0.15)
    inner_x2 = x2 - int((x2 - x1) * 0.15)
    inner_y1 = y1 + int((y2 - y1) * 0.25)
    inner_y2 = y1 + int((y2 - y1) * 0.90)
    crop = frame[inner_y1:inner_y2, inner_x1:inner_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 24, 16), (180, 255, 255))
    histogram = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256]).reshape(-1)
    magnitude = float(np.linalg.norm(histogram))
    if magnitude <= 1e-9:
        return None
    return tuple(float(value / magnitude) for value in histogram)
