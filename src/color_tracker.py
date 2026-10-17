import cv2
import numpy as np


class ColorTarget:
    """A thing to find: a name, one or more HSV ranges, and a draw color."""

    def __init__(self, name, ranges, draw_color):
        self.name = name
        # ranges is a list of (lower_hsv, upper_hsv) pairs
        self.ranges = [(np.array(lo), np.array(hi)) for lo, hi in ranges]
        self.draw_color = draw_color  # BGR, used only for drawing


class ColorTracker:
    def __init__(self, targets, min_area=400):
        self.targets = targets
        self.min_area = min_area  # ignore blobs smaller than this (noise)
        self.kernel = np.ones((5, 5), np.uint8)
        self.last_hsv = None  # kept so main.py can inspect pixel colors

    def update(self, frame):
        """Returns {name: {"center": (x, y), "bbox": (x, y, w, h)} or None}."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        self.last_hsv = hsv

        results = {}
        for target in self.targets:
            # Build a black/white mask: white where the color matches
            mask = None
            for lo, hi in target.ranges:
                part = cv2.inRange(hsv, lo, hi)
                mask = part if mask is None else cv2.bitwise_or(mask, part)

            # Clean up speckles and small holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

            # Keep only the biggest blob
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            best = max(contours, key=cv2.contourArea, default=None)
            if best is None or cv2.contourArea(best) < self.min_area:
                results[target.name] = None
                continue

            x, y, w, h = cv2.boundingRect(best)
            results[target.name] = {
                "center": (x + w // 2, y + h // 2),
                "bbox": (x, y, w, h),
            }
        return results