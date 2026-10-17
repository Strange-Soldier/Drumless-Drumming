import time
import cv2

from camera import Camera
from color_tracker import ColorTarget, ColorTracker

# OpenCV hue runs 0-179. Tune these for YOUR objects and lighting.
TARGETS = [
    ColorTarget("blue_book", [((100, 150, 50), (130, 255, 255))], (255, 0, 0)),
    ColorTarget(
        "red_book",
        [((0, 150, 70), (10, 255, 255)), ((170, 150, 70), (179, 255, 255))],
        (0, 0, 255),
    ),
    ColorTarget("tip_marker", [((40, 100, 100), (80, 255, 255))], (0, 255, 0)),
]

WINDOW = "Drumless Drumming"


def main():
    cam = Camera(index=0, width=640, height=480, fps=60)
    tracker = ColorTracker(TARGETS)

    def on_click(event, x, y, flags, param):
        # Debug helper: left-click prints the HSV value under the cursor
        if event == cv2.EVENT_LBUTTONDOWN and tracker.last_hsv is not None:
            h, s, v = tracker.last_hsv[y, x]
            print(f"HSV at ({x}, {y}): H={h} S={s} V={v}")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_click)

    prev = time.perf_counter()
    fps = 0.0

    while True:
        frame = cam.read()
        if frame is None:
            break

        found = tracker.update(frame)
        for target in TARGETS:
            hit = found[target.name]
            if hit is None:
                continue
            x, y, w, h = hit["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), target.draw_color, 2)
            cv2.putText(frame, target.name, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, target.draw_color, 2)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev))
        prev = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()