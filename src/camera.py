import cv2


class Camera:
    def __init__(self, index=0, width=640, height=480, fps=60):
        self.cap = cv2.VideoCapture(index)
        # MJPG lets most webcams reach higher FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")

    def read(self):
        """Returns the next frame, or None if the read failed."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()