import cv2

class Camera:
    """
    Simple camera wrapper around OpenCV VideoCapture.
    By default uses webcam 0. Replace with RTSP/IPC URL for real deployment.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")

        # Optional: set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
