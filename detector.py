import cv2
import numpy as np

# Optional: YOLOv8 integration (uncomment when you have model file)
# from ultralytics import YOLO
# yolo_model = YOLO("yolov8n.pt")  # or your custom trained weights

class CrowdDetector:
    """
    Skeleton for crowd detection.
    Currently: simple motion-based approximation + ROI.
    Later: plug YOLOv8 person detection for accurate counting.
    """

    def __init__(self):
        self.prev_gray = None
        # Define ROI polygon as example (full frame here).
        self.roi_mask = None

    def _ensure_roi_mask(self, frame_shape):
        if self.roi_mask is not None:
            return
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Example ROI: central rectangle. Customize as needed.
        x1, y1 = int(0.1 * w), int(0.2 * h)
        x2, y2 = int(0.9 * w), int(0.9 * h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        self.roi_mask = mask

    def process(self, frame):
        """
        Input: BGR frame.
        Output:
          - annotated_frame (for display)
          - count (approximate people count)
          - density_level ("Low"/"Medium"/"High")
          - occupancy_pct (0–100)
        """

        self._ensure_roi_mask(frame.shape)
        annotated = frame.copy()

        # Draw ROI overlay
        roi_vis = self.roi_mask.copy()
        contours, _ = cv2.findContours(roi_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (255, 255, 0), 2)

        # ----- SIMPLE PLACEHOLDER CROWD ESTIMATION -----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return annotated, 0, "Low", 0

        # Frame difference inside ROI
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        # Apply ROI mask
        diff_roi = cv2.bitwise_and(diff, diff, mask=self.roi_mask)

        # Threshold + morphology to get moving blobs (very rough)
        _, thresh = cv2.threshold(diff_roi, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours as proxy for separate moving groups
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate count based on contour area
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # ignore noise
                continue
            count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cap count and compute occupancy %
        capacity = 100  # example capacity, tune per ROI
        if count > capacity:
            count = capacity

        occupancy_pct = int(100 * count / max(1, capacity))

        if occupancy_pct < 40:
            density_level = "Low"
            color = (0, 255, 0)
        elif occupancy_pct < 80:
            density_level = "Medium"
            color = (0, 255, 255)
        else:
            density_level = "High"
            color = (0, 0, 255)

        # On-screen text overlay
        status_text = f"People ~ {count} | {density_level} ({occupancy_pct}%)"
        cv2.rectangle(annotated, (10, 10), (380, 60), (0, 0, 0), -1)
        cv2.putText(annotated, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        return annotated, int(count), density_level, occupancy_pct

        # ----- LATER: YOLOv8 PERSON DETECTION -----
        # Example (when you have GPU/CPU model working):
        # results = yolo_model(frame, verbose=False)
        # boxes = results[0].boxes
        # people_boxes = [b for b in boxes if int(b.cls[0]) == 0]  # COCO 'person'
        # count = len(people_boxes)
        # for b in people_boxes:
        #     x1, y1, x2, y2 = map(int, b.xyxy[0])
        #     cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
