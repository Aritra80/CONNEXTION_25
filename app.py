from flask import Flask, render_template, Response, jsonify
import cv2
from camera import Camera
from detector import CrowdDetector

app = Flask(__name__)

camera = Camera(src=0)  # 0 for default webcam; replace with RTSP URL later.
detector = CrowdDetector()

latest_metrics = {
    "count": 0,
    "density_level": "Low",
    "occupancy_pct": 0
}

def generate_stream():
    global latest_metrics

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        processed, count, level, occ = detector.process(frame)
        latest_metrics = {
            "count": int(count),
            "density_level": level,
            "occupancy_pct": int(occ)
        }

        # Encode to JPEG for HTTP streaming
        ok, buffer = cv2.imencode(".jpg", processed)
        if not ok:
            continue
        jpg_bytes = buffer.tobytes()

        # MJPEG stream format
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n\r\n")


@app.route("/")
def index():
    """
    Main dashboard page: shows live video feed and stats.
    """
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """
    HTTP MJPEG stream of processed frames with bounding boxes and overlays.
    Use <img src="/video_feed"> on the frontend.
    """
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/metrics")
def api_metrics():
    """
    Lightweight JSON API to expose current crowd metrics for the frontend.
    """
    return jsonify(latest_metrics)


if __name__ == "__main__":
    # In dev, use debug=True. For production, run with gunicorn/uwsgi.
    app.run(host="0.0.0.0", port=5000, debug=True)
