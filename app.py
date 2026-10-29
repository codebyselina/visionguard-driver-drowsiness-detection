
import os
import cv2
import math
import time
import threading
import winsound

from collections import deque
from flask import Flask, render_template, Response, jsonify
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


app = Flask(__name__)



MODEL_PATH = "models/face_landmarker.task"
ALARM_PATH = "static/alarm.wav"

EAR_THRESHOLD = 0.20
EYE_CLOSED_SECONDS = 1.5

MAR_THRESHOLD = 0.60
YAWN_SECONDS = 1.0

TILT_THRESHOLD = 15
HEAD_DOWN_RATIO = 0.65
SMOOTHING_FRAMES = 5



status_lock = threading.Lock()

status_data = {
    "eye": "Waiting...",
    "yawn": "Waiting...",
    "head": "Waiting...",
    "overall": "System Ready",
    "ear": 0.0,
    "mar": 0.0,
    "face_detected": False
}



alarm_active = threading.Event()
camera_running = threading.Event()
camera_running.set()


def alarm_loop():
    while alarm_active.is_set():
        try:
            winsound.PlaySound(
                ALARM_PATH,
                winsound.SND_FILENAME
            )
        except Exception as error:
            print("Alarm error:", error)
            alarm_active.clear()


def start_alarm():
    if not os.path.isfile(ALARM_PATH):
        print("Alarm file not found:", ALARM_PATH)
        return

    if not alarm_active.is_set():
        alarm_active.set()

        threading.Thread(
            target=alarm_loop,
            daemon=True
        ).start()


def stop_alarm():
    alarm_active.clear()

    try:
        winsound.PlaySound(
            None,
            winsound.SND_PURGE
        )
    except Exception:
        pass



if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(
    options
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
NOSE_TIP = 1



def distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def xy(point):
    return point.x, point.y


def calculate_ear(points, indices):
    p = [xy(points[i]) for i in indices]

    vertical1 = distance(p[1], p[5])
    vertical2 = distance(p[2], p[4])
    horizontal = distance(p[0], p[3])

    if horizontal < 1e-6:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def calculate_mar(points):
    opening = distance(
        xy(points[UPPER_LIP]),
        xy(points[LOWER_LIP])
    )

    mouth_width = distance(
        xy(points[MOUTH_LEFT]),
        xy(points[MOUTH_RIGHT])
    )

    if mouth_width < 1e-6:
        return 0.0

    return opening / mouth_width


def calculate_head_position(points):
    left_eye = xy(points[33])
    right_eye = xy(points[263])
    nose = xy(points[NOSE_TIP])

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    tilt_angle = abs(
        math.degrees(math.atan2(dy, dx))
    )

    eye_center_y = (
        left_eye[1] + right_eye[1]
    ) / 2.0

    eye_width = distance(left_eye, right_eye)

    if eye_width < 1e-6:
        return tilt_angle, 0.0, "HEAD NORMAL"

    nose_ratio = (
        nose[1] - eye_center_y
    ) / eye_width

    if nose_ratio > HEAD_DOWN_RATIO:
        head_status = "HEAD DOWN"
    elif tilt_angle > TILT_THRESHOLD:
        head_status = "HEAD TILTED"
    else:
        head_status = "HEAD NORMAL"

    return tilt_angle, nose_ratio, head_status


def draw_text(frame, text, position, color, scale=0.65):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA
    )


# ==================================================
# CAMERA AND DETECTION
# ==================================================

camera = None
camera_lock = threading.Lock()

eye_history = deque(maxlen=SMOOTHING_FRAMES)
mar_history = deque(maxlen=SMOOTHING_FRAMES)

eyes_closed_start = None
yawn_start = None


def generate_frames():
    global camera, eyes_closed_start, yawn_start

    try:
        while camera_running.is_set():

            with camera_lock:
                if camera is None or not camera.isOpened():
                    camera = cv2.VideoCapture(0)

                current_camera = camera

            if current_camera is None or not current_camera.isOpened():
                with status_lock:
                    status_data.update({
                        "eye": "Camera unavailable",
                        "yawn": "Waiting...",
                        "head": "Waiting...",
                        "overall": "Camera unavailable",
                        "face_detected": False
                    })

                time.sleep(0.5)
                continue

            success, frame = current_camera.read()

            if not success:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)

            height, width, _ = frame.shape

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 48),
                (35, 25, 15),
                -1
            )

            draw_text(
                frame,
                "VISIONGUARD | DRIVER MONITOR",
                (15, 32),
                (255, 220, 120),
                0.7
            )

            now = time.monotonic()

            if not result.face_landmarks:
                eyes_closed_start = None
                yawn_start = None
                eye_history.clear()
                mar_history.clear()
                stop_alarm()

                eye_status = "NO FACE DETECTED"
                yawn_status = "UNKNOWN"
                head_status = "UNKNOWN"
                overall = "NO FACE DETECTED"

                draw_text(
                    frame,
                    "NO FACE DETECTED",
                    (20, 90),
                    (0, 0, 255),
                    0.8
                )

                with status_lock:
                    status_data.update({
                        "eye": eye_status,
                        "yawn": yawn_status,
                        "head": head_status,
                        "overall": overall,
                        "face_detected": False
                    })

            else:
                points = result.face_landmarks[0]

                # Draw all face landmarks
                for point in points:
                    x = int(point.x * width)
                    y = int(point.y * height)

                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(
                            frame,
                            (x, y),
                            1,
                            (0, 255, 0),
                            -1,
                            cv2.LINE_AA
                        )

                # Highlight eye and mouth points
                for index in (
                    LEFT_EYE + RIGHT_EYE +
                    [UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT]
                ):
                    x = int(points[index].x * width)
                    y = int(points[index].y * height)

                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(
                            frame,
                            (x, y),
                            3,
                            (0, 255, 255),
                            -1,
                            cv2.LINE_AA
                        )

                # Eye closure
                left_ear = calculate_ear(points, LEFT_EYE)
                right_ear = calculate_ear(points, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                eye_history.append(ear)
                smooth_ear = sum(eye_history) / len(eye_history)

                if smooth_ear < EAR_THRESHOLD:
                    if eyes_closed_start is None:
                        eyes_closed_start = now

                    closed_duration = now - eyes_closed_start

                    if closed_duration >= EYE_CLOSED_SECONDS:
                        eye_status = "DROWSINESS WARNING!"
                        eye_color = (0, 0, 255)
                        start_alarm()
                    else:
                        eye_status = "EYES CLOSED"
                        eye_color = (0, 165, 255)
                else:
                    eyes_closed_start = None
                    eye_status = "EYES OPEN"
                    eye_color = (0, 255, 0)
                    stop_alarm()

                # Yawning estimate
                mar = calculate_mar(points)
                mar_history.append(mar)
                smooth_mar = sum(mar_history) / len(mar_history)

                if smooth_mar > MAR_THRESHOLD:
                    if yawn_start is None:
                        yawn_start = now

                    if now - yawn_start >= YAWN_SECONDS:
                        yawn_status = "YAWNING DETECTED"
                        yawn_color = (0, 165, 255)
                    else:
                        yawn_status = "MOUTH OPEN"
                        yawn_color = (0, 200, 255)
                else:
                    yawn_start = None
                    yawn_status = "NO YAWN"
                    yawn_color = (0, 255, 0)

                # Head position estimate
                tilt_angle, nose_ratio, head_status = (
                    calculate_head_position(points)
                )

                head_color = (
                    (0, 255, 0)
                    if head_status == "HEAD NORMAL"
                    else (0, 165, 255)
                )

                # Overall status
                if eye_status == "DROWSINESS WARNING!":
                    overall = "DROWSINESS WARNING!"
                elif yawn_status == "YAWNING DETECTED":
                    overall = "FATIGUE WARNING"
                elif head_status != "HEAD NORMAL":
                    overall = "CHECK HEAD POSITION"
                else:
                    overall = "ALERT"

                # Draw live values
                draw_text(
                    frame,
                    f"EAR: {smooth_ear:.3f}",
                    (20, 82),
                    (255, 255, 0)
                )
                draw_text(
                    frame,
                    f"MAR: {smooth_mar:.3f}",
                    (20, 112),
                    (255, 255, 0)
                )
                draw_text(
                    frame,
                    eye_status,
                    (20, 148),
                    eye_color,
                    0.75
                )
                draw_text(
                    frame,
                    head_status,
                    (20, 180),
                    head_color,
                    0.75
                )
                draw_text(
                    frame,
                    yawn_status,
                    (20, 212),
                    yawn_color,
                    0.75
                )
                draw_text(
                    frame,
                    f"HEAD TILT: {tilt_angle:.1f} deg",
                    (20, 244),
                    (255, 255, 255),
                    0.6
                )

                if eye_status == "DROWSINESS WARNING!":
                    cv2.rectangle(
                        frame,
                        (5, 53),
                        (width - 5, height - 5),
                        (0, 0, 255),
                        3
                    )

                with status_lock:
                    status_data.update({
                        "eye": eye_status,
                        "yawn": yawn_status,
                        "head": head_status,
                        "overall": overall,
                        "ear": round(smooth_ear, 3),
                        "mar": round(smooth_mar, 3),
                        "face_detected": True
                    })

            # Encode webcam frame for browser
            ok, buffer = cv2.imencode(".jpg", frame)

            if not ok:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame_bytes +
                b"\r\n"
            )

    finally:
        stop_alarm()


# ==================================================
# FLASK ROUTES
# ==================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def get_status():
    with status_lock:
        return jsonify(status_data)


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":
    try:
        print("VisionGuard dashboard starting...")
        print("Open http://127.0.0.1:5000 in your browser")

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )

    finally:
        camera_running.clear()
        stop_alarm()

        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None

        detector.close()