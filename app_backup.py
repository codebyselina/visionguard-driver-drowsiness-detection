
import cv2
import mediapipe as mp
import math
import time
import threading
import os
import winsound

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = "models/face_landmarker.task"
ALARM_PATH = "static/alarm.wav"

EAR_THRESHOLD = 0.20
EYE_CLOSED_SECONDS = 1.5

MAR_THRESHOLD = 0.60
YAWN_SECONDS = 1.0

TILT_THRESHOLD = 15
HEAD_DOWN_RATIO = 0.65
SMOOTHING_FRAMES = 5

# ==================================================
# ALARM CONTROL (WINDOWS)
# ==================================================

alarm_active = threading.Event()
alarm_thread = None


def alarm_loop():
    """Repeat the WAV alarm until the warning clears."""
    while alarm_active.is_set():
        try:
            winsound.PlaySound(
                ALARM_PATH,
                winsound.SND_FILENAME
            )
        except Exception as error:
            print("Alarm error:", error)
            alarm_active.clear()
            break


def start_alarm():
    global alarm_thread

    if not os.path.isfile(ALARM_PATH):
        print("Alarm file not found:", ALARM_PATH)
        return

    if not alarm_active.is_set():
        alarm_active.set()
        alarm_thread = threading.Thread(
            target=alarm_loop,
            daemon=True
        )
        alarm_thread.start()


def stop_alarm():
    alarm_active.clear()
    # Interrupt currently playing WAV on Windows.
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass


# ==================================================
# MEDIAPIPE SETUP
# ==================================================

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

# Six landmarks per eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
NOSE_TIP = 1

# ==================================================
# HELPER FUNCTIONS
# ==================================================

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

    return (
        vertical1 + vertical2
    ) / (2.0 * horizontal)


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

    # Rough estimate; not calibrated 3D head pose.
    if nose_ratio > HEAD_DOWN_RATIO:
        status = "HEAD DOWN"
    elif tilt_angle > TILT_THRESHOLD:
        status = "HEAD TILTED"
    else:
        status = "HEAD NORMAL"

    return tilt_angle, nose_ratio, status


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
# CAMERA AND DETECTION LOOP
# ==================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    detector.close()
    raise RuntimeError(
        "Could not open webcam. Close other camera apps and try again."
    )

eye_history = deque(maxlen=SMOOTHING_FRAMES)
mar_history = deque(maxlen=SMOOTHING_FRAMES)

eyes_closed_start = None
yawn_start = None

print("VisionGuard is running. Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        height, width, _ = frame.shape

        # Modern dashboard-style header
        cv2.rectangle(
            frame, (0, 0), (width, 48),
            (35, 25, 15), -1
        )
        draw_text(
            frame, "VISIONGUARD | DRIVER MONITOR",
            (15, 32), (255, 220, 120), 0.7
        )

        if not result.face_landmarks:
            eyes_closed_start = None
            yawn_start = None
            eye_history.clear()
            mar_history.clear()
            stop_alarm()

            draw_text(
                frame, "NO FACE DETECTED",
                (20, 90), (0, 0, 255), 0.8
            )

        else:
            points = result.face_landmarks[0]

            # ------------------------------------------
            # DRAW GREEN DOTS OVER THE ENTIRE FACE
            # ------------------------------------------

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

            # Highlight eye and mouth measurement points
            for index in (
                LEFT_EYE + RIGHT_EYE +
                [UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT]
            ):
                x = int(points[index].x * width)
                y = int(points[index].y * height)

                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(
                        frame, (x, y), 3,
                        (0, 255, 255), -1,
                        cv2.LINE_AA
                    )

            # ------------------------------------------
            # EYE CLOSURE / DROWSINESS
            # ------------------------------------------

            left_ear = calculate_ear(points, LEFT_EYE)
            right_ear = calculate_ear(points, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            eye_history.append(ear)
            smooth_ear = sum(eye_history) / len(eye_history)

            now = time.monotonic()

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

            # ------------------------------------------
            # YAWNING ESTIMATE
            # ------------------------------------------

            mar = calculate_mar(points)
            mar_history.append(mar)
            smooth_mar = sum(mar_history) / len(mar_history)

            if smooth_mar > MAR_THRESHOLD:
                if yawn_start is None:
                    yawn_start = now

                yawn_duration = now - yawn_start

                if yawn_duration >= YAWN_SECONDS:
                    yawn_status = "YAWNING DETECTED"
                    yawn_color = (0, 165, 255)
                else:
                    yawn_status = "MOUTH OPEN"
                    yawn_color = (0, 200, 255)
            else:
                yawn_start = None
                yawn_status = "NO YAWN"
                yawn_color = (0, 255, 0)

            # ------------------------------------------
            # HEAD POSITION ESTIMATE
            # ------------------------------------------

            tilt_angle, nose_ratio, head_status = (
                calculate_head_position(points)
            )

            if head_status == "HEAD NORMAL":
                head_color = (0, 255, 0)
            else:
                head_color = (0, 165, 255)

            # ------------------------------------------
            # DISPLAY LIVE STATUS
            # ------------------------------------------

            draw_text(
                frame, f"EAR: {smooth_ear:.3f}",
                (20, 82), (255, 255, 0)
            )
            draw_text(
                frame, f"MAR: {smooth_mar:.3f}",
                (20, 112), (255, 255, 0)
            )
            draw_text(
                frame, eye_status,
                (20, 148), eye_color, 0.75
            )
            draw_text(
                frame, head_status,
                (20, 180), head_color, 0.75
            )
            draw_text(
                frame, yawn_status,
                (20, 212), yawn_color, 0.75
            )
            draw_text(
                frame, f"HEAD TILT: {tilt_angle:.1f} deg",
                (20, 244), (255, 255, 255), 0.6
            )

            if eye_status == "DROWSINESS WARNING!":
                cv2.rectangle(
                    frame,
                    (5, 53),
                    (width - 5, height - 5),
                    (0, 0, 255),
                    3
                )

        cv2.imshow(
            "VisionGuard - Driver Monitoring",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stop_alarm()
    camera.release()
    cv2.destroyAllWindows()
    detector.close()
    print("VisionGuard stopped.")