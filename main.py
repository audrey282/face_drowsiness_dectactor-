import cv2
import numpy as np
import pygame
import threading
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pygame.mixer.init()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

EYE_CLOSED_THRESHOLD = 0.22
YAWN_THRESHOLD = 0.60
DROWSY_TIME = 2.0

alarm_playing = False


def play_alarm():
    global alarm_playing

    if not alarm_playing:
        alarm_playing = True
        pygame.mixer.music.load("alarm.wav.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        alarm_playing = False


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_points, width, height):
    points = []

    for idx in eye_points:
        lm = landmarks[idx]
        points.append((int(lm.x * width), int(lm.y * height)))

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, width, height):
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    top = (int(top.x * width), int(top.y * height))
    bottom = (int(bottom.x * width), int(bottom.y * height))
    left = (int(left.x * width), int(left.y * height))
    right = (int(right.x * width), int(right.y * height))

    vertical = distance(top, bottom)
    horizontal = distance(left, right)

    return vertical / horizontal


base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
closed_start_time = None

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    status = "Normal"

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_ear = eye_aspect_ratio(
            landmarks, LEFT_EYE, width, height
        )

        right_ear = eye_aspect_ratio(
            landmarks, RIGHT_EYE, width, height
        )

        avg_ear = (left_ear + right_ear) / 2

        mar = mouth_aspect_ratio(
            landmarks, width, height
        )

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if avg_ear < EYE_CLOSED_THRESHOLD:
            if closed_start_time is None:
                closed_start_time = time.time()

            closed_duration = time.time() - closed_start_time

            if closed_duration >= DROWSY_TIME:
                status = "DROWSY - EYES CLOSED"
                threading.Thread(target=play_alarm, daemon=True).start()
        else:
            closed_start_time = None

        if mar > YAWN_THRESHOLD:
            status = "DROWSY - YAWNING"
            threading.Thread(target=play_alarm, daemon=True).start()

    else:
        status = "NO FACE DETECTED"

    color = (0, 255, 0)

    if "DROWSY" in status:
        color = (0, 0, 255)

    cv2.putText(frame, status, (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()