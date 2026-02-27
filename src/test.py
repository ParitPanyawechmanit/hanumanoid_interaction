import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1) Download a hand_landmarker.task model file (see MediaPipe docs) and put it next to your script.
MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = vision.MpImage(image_format=vision.ImageFormat.SRGB, data=frame_rgb)

    result = detector.detect(mp_image)

    # result.hand_landmarks is a list of hands, each is a list of 21 landmarks
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            # example: landmark 8 = index fingertip
            tip = hand[8]
            h, w = frame_bgr.shape[:2]
            u, v = int(tip.x * w), int(tip.y * h)
            cv2.circle(frame_bgr, (u, v), 6, (0, 255, 0), -1)

    cv2.imshow("hands", frame_bgr)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
