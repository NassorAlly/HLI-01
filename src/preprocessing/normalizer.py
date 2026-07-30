import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATA_DIR = "dataset"
SIGN_NAME = "peace"   # change this for each sign
NUM_SAMPLES = 100
SEQUENCE_LENGTH = 30

os.makedirs(f"{DATA_DIR}/{SIGN_NAME}", exist_ok=True)

cap = cv2.VideoCapture(0)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

sample_count = 0
sequence = []

while sample_count < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

    if len(keypoints) == 63:
        sequence.append(keypoints)

    if len(sequence) == SEQUENCE_LENGTH:
        np.save(
            f"{DATA_DIR}/{SIGN_NAME}/{sample_count}.npy",
            np.array(sequence)
        )
        sequence = []
        sample_count += 1
        print(f"Saved sample {sample_count}/{NUM_SAMPLES}")

    cv2.putText(frame, f"Collecting: {SIGN_NAME} {sample_count}/{NUM_SAMPLES}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
