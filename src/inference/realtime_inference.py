"""
realtime_inference.py

Real-time webcam inference pipeline for HLI-01 v0.9.0.

Pipeline:
Webcam
    -> MediaPipe hand landmarks
    -> 30-frame sequence
    -> Predictor
    -> Predicted sign + confidence
"""

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.config.settings import (
    BEST_MODEL_FILENAME,
    CHECKPOINT_DIR,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_CLASSES,
    NUM_LAYERS,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
)

from src.inference.predictor import Predictor
from src.models.lstm_model import LSTMModel
from src.training.checkpoint_manager import CheckpointManager


CLASS_NAMES = [
    "hello",
    "no",
    "peace",
    "yes",
]

CONFIDENCE_THRESHOLD = 0.60


def create_model(device):
    """
    Create the same model architecture used during training.
    """

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
    )

    model = model.to(device)

    return model


def load_trained_model(device):
    """
    Load the trained HLI-01 checkpoint.
    """

    model = create_model(
        device=device
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=CHECKPOINT_DIR
    )

    checkpoint_manager.load_model(
        model=model,
        filename=BEST_MODEL_FILENAME,
        device=device,
    )

    model.eval()

    return model


def extract_keypoints(results):
    """
    Extract 63 MediaPipe hand-landmark features.

    The current HLI-01 dataset uses one hand:
    21 landmarks x 3 coordinates = 63 features.
    """

    keypoints = []

    if results.multi_hand_landmarks:

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )

        for landmark in hand_landmarks.landmark:

            keypoints.extend(
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]
            )

    if len(keypoints) != NUM_FEATURES:

        return None

    return np.asarray(
        keypoints,
        dtype=np.float32,
    )


def main():
    """
    Run real-time sign-language inference.
    """

    print("=" * 60)
    print("HLI-01 v0.9.0 - Real-Time Inference")
    print("=" * 60)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device            :",
        device,
    )

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    print()
    print(
        "Loading trained model..."
    )

    model = load_trained_model(
        device=device
    )

    predictor = Predictor(
        model=model,
        class_names=CLASS_NAMES,
        device=device,
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------
    # MediaPipe
    # --------------------------------------------------

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # --------------------------------------------------
    # Webcam
    # --------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        hands.close()

        raise RuntimeError(
            "Could not open webcam."
        )

    sequence = []

    predicted_label = "Waiting..."
    confidence = 0.0

    print()
    print(
        "Camera started."
    )

    print(
        "Press Q to exit."
    )

    # --------------------------------------------------
    # Real-time loop
    # --------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        image_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = hands.process(
            image_rgb
        )

        # ----------------------------------------------
        # Draw detected hand landmarks
        # ----------------------------------------------

        if results.multi_hand_landmarks:

            for hand_landmarks in (
                results.multi_hand_landmarks
            ):

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        # ----------------------------------------------
        # Extract features
        # ----------------------------------------------

        keypoints = extract_keypoints(
            results
        )

        if keypoints is not None:

            sequence.append(
                keypoints
            )

            if len(sequence) > SEQUENCE_LENGTH:

                sequence.pop(0)

        else:

            sequence.clear()

        # ----------------------------------------------
        # Prediction
        # ----------------------------------------------

        if len(sequence) == SEQUENCE_LENGTH:

            input_sequence = np.asarray(
                sequence,
                dtype=np.float32,
            )

            prediction = predictor.predict(
                input_sequence
            )

            confidence = prediction[
                "confidence"
            ]

            if confidence >= CONFIDENCE_THRESHOLD:

                predicted_label = prediction[
                    "label"
                ]

            else:

                predicted_label = "Uncertain"

        # ----------------------------------------------
        # Display status
        # ----------------------------------------------

        status_text = (
            f"Sign: {predicted_label}"
        )

        confidence_text = (
            f"Confidence: {confidence:.2%}"
        )

        sequence_text = (
            f"Frames: "
            f"{len(sequence)}/{SEQUENCE_LENGTH}"
        )

        cv2.putText(
            frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            confidence_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            sequence_text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Press Q to exit",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "HLI-01 Real-Time Inference",
            frame,
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    cap.release()

    hands.close()

    cv2.destroyAllWindows()

    print()
    print(
        "Real-time inference stopped."
    )


if __name__ == "__main__":
    main()