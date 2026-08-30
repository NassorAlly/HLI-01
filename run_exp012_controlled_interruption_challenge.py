"""
run_exp012_controlled_interruption_challenge.py

EXP-012:
Controlled End-to-End Detection Interruption Challenge

Research question:
Does the two-miss temporal recovery strategy improve recovery from
short, controlled visual interruptions introduced after the temporal
sequence buffer has already been established?

Experimental design:
- 40 frozen controlled recordings from EXP-011
- 4 signs: hello, no, peace, yes
- 10 recordings per sign
- 150 frames per recording
- 30 FPS
- Deterministic visual dropout beginning at frame 60
- 1-frame, 2-frame, and 3-frame interruption conditions
- Visual dropout applied before MediaPipe processing
- Same MediaPipe event stream replayed through both policies
- Policies: full_reset and two_miss_tolerance
- Same trained HLI-01 model and checkpoint
- No retraining
- No architecture or hyperparameter changes
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import statistics
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.config.settings import SEQUENCE_LENGTH
from src.inference.predictor import Predictor
from src.inference.realtime_inference import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    extract_keypoints,
    load_trained_model,
)


# ============================================================
# Configuration
# ============================================================

RECORDING_ROOT = Path(
    "exp011_controlled_recordings"
)

EXPERIMENT_ROOT = Path(
    "outputs/experiments"
)

EXPECTED_LABELS = [
    "hello",
    "no",
    "peace",
    "yes",
]

EXPECTED_RECORDINGS_PER_CLASS = 10

EXPECTED_FRAMES_PER_VIDEO = 150

POLICIES = [
    "full_reset",
    "two_miss_tolerance",
]

# Controlled visual interruption begins only after the
# 30-frame temporal sequence has already been established.
CHALLENGE_START_FRAME = 60

CHALLENGE_CONDITIONS = {
    "dropout_1_frame": {
        CHALLENGE_START_FRAME,
    },
    "dropout_2_frames": {
        CHALLENGE_START_FRAME,
        CHALLENGE_START_FRAME + 1,
    },
    "dropout_3_frames": {
        CHALLENGE_START_FRAME,
        CHALLENGE_START_FRAME + 1,
        CHALLENGE_START_FRAME + 2,
    },
}


# ============================================================
# Utility functions
# ============================================================

def sha256_file(path):
    """
    Calculate SHA-256 checksum for dataset freezing.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:

        while True:

            chunk = file_handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def safe_mean(values):
    """
    Return mean or None for empty input.
    """

    if not values:
        return None

    return float(
        statistics.mean(values)
    )


def safe_std(values):
    """
    Sample standard deviation.
    """

    if len(values) < 2:
        return 0.0

    return float(
        statistics.stdev(values)
    )


def majority_label(labels):
    """
    Return deterministic majority label.

    Ties are resolved alphabetically for reproducibility.
    """

    if not labels:
        return None

    counts = Counter(labels)

    maximum = max(
        counts.values()
    )

    winners = sorted(
        label
        for label, count
        in counts.items()
        if count == maximum
    )

    return winners[0]


# ============================================================
# Dataset validation and freezing
# ============================================================

def build_manifest():
    """
    Validate and freeze the controlled EXP-011 recording set.
    """

    manifest = []

    for label in EXPECTED_LABELS:

        label_dir = (
            RECORDING_ROOT / label
        )

        if not label_dir.exists():

            raise RuntimeError(
                f"Missing recording directory: "
                f"{label_dir}"
            )

        videos = sorted(
            label_dir.glob("*.mp4")
        )

        if (
            len(videos)
            != EXPECTED_RECORDINGS_PER_CLASS
        ):

            raise RuntimeError(
                f"{label}: expected "
                f"{EXPECTED_RECORDINGS_PER_CLASS} "
                f"videos but found "
                f"{len(videos)}."
            )

        for video_path in videos:

            cap = cv2.VideoCapture(
                str(video_path)
            )

            if not cap.isOpened():

                raise RuntimeError(
                    f"Could not open "
                    f"{video_path}"
                )

            frame_count = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            fps = float(
                cap.get(
                    cv2.CAP_PROP_FPS
                )
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            cap.release()

            if (
                frame_count
                != EXPECTED_FRAMES_PER_VIDEO
            ):

                raise RuntimeError(
                    f"{video_path}: expected "
                    f"{EXPECTED_FRAMES_PER_VIDEO} "
                    f"frames but found "
                    f"{frame_count}."
                )

            duration = (
                frame_count / fps
                if fps > 0
                else 0.0
            )

            manifest.append(
                {
                    "video": str(
                        video_path
                    ),
                    "filename":
                        video_path.name,
                    "ground_truth": label,
                    "frames": frame_count,
                    "fps": fps,
                    "duration_seconds":
                        duration,
                    "width": width,
                    "height": height,
                    "sha256":
                        sha256_file(
                            video_path
                        ),
                }
            )

    return manifest


# ============================================================
# MediaPipe event extraction
# ============================================================

def extract_detection_events(
    video_path,
    challenge_frames,
):
    """
    Process one recording once with MediaPipe under a
    deterministic controlled visual-interruption condition.

    Frames listed in challenge_frames are replaced in memory
    with blank frames before MediaPipe processing. The source
    video on disk is never modified.

    Returns a frame-level event stream containing either:
    - 63 landmark features, or
    - None for a missed landmark detection.

    The same resulting event stream is replayed through both
    buffering policies, ensuring a paired comparison.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    mp_hands = mp.solutions.hands

    events = []

    processing_times_ms = []

    # A fresh MediaPipe tracker is used for every
    # recording to avoid state leakage between videos.
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # EXP-012 controlled visual interruption.
            # The frame is replaced only in memory, before
            # MediaPipe sees it. The original recording
            # remains unchanged.
            challenge_applied = (
                frame_number
                in challenge_frames
            )

            if challenge_applied:
                frame = np.zeros_like(frame)

            image_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            start = time.perf_counter()

            results = hands.process(
                image_rgb
            )

            processing_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            processing_times_ms.append(
                processing_ms
            )

            keypoints = extract_keypoints(
                results
            )

            events.append(
                {
                    "frame_number":
                        frame_number,
                    "detected":
                        keypoints
                        is not None,
                    "keypoints":
                        keypoints,
                    "challenge_applied":
                        challenge_applied,
                }
            )

    cap.release()

    return (
        events,
        processing_times_ms,
    )


def detection_statistics(events):
    """
    Summarize naturally occurring MediaPipe interruptions.
    """

    detected_frames = sum(
        event["detected"]
        for event in events
    )

    missed_frames = (
        len(events)
        - detected_frames
    )

    miss_run_lengths = []

    current_run = 0

    for event in events:

        if not event["detected"]:

            current_run += 1

        else:

            if current_run > 0:

                miss_run_lengths.append(
                    current_run
                )

                current_run = 0

    if current_run > 0:

        miss_run_lengths.append(
            current_run
        )

    return {
        "total_frames":
            len(events),

        "detected_frames":
            detected_frames,

        "missed_frames":
            missed_frames,

        "detection_rate":
            (
                detected_frames
                / len(events)
                if events
                else 0.0
            ),

        "interruption_count":
            len(
                miss_run_lengths
            ),

        "single_miss_runs":
            sum(
                length == 1
                for length
                in miss_run_lengths
            ),

        "two_miss_runs":
            sum(
                length == 2
                for length
                in miss_run_lengths
            ),

        "three_plus_miss_runs":
            sum(
                length >= 3
                for length
                in miss_run_lengths
            ),

        "maximum_miss_run":
            (
                max(
                    miss_run_lengths
                )
                if miss_run_lengths
                else 0
            ),

        "mean_miss_run":
            (
                safe_mean(
                    miss_run_lengths
                )
                or 0.0
            ),
    }


# ============================================================
# Policy replay
# ============================================================

def replay_policy(
    events,
    policy,
    predictor,
    ground_truth,
    challenge_end_frame=None,
):
    """
    Replay one fixed MediaPipe event stream through
    one temporal buffering policy.
    """

    if policy not in POLICIES:

        raise ValueError(
            f"Unknown policy: {policy}"
        )

    sequence = []

    consecutive_misses = 0

    resets = 0

    tolerated_misses = 0

    recovery_events = 0

    acquisition_count = 0

    first_prediction_frame = None

    predictions = []

    previous_sequence_full = False

    # Number of tolerated misses waiting for
    # a successful detection recovery.
    pending_tolerated_misses = 0

    for event in events:

        frame_number = event[
            "frame_number"
        ]

        detected = event[
            "detected"
        ]

        keypoints = event[
            "keypoints"
        ]

        # ------------------------------------------------
        # Valid hand-landmark detection
        # ------------------------------------------------

        if detected:

            if (
                policy
                == "two_miss_tolerance"
                and pending_tolerated_misses
                > 0
            ):

                recovery_events += 1

            pending_tolerated_misses = 0

            consecutive_misses = 0

            sequence.append(
                keypoints
            )

            if (
                len(sequence)
                > SEQUENCE_LENGTH
            ):

                sequence.pop(0)

        # ------------------------------------------------
        # Missed detection
        # ------------------------------------------------

        else:

            if policy == "full_reset":

                if sequence:

                    resets += 1

                sequence.clear()

                consecutive_misses = 0

                pending_tolerated_misses = 0

            elif (
                policy
                == "two_miss_tolerance"
            ):

                consecutive_misses += 1

                if consecutive_misses <= 2:

                    tolerated_misses += 1

                    pending_tolerated_misses = (
                        consecutive_misses
                    )

                else:

                    if sequence:

                        resets += 1

                    sequence.clear()

                    consecutive_misses = 0

                    pending_tolerated_misses = 0

        # ------------------------------------------------
        # Sequence acquisition
        # ------------------------------------------------

        sequence_full = (
            len(sequence)
            == SEQUENCE_LENGTH
        )

        if (
            sequence_full
            and not previous_sequence_full
        ):

            acquisition_count += 1

        previous_sequence_full = (
            sequence_full
        )

        # ------------------------------------------------
        # Prediction
        #
        # Prediction is made only when a NEW valid
        # detection arrives. Missed frames do not generate
        # duplicate predictions from stale sequences.
        # ------------------------------------------------

        if (
            detected
            and sequence_full
        ):

            input_sequence = np.asarray(
                sequence,
                dtype=np.float32,
            )

            start = time.perf_counter()

            prediction = predictor.predict(
                input_sequence
            )

            inference_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            if (
                first_prediction_frame
                is None
            ):

                first_prediction_frame = (
                    frame_number
                )

            raw_label = prediction[
                "label"
            ]

            confidence = float(
                prediction[
                    "confidence"
                ]
            )

            accepted = (
                confidence
                >= CONFIDENCE_THRESHOLD
            )

            displayed_label = (
                raw_label
                if accepted
                else "Uncertain"
            )

            predictions.append(
                {
                    "frame_number":
                        frame_number,
                    "raw_label":
                        raw_label,
                    "displayed_label":
                        displayed_label,
                    "confidence":
                        confidence,
                    "accepted":
                        accepted,
                    "correct":
                        raw_label
                        == ground_truth,
                    "inference_ms":
                        inference_ms,
                }
            )

    raw_labels = [
        item["raw_label"]
        for item in predictions
    ]

    confident_labels = [
        item["raw_label"]
        for item in predictions
        if item["accepted"]
    ]

    video_raw_label = (
        majority_label(
            raw_labels
        )
    )

    video_confident_label = (
        majority_label(
            confident_labels
        )
    )

    if (
        video_confident_label
        is None
    ):

        video_confident_label = (
            "Uncertain"
        )

    prediction_count = len(
        predictions
    )

    confident_prediction_count = sum(
        item["accepted"]
        for item in predictions
    )

    correct_prediction_count = sum(
        item["correct"]
        for item in predictions
    )

    confident_correct_count = sum(
        item["correct"]
        and item["accepted"]
        for item in predictions
    )

    inference_times = [
        item["inference_ms"]
        for item in predictions
    ]

    confidences = [
        item["confidence"]
        for item in predictions
    ]

    # ------------------------------------------------
    # EXP-012 post-interruption recovery
    # ------------------------------------------------

    first_post_challenge_prediction_frame = None
    post_challenge_recovery_delay = None

    if challenge_end_frame is not None:

        post_challenge_predictions = [
            item["frame_number"]
            for item in predictions
            if item["frame_number"]
            > challenge_end_frame
        ]

        if post_challenge_predictions:

            first_post_challenge_prediction_frame = (
                min(
                    post_challenge_predictions
                )
            )

            post_challenge_recovery_delay = (
                first_post_challenge_prediction_frame
                - challenge_end_frame
            )

    return {
        "policy":
            policy,

        "resets":
            resets,

        "tolerated_misses":
            tolerated_misses,

        "recovery_events":
            recovery_events,

        "acquisition_count":
            acquisition_count,

        "first_prediction_frame":
            first_prediction_frame,

        "first_post_challenge_prediction_frame":
            first_post_challenge_prediction_frame,

        "post_challenge_recovery_delay":
            post_challenge_recovery_delay,

        "prediction_count":
            prediction_count,

        "confident_prediction_count":
            confident_prediction_count,

        "prediction_coverage":
            (
                prediction_count
                / sum(
                    event["detected"]
                    for event in events
                )
                if events
                else 0.0
            ),

        "frame_prediction_accuracy":
            (
                correct_prediction_count
                / prediction_count
                if prediction_count
                else 0.0
            ),

        "confident_prediction_accuracy":
            (
                confident_correct_count
                / confident_prediction_count
                if confident_prediction_count
                else 0.0
            ),

        "mean_confidence":
            (
                safe_mean(
                    confidences
                )
                or 0.0
            ),

        "mean_inference_ms":
            (
                safe_mean(
                    inference_times
                )
                or 0.0
            ),

        "video_raw_label":
            video_raw_label
            or "No prediction",

        "video_raw_correct":
            (
                video_raw_label
                == ground_truth
            ),

        "video_confident_label":
            video_confident_label,

        "video_confident_correct":
            (
                video_confident_label
                == ground_truth
            ),
    }


# ============================================================
# CSV utilities
# ============================================================

def write_csv(
    path,
    rows,
):
    """
    Write list of dictionaries to CSV.
    """

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:

        writer = csv.DictWriter(
            file_handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Aggregate summaries
# ============================================================

def summarize_policy(
    rows,
    policy,
):
    """
    Aggregate one policy across all recordings.
    """

    selected = [
        row
        for row in rows
        if row["policy"] == policy
    ]

    if not selected:

        return {}

    return {
        "videos":
            len(selected),

        "video_raw_accuracy":
            safe_mean(
                [
                    float(
                        row[
                            "video_raw_correct"
                        ]
                    )
                    for row in selected
                ]
            ),

        "video_confident_accuracy":
            safe_mean(
                [
                    float(
                        row[
                            "video_confident_correct"
                        ]
                    )
                    for row in selected
                ]
            ),

        "mean_resets":
            safe_mean(
                [
                    row["resets"]
                    for row in selected
                ]
            ),

        "std_resets":
            safe_std(
                [
                    row["resets"]
                    for row in selected
                ]
            ),

        "mean_prediction_count":
            safe_mean(
                [
                    row[
                        "prediction_count"
                    ]
                    for row in selected
                ]
            ),

        "std_prediction_count":
            safe_std(
                [
                    row[
                        "prediction_count"
                    ]
                    for row in selected
                ]
            ),

        "mean_prediction_coverage":
            safe_mean(
                [
                    row[
                        "prediction_coverage"
                    ]
                    for row in selected
                ]
            ),

        "mean_frame_prediction_accuracy":
            safe_mean(
                [
                    row[
                        "frame_prediction_accuracy"
                    ]
                    for row in selected
                ]
            ),

        "mean_confidence":
            safe_mean(
                [
                    row[
                        "mean_confidence"
                    ]
                    for row in selected
                ]
            ),

        "mean_recovery_events":
            safe_mean(
                [
                    row[
                        "recovery_events"
                    ]
                    for row in selected
                ]
            ),

        "mean_tolerated_misses":
            safe_mean(
                [
                    row[
                        "tolerated_misses"
                    ]
                    for row in selected
                ]
            ),

        "videos_with_prediction":
            sum(
                row[
                    "prediction_count"
                ] > 0
                for row in selected
            ),
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    print("=" * 70)
    print(
        "HLI-01 EXP-012"
    )
    print(
        "Controlled End-to-End "
        "Detection Interruption Challenge"
    )
    print("=" * 70)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        EXPERIMENT_ROOT
        / (
            f"EXP_{timestamp}_"
            f"exp012_controlled_interruption"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print()
    print(
        "Output directory:",
        output_dir,
    )

    # ----------------------------------------------------
    # Freeze recording set
    # ----------------------------------------------------

    print()
    print(
        "Validating controlled "
        "recording set..."
    )

    manifest = build_manifest()

    print(
        f"Validated recordings: "
        f"{len(manifest)}"
    )

    write_csv(
        output_dir
        / "exp012_recording_manifest.csv",
        manifest,
    )

    # ----------------------------------------------------
    # Device/model
    # ----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device,
    )

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
        "Model loaded."
    )

    # ----------------------------------------------------
    # Process videos
    # ----------------------------------------------------

    detection_rows = []

    policy_rows = []

    paired_rows = []

    total_videos = len(
        manifest
    )

    total_conditions = len(
        CHALLENGE_CONDITIONS
    )

    total_runs = (
        total_videos
        * total_conditions
    )

    run_index = 0

    for condition_name, challenge_frames in (
        CHALLENGE_CONDITIONS.items()
    ):

        print()
        print("=" * 70)
        print(
            "CONDITION:",
            condition_name,
        )
        print(
            "Injected frames:",
            sorted(
                challenge_frames
            ),
        )
        print("=" * 70)

        for index, item in enumerate(
            manifest,
            start=1,
        ):

            video_path = Path(
                item["video"]
            )

            ground_truth = item[
                "ground_truth"
            ]

            print()
            print(
                f"[{index:02d}/"
                f"{total_videos:02d}] "
                f"{video_path.name} "
                f"({ground_truth})"
            )

            # ----------------------------------------------
            # Extract MediaPipe events ONCE
            # ----------------------------------------------

            events, mediapipe_times = (
                extract_detection_events(
                    video_path,
                    challenge_frames,
                )
            )

            detection = (
                detection_statistics(
                    events
                )
            )

            challenge_end_frame = max(
                challenge_frames
            )

            challenged_events = [
                event
                for event in events
                if event[
                    "challenge_applied"
                ]
            ]

            challenge_frames_missed = sum(
                not event["detected"]
                for event
                in challenged_events
            )

            challenge_miss_rate = (
                challenge_frames_missed
                / len(challenged_events)
                if challenged_events
                else 0.0
            )

            detection_row = {
                "condition":
                    condition_name,

                "challenge_start_frame":
                    CHALLENGE_START_FRAME,

                "challenge_end_frame":
                    challenge_end_frame,

                "challenge_length":
                    len(
                        challenge_frames
                    ),

                "challenge_frames_missed":
                    challenge_frames_missed,

                "challenge_miss_rate":
                    challenge_miss_rate,

                "video":
                    str(video_path),

                "filename":
                    video_path.name,

                "ground_truth":
                    ground_truth,

                **detection,

                "mean_mediapipe_ms":
                    (
                        safe_mean(
                            mediapipe_times
                        )
                        or 0.0
                    ),
            }

            detection_rows.append(
                detection_row
            )

            print(
                "    Detection rate : "
                f"{detection['detection_rate']:.3f}"
            )

            print(
                "    Misses         : "
                f"{detection['missed_frames']}"
            )

            print(
                "    Max miss run   : "
                f"{detection['maximum_miss_run']}"
            )

            video_policy_results = {}

            # ----------------------------------------------
            # Replay exact same events through both policies
            # ----------------------------------------------

            for policy in POLICIES:

                result = replay_policy(
                    events=events,
                    policy=policy,
                    predictor=predictor,
                    ground_truth=ground_truth,
                    challenge_end_frame=(
                        challenge_end_frame
                    ),
                )

                row = {
                    "condition":
                        condition_name,

                    "challenge_start_frame":
                        CHALLENGE_START_FRAME,

                    "challenge_end_frame":
                        challenge_end_frame,

                    "challenge_length":
                        len(
                            challenge_frames
                        ),

                    "challenge_frames_missed":
                        challenge_frames_missed,

                    "video":
                        str(video_path),

                    "filename":
                        video_path.name,

                    "ground_truth":
                        ground_truth,

                    "detected_frames":
                        detection[
                            "detected_frames"
                        ],

                    "missed_frames":
                        detection[
                            "missed_frames"
                        ],

                    "detection_rate":
                        detection[
                            "detection_rate"
                        ],

                    **result,
                }

                policy_rows.append(
                    row
                )

                video_policy_results[
                    policy
                ] = result

                print(
                    f"    {policy:20s} "
                    f"resets={result['resets']:3d} "
                    f"pred={result['prediction_count']:3d} "
                    f"label="
                    f"{result['video_raw_label']:12s} "
                    f"correct="
                    f"{result['video_raw_correct']}"
                )

            full = video_policy_results[
                "full_reset"
            ]

            two = video_policy_results[
                "two_miss_tolerance"
            ]

            paired_rows.append(
                {
                    "condition":
                        condition_name,

                    "challenge_start_frame":
                        CHALLENGE_START_FRAME,

                    "challenge_end_frame":
                        challenge_end_frame,

                    "challenge_length":
                        len(
                            challenge_frames
                        ),

                    "challenge_frames_missed":
                        challenge_frames_missed,

                    "video":
                        str(video_path),

                    "filename":
                        video_path.name,

                    "ground_truth":
                        ground_truth,

                    "full_reset_resets":
                        full["resets"],

                    "two_miss_resets":
                        two["resets"],

                    "reset_reduction":
                        (
                            full["resets"]
                            - two["resets"]
                        ),

                    "full_reset_predictions":
                        full[
                            "prediction_count"
                        ],

                    "two_miss_predictions":
                        two[
                            "prediction_count"
                        ],

                    "prediction_gain":
                        (
                            two[
                                "prediction_count"
                            ]
                            - full[
                                "prediction_count"
                            ]
                        ),

                    "full_reset_coverage":
                        full[
                            "prediction_coverage"
                        ],

                    "two_miss_coverage":
                        two[
                            "prediction_coverage"
                        ],

                    "coverage_delta":
                        (
                            two[
                                "prediction_coverage"
                            ]
                            - full[
                                "prediction_coverage"
                            ]
                        ),

                    "full_reset_video_correct":
                        full[
                            "video_raw_correct"
                        ],

                    "two_miss_video_correct":
                        two[
                            "video_raw_correct"
                        ],

                    "full_reset_frame_accuracy":
                        full[
                            "frame_prediction_accuracy"
                        ],

                    "two_miss_frame_accuracy":
                        two[
                            "frame_prediction_accuracy"
                        ],

                    "frame_accuracy_delta":
                        (
                            two[
                                "frame_prediction_accuracy"
                            ]
                            - full[
                                "frame_prediction_accuracy"
                            ]
                        ),

                    "two_miss_tolerated_misses":
                        two[
                            "tolerated_misses"
                        ],

                    "two_miss_recovery_events":
                        two[
                            "recovery_events"
                        ],

                    "full_reset_recovery_frame":
                        full[
                            "first_post_challenge_prediction_frame"
                        ],

                    "two_miss_recovery_frame":
                        two[
                            "first_post_challenge_prediction_frame"
                        ],

                    "full_reset_recovery_delay":
                        full[
                            "post_challenge_recovery_delay"
                        ],

                    "two_miss_recovery_delay":
                        two[
                            "post_challenge_recovery_delay"
                        ],

                    "recovery_delay_reduction":
                        (
                            full[
                                "post_challenge_recovery_delay"
                            ]
                            - two[
                                "post_challenge_recovery_delay"
                            ]
                            if (
                                full[
                                    "post_challenge_recovery_delay"
                                ]
                                is not None
                                and two[
                                    "post_challenge_recovery_delay"
                                ]
                                is not None
                            )
                            else None
                        ),
                }
            )

    # ----------------------------------------------------
    # Save raw results
    # ----------------------------------------------------

    write_csv(
        output_dir
        / "exp012_detection_events_summary.csv",
        detection_rows,
    )

    write_csv(
        output_dir
        / "exp012_policy_results.csv",
        policy_rows,
    )

    write_csv(
        output_dir
        / "exp012_paired_deltas.csv",
        paired_rows,
    )

    # ----------------------------------------------------
    # Aggregate by challenge condition
    # ----------------------------------------------------

    condition_summaries = {}

    for condition_name, challenge_frames in (
        CHALLENGE_CONDITIONS.items()
    ):

        condition_detection_rows = [
            row
            for row in detection_rows
            if row["condition"]
            == condition_name
        ]

        condition_policy_rows = [
            row
            for row in policy_rows
            if row["condition"]
            == condition_name
        ]

        condition_paired_rows = [
            row
            for row in paired_rows
            if row["condition"]
            == condition_name
        ]

        full_summary = summarize_policy(
            condition_policy_rows,
            "full_reset",
        )

        two_summary = summarize_policy(
            condition_policy_rows,
            "two_miss_tolerance",
        )

        recovery_reductions = [
            row[
                "recovery_delay_reduction"
            ]
            for row in condition_paired_rows
            if row[
                "recovery_delay_reduction"
            ] is not None
        ]

        paired_summary = {
            "mean_reset_reduction":
                (
                    safe_mean(
                        [
                            row[
                                "reset_reduction"
                            ]
                            for row
                            in condition_paired_rows
                        ]
                    )
                    or 0.0
                ),

            "mean_prediction_gain":
                (
                    safe_mean(
                        [
                            row[
                                "prediction_gain"
                            ]
                            for row
                            in condition_paired_rows
                        ]
                    )
                    or 0.0
                ),

            "mean_coverage_delta":
                (
                    safe_mean(
                        [
                            row[
                                "coverage_delta"
                            ]
                            for row
                            in condition_paired_rows
                        ]
                    )
                    or 0.0
                ),

            "mean_frame_accuracy_delta":
                (
                    safe_mean(
                        [
                            row[
                                "frame_accuracy_delta"
                            ]
                            for row
                            in condition_paired_rows
                        ]
                    )
                    or 0.0
                ),

            "mean_recovery_delay_reduction":
                (
                    safe_mean(
                        recovery_reductions
                    )
                    or 0.0
                ),

            "videos_with_prediction_gain":
                sum(
                    row[
                        "prediction_gain"
                    ] > 0
                    for row
                    in condition_paired_rows
                ),

            "videos_with_reset_reduction":
                sum(
                    row[
                        "reset_reduction"
                    ] > 0
                    for row
                    in condition_paired_rows
                ),

            "videos_with_faster_recovery":
                sum(
                    row[
                        "recovery_delay_reduction"
                    ] is not None
                    and row[
                        "recovery_delay_reduction"
                    ] > 0
                    for row
                    in condition_paired_rows
                ),
        }

        mean_detection_rate = (
            safe_mean(
                [
                    row[
                        "detection_rate"
                    ]
                    for row
                    in condition_detection_rows
                ]
            )
            or 0.0
        )

        mean_challenge_miss_rate = (
            safe_mean(
                [
                    row[
                        "challenge_miss_rate"
                    ]
                    for row
                    in condition_detection_rows
                ]
            )
            or 0.0
        )

        total_challenge_frames_missed = sum(
            row[
                "challenge_frames_missed"
            ]
            for row
            in condition_detection_rows
        )

        condition_summaries[
            condition_name
        ] = {
            "challenge_frames":
                sorted(
                    challenge_frames
                ),

            "challenge_length":
                len(
                    challenge_frames
                ),

            "mean_detection_rate":
                mean_detection_rate,

            "mean_challenge_miss_rate":
                mean_challenge_miss_rate,

            "total_challenge_frames_missed":
                total_challenge_frames_missed,

            "full_reset":
                full_summary,

            "two_miss_tolerance":
                two_summary,

            "paired_comparison":
                paired_summary,
        }

    # ----------------------------------------------------
    # Final EXP-012 summary
    # ----------------------------------------------------

    summary = {
        "experiment":
            "EXP-012",

        "title":
            (
                "Controlled End-to-End "
                "Detection Interruption Challenge"
            ),

        "timestamp":
            timestamp,

        "device":
            str(device),

        "source_dataset":
            "EXP-011 controlled recordings",

        "recordings":
            len(manifest),

        "challenge_conditions":
            len(
                CHALLENGE_CONDITIONS
            ),

        "video_condition_runs":
            len(
                detection_rows
            ),

        "classes":
            EXPECTED_LABELS,

        "recordings_per_class":
            EXPECTED_RECORDINGS_PER_CLASS,

        "frames_per_recording":
            EXPECTED_FRAMES_PER_VIDEO,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "sequence_length":
            SEQUENCE_LENGTH,

        "challenge_start_frame":
            CHALLENGE_START_FRAME,

        "mediapipe_configuration":
            {
                "max_num_hands": 1,
                "min_detection_confidence":
                    0.7,
                "min_tracking_confidence":
                    0.7,
            },

        "methodological_controls":
            {
                "visual_dropout_before_mediapipe":
                    True,

                "source_videos_modified":
                    False,

                "same_event_stream_for_both_policies":
                    True,

                "model_retrained":
                    False,

                "architecture_changed":
                    False,

                "hyperparameters_changed":
                    False,
            },

        "conditions":
            condition_summaries,
    }

    with (
        output_dir
        / "exp012_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file_handle:

        json.dump(
            summary,
            file_handle,
            indent=2,
        )

    # ----------------------------------------------------
    # Console summary
    # ----------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EXP-012 SUMMARY"
    )
    print("=" * 70)

    print(
        "Source recordings          :",
        len(manifest),
    )

    print(
        "Challenge conditions       :",
        len(
            CHALLENGE_CONDITIONS
        ),
    )

    print(
        "Video-condition runs       :",
        len(
            detection_rows
        ),
    )

    for condition_name in (
        CHALLENGE_CONDITIONS
    ):

        condition = (
            condition_summaries[
                condition_name
            ]
        )

        full = condition[
            "full_reset"
        ]

        two = condition[
            "two_miss_tolerance"
        ]

        paired = condition[
            "paired_comparison"
        ]

        print()
        print("-" * 70)
        print(
            "CONDITION:",
            condition_name,
        )

        print(
            "Injected frames           :",
            condition[
                "challenge_frames"
            ],
        )

        print(
            "Challenge miss rate       :",
            f"{condition['mean_challenge_miss_rate']:.4f}",
        )

        print()
        print(
            "FULL RESET"
        )

        print(
            "  Video accuracy          :",
            f"{full['video_raw_accuracy']:.4f}",
        )

        print(
            "  Mean resets/video       :",
            f"{full['mean_resets']:.4f}",
        )

        print(
            "  Mean predictions/video  :",
            f"{full['mean_prediction_count']:.4f}",
        )

        print(
            "  Mean prediction coverage:",
            f"{full['mean_prediction_coverage']:.4f}",
        )

        print()
        print(
            "TWO-MISS TOLERANCE"
        )

        print(
            "  Video accuracy          :",
            f"{two['video_raw_accuracy']:.4f}",
        )

        print(
            "  Mean resets/video       :",
            f"{two['mean_resets']:.4f}",
        )

        print(
            "  Mean predictions/video  :",
            f"{two['mean_prediction_count']:.4f}",
        )

        print(
            "  Mean prediction coverage:",
            f"{two['mean_prediction_coverage']:.4f}",
        )

        print()
        print(
            "PAIRED EFFECT"
        )

        print(
            "  Mean recovery advantage :",
            (
                f"{paired['mean_recovery_delay_reduction']:+.4f} "
                "frames"
            ),
        )

        print(
            "  Mean reset reduction    :",
            f"{paired['mean_reset_reduction']:.4f}",
        )

        print(
            "  Mean prediction gain    :",
            f"{paired['mean_prediction_gain']:.4f}",
        )

        print(
            "  Mean coverage delta     :",
            f"{paired['mean_coverage_delta']:+.4f}",
        )

        print(
            "  Mean frame acc. delta   :",
            f"{paired['mean_frame_accuracy_delta']:+.4f}",
        )

        print(
            "  Videos faster recovery  :",
            (
                f"{paired['videos_with_faster_recovery']}"
                f"/{len(manifest)}"
            ),
        )

        print(
            "  Videos reset reduction  :",
            (
                f"{paired['videos_with_reset_reduction']}"
                f"/{len(manifest)}"
            ),
        )

    print()
    print("=" * 70)

    print(
        "Results saved to:"
    )

    print(
        output_dir
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
