"""
record_exp011_videos.py

Controlled recording utility for EXP-011:
End-to-End Real-Time Temporal Recovery Validation.

Each recording:
- belongs to one of four HLI-01 classes
- lasts exactly 5 seconds
- targets 30 FPS
- is saved separately from exploratory recordings
"""

from pathlib import Path
import time
import cv2


LABELS = ["hello", "no", "peace", "yes"]

RECORDINGS_PER_LABEL = 10
RECORDING_SECONDS = 5
FPS = 30.0

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

OUTPUT_ROOT = Path("exp011_controlled_recordings")


def next_recording_number(label_dir, label):
    numbers = []

    for path in label_dir.glob(f"{label}_*.mp4"):
        try:
            numbers.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass

    return max(numbers, default=0) + 1


def main():
    print("=" * 65)
    print("HLI-01 EXP-011 Controlled Video Recorder")
    print("=" * 65)
    print(f"Target duration : {RECORDING_SECONDS} seconds")
    print(f"Target FPS      : {FPS}")
    print(f"Clips per class : {RECORDINGS_PER_LABEL}")
    print(f"Output folder   : {OUTPUT_ROOT}")
    print("=" * 65)

    for label in LABELS:
        (OUTPUT_ROOT / label).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {width} x {height}")
    print()
    print("Controls:")
    print("  SPACE = record one 5-second clip")
    print("  N     = next sign")
    print("  P     = previous sign")
    print("  Q     = quit")
    print()

    current_label_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            break

        label = LABELS[current_label_index]
        label_dir = OUTPUT_ROOT / label

        recorded_count = len(
            list(label_dir.glob(f"{label}_*.mp4"))
        )

        display = frame.copy()

        cv2.putText(
            display,
            f"EXP-011 | Sign: {label.upper()}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"Recorded: {recorded_count}/{RECORDINGS_PER_LABEL}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "SPACE: record 5 sec | N/P: sign | Q: quit",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if recorded_count >= RECORDINGS_PER_LABEL:
            cv2.putText(
                display,
                "10 RECORDINGS COMPLETE",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        cv2.imshow("HLI-01 EXP-011 Recorder", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("n"):
            current_label_index = (
                current_label_index + 1
            ) % len(LABELS)

        elif key == ord("p"):
            current_label_index = (
                current_label_index - 1
            ) % len(LABELS)

        elif key == 32:
            if recorded_count >= RECORDINGS_PER_LABEL:
                print(
                    f"{label.upper()} already has "
                    f"{RECORDINGS_PER_LABEL} controlled clips."
                )
                continue

            number = next_recording_number(label_dir, label)

            output_path = (
                label_dir / f"{label}_{number:02d}.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                FPS,
                (width, height),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not create {output_path}"
                )

            print()
            print(
                f"Get ready: {label.upper()} "
                f"clip {number:02d}"
            )

            # Three-second preparation countdown.
            for countdown in [3, 2, 1]:
                countdown_start = time.perf_counter()

                while (
                    time.perf_counter() - countdown_start
                    < 1.0
                ):
                    ret, countdown_frame = cap.read()

                    if not ret:
                        break

                    preview = countdown_frame.copy()

                    cv2.putText(
                        preview,
                        f"Starting in {countdown}",
                        (width // 3, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        3,
                    )

                    cv2.imshow(
                        "HLI-01 EXP-011 Recorder",
                        preview,
                    )

                    cv2.waitKey(1)

            print(
                f"Recording {label.upper()}..."
            )

            target_frames = int(
                RECORDING_SECONDS * FPS
            )

            frames_written = 0

            while frames_written < target_frames:
                ret, recording_frame = cap.read()

                if not ret:
                    break

                # Save the raw camera frame.
                writer.write(recording_frame)
                frames_written += 1

                preview = recording_frame.copy()

                cv2.putText(
                    preview,
                    "RECORDING",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    preview,
                    (
                        f"{frames_written}/"
                        f"{target_frames} frames"
                    ),
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "HLI-01 EXP-011 Recorder",
                    preview,
                )

                cv2.waitKey(1)

            writer.release()

            print(
                f"Saved: {output_path} "
                f"({frames_written} frames)"
            )

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("EXP-011 recorder stopped.")


if __name__ == "__main__":
    main()
