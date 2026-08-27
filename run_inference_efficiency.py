"""
run_inference_efficiency.py

EXP-010: Inference performance and computational efficiency
evaluation for HLI-01 v1.0.0.

Measures raw model and Predictor.predict() latency
on CPU and CUDA using batch size 1.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import torch

from run_input_robustness import (
    REFERENCE_CHECKPOINT,
    build_model,
)
from src.config.settings import DATASET_PATH
from src.dataset.dataset_collector import DatasetCollector
from src.inference.predictor import Predictor


WARMUP_ITERATIONS = 50
TIMED_ITERATIONS = 500

OUTPUT_ROOT = Path("outputs/experiments")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, q):
    return float(
        np.percentile(values, q)
    )


def summarize_latencies(latencies_ms):
    mean_latency = mean(latencies_ms)

    return {
        "mean_ms": mean_latency,
        "median_ms": median(latencies_ms),
        "std_ms": (
            stdev(latencies_ms)
            if len(latencies_ms) > 1
            else 0.0
        ),
        "p95_ms": percentile(
            latencies_ms, 95
        ),
        "p99_ms": percentile(
            latencies_ms, 99
        ),
        "sequences_per_second":
            1000.0 / mean_latency,
    }


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


def benchmark_raw_model(
    model,
    tensor,
    device,
):
    """
    Benchmark model forward-pass latency.
    """

    model.eval()

    with torch.no_grad():

        # Warm-up
        for _ in range(
            WARMUP_ITERATIONS
        ):
            _ = model(tensor)

        synchronize(device)

        latencies_ms = []

        # Timed inference
        for _ in range(
            TIMED_ITERATIONS
        ):
            synchronize(device)

            start = time.perf_counter_ns()

            _ = model(tensor)

            synchronize(device)

            end = time.perf_counter_ns()

            latency_ms = (
                end - start
            ) / 1_000_000.0

            latencies_ms.append(
                latency_ms
            )

    return summarize_latencies(
        latencies_ms
    )


def benchmark_predictor(
    predictor,
    sequence,
    device,
):
    """
    Benchmark end-to-end Predictor.predict() latency.
    """

    # Warm-up
    for _ in range(
        WARMUP_ITERATIONS
    ):
        _ = predictor.predict(
            sequence
        )

    synchronize(device)

    latencies_ms = []

    # Timed prediction
    for _ in range(
        TIMED_ITERATIONS
    ):
        synchronize(device)

        start = time.perf_counter_ns()

        _ = predictor.predict(
            sequence
        )

        synchronize(device)

        end = time.perf_counter_ns()

        latency_ms = (
            end - start
        ) / 1_000_000.0

        latencies_ms.append(
            latency_ms
        )

    return summarize_latencies(
        latencies_ms
    )


def benchmark_device(device_name):
    """
    Benchmark raw-model and Predictor latency
    on one execution device.
    """

    device = torch.device(
        device_name
    )

    print()
    print("=" * 60)
    print(
        "Benchmarking device:",
        device,
    )
    print("=" * 60)

    model = build_model(
        device
    )

    class_names = DatasetCollector(
        DATASET_PATH
    ).get_classes()

    predictor = Predictor(
        model=model,
        class_names=class_names,
        device=device,
    )

    sequence = np.zeros(
        (30, 63),
        dtype=np.float32,
    )

    tensor = torch.from_numpy(
        sequence
    ).unsqueeze(0).to(
        device
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            device
        )

    raw_stats = benchmark_raw_model(
        model=model,
        tensor=tensor,
        device=device,
    )

    predictor_stats = benchmark_predictor(
        predictor=predictor,
        sequence=sequence,
        device=device,
    )

    peak_memory_bytes = None

    if device.type == "cuda":
        synchronize(device)

        peak_memory_bytes = int(
            torch.cuda.max_memory_allocated(
                device
            )
        )

    print()
    print("Raw model forward")
    print(
        "  Mean latency :",
        f"{raw_stats['mean_ms']:.4f} ms",
    )
    print(
        "  Median       :",
        f"{raw_stats['median_ms']:.4f} ms",
    )
    print(
        "  P95          :",
        f"{raw_stats['p95_ms']:.4f} ms",
    )
    print(
        "  P99          :",
        f"{raw_stats['p99_ms']:.4f} ms",
    )
    print(
        "  Throughput   :",
        f"{raw_stats['sequences_per_second']:.2f} seq/s",
    )

    print()
    print("Predictor.predict()")
    print(
        "  Mean latency :",
        f"{predictor_stats['mean_ms']:.4f} ms",
    )
    print(
        "  Median       :",
        f"{predictor_stats['median_ms']:.4f} ms",
    )
    print(
        "  P95          :",
        f"{predictor_stats['p95_ms']:.4f} ms",
    )
    print(
        "  P99          :",
        f"{predictor_stats['p99_ms']:.4f} ms",
    )
    print(
        "  Throughput   :",
        f"{predictor_stats['sequences_per_second']:.2f} seq/s",
    )

    if peak_memory_bytes is not None:
        print()
        print(
            "Peak CUDA memory:",
            f"{peak_memory_bytes / (1024 ** 2):.2f} MB",
        )

    return {
        "device": str(device),
        "raw_model": raw_stats,
        "predictor": predictor_stats,
        "peak_cuda_memory_bytes":
            peak_memory_bytes,
    }


def save_results(results):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        OUTPUT_ROOT
        / (
            f"EXP_{timestamp}_"
            "exp010_inference_efficiency"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_json = (
        output_dir
        / "exp010_efficiency_summary.json"
    )

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    summary_csv = (
        output_dir
        / "exp010_efficiency_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "device",
                "path",
                "mean_ms",
                "median_ms",
                "std_ms",
                "p95_ms",
                "p99_ms",
                "sequences_per_second",
                "peak_cuda_memory_mb",
            ],
        )

        writer.writeheader()

        for device_result in results["benchmarks"]:
            for path_name in (
                "raw_model",
                "predictor",
            ):
                stats = device_result[
                    path_name
                ]

                peak_mb = None

                if (
                    device_result[
                        "peak_cuda_memory_bytes"
                    ]
                    is not None
                ):
                    peak_mb = (
                        device_result[
                            "peak_cuda_memory_bytes"
                        ]
                        / (1024 ** 2)
                    )

                writer.writerow(
                    {
                        "device":
                            device_result["device"],
                        "path":
                            path_name,
                        "mean_ms":
                            stats["mean_ms"],
                        "median_ms":
                            stats["median_ms"],
                        "std_ms":
                            stats["std_ms"],
                        "p95_ms":
                            stats["p95_ms"],
                        "p99_ms":
                            stats["p99_ms"],
                        "sequences_per_second":
                            stats[
                                "sequences_per_second"
                            ],
                        "peak_cuda_memory_mb":
                            peak_mb,
                    }
                )

    return output_dir


def main():
    print("=" * 60)
    print(
        "EXP-010 Inference Performance "
        "and Computational Efficiency"
    )
    print("=" * 60)

    checkpoint_size_bytes = (
        REFERENCE_CHECKPOINT.stat().st_size
    )

    cpu_model = build_model(
        torch.device("cpu")
    )

    parameter_info = count_parameters(
        cpu_model
    )

    print()
    print(
        "Reference checkpoint:",
        REFERENCE_CHECKPOINT,
    )
    print(
        "Checkpoint size      :",
        f"{checkpoint_size_bytes / (1024 ** 2):.2f} MB",
    )
    print(
        "Total parameters     :",
        parameter_info["total_parameters"],
    )
    print(
        "Trainable parameters :",
        parameter_info["trainable_parameters"],
    )
    print(
        "Warm-up iterations   :",
        WARMUP_ITERATIONS,
    )
    print(
        "Timed iterations     :",
        TIMED_ITERATIONS,
    )

    benchmarks = []

    benchmarks.append(
        benchmark_device("cpu")
    )

    if torch.cuda.is_available():
        benchmarks.append(
            benchmark_device("cuda")
        )

    results = {
        "experiment": "EXP-010",
        "description":
            (
                "Inference performance and "
                "computational efficiency"
            ),
        "reference_checkpoint":
            str(REFERENCE_CHECKPOINT),
        "checkpoint_size_bytes":
            checkpoint_size_bytes,
        "checkpoint_size_mb":
            checkpoint_size_bytes
            / (1024 ** 2),
        "parameters":
            parameter_info,
        "warmup_iterations":
            WARMUP_ITERATIONS,
        "timed_iterations":
            TIMED_ITERATIONS,
        "batch_size":
            1,
        "input_shape":
            [1, 30, 63],
        "benchmarks":
            benchmarks,
    }

    output_dir = save_results(
        results
    )

    print()
    print("=" * 60)
    print("EXP-010 Summary")
    print("=" * 60)

    for result in benchmarks:
        print()
        print(
            "Device:",
            result["device"],
        )
        print(
            "Raw mean latency:",
            f"{result['raw_model']['mean_ms']:.4f} ms",
        )
        print(
            "Predictor mean latency:",
            f"{result['predictor']['mean_ms']:.4f} ms",
        )

    print()
    print("EXP-010 results saved to:")
    print(output_dir)

    print()
    print("Files:")
    print(
        "  exp010_efficiency_summary.csv"
    )
    print(
        "  exp010_efficiency_summary.json"
    )
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
