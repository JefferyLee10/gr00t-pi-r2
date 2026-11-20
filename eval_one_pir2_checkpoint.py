#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        required=True,
    )

    ap.add_argument(
        "--outdir",
        required=True,
    )

    # =========================================================
    # IMPORTANT:
    # Validation must use the COMPLETE exact-22D dataset,
    # not the training-only view.
    # =========================================================
    ap.add_argument(
        "--dataset",
        default=(
            "/path/to/lab-home/Isaac-GR00T/"
            "local_datasets/"
            "bag_transfer_0819_v2_exact_22d"
        ),
    )

    # Validation protocol is stored separately from dataset view.
    ap.add_argument(
        "--split-json",
        default=(
            "/path/to/lab-home/Isaac-GR00T/"
            "experiments/"
            "bag_transfer_22d_rgb_abs_exact/"
            "protocol/split.json"
        ),
    )

    ap.add_argument(
        "--eval-script",
        default=(
            "/path/to/lab-home/Isaac-GR00T/"
            "diagnostics/"
            "eval_22d_full_horizon_val.py"
        ),
    )

    ap.add_argument(
        "--denoise",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--quick",
        action="store_true",
    )

    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    dataset = Path(args.dataset).resolve()
    split_json = Path(args.split_json).resolve()
    outdir = Path(args.outdir).resolve()
    eval_script = Path(args.eval_script).resolve()

    # =========================================================
    # Basic validation
    # =========================================================

    if not checkpoint.is_dir():
        raise RuntimeError(
            f"checkpoint does not exist: {checkpoint}"
        )

    if not dataset.is_dir():
        raise RuntimeError(
            f"dataset does not exist: {dataset}"
        )

    if not split_json.is_file():
        raise RuntimeError(
            f"split.json does not exist: {split_json}"
        )

    if not eval_script.is_file():
        raise RuntimeError(
            f"evaluator does not exist: {eval_script}"
        )

    # =========================================================
    # Verify exact validation protocol
    # =========================================================

    with split_json.open(
        "r",
        encoding="utf-8",
    ) as f:
        split = json.load(f)

    val_eps = [
        int(x)
        for x in split["val_episode_indices"]
    ]

    expected_val = [
        6, 7, 8, 22, 23, 26, 28,
        35, 55, 57, 62, 70, 108, 139,
    ]

    if val_eps != expected_val:
        raise RuntimeError(
            "Unexpected validation split.\n"
            f"expected={expected_val}\n"
            f"actual  ={val_eps}"
        )

    # Make sure every validation episode is physically available.
    for ep in val_eps:

        parquet = (
            dataset
            / "data"
            / "chunk-000"
            / f"episode_{ep:06d}.parquet"
        )

        if not parquet.is_file():
            raise RuntimeError(
                f"validation parquet missing: {parquet}"
            )

        for camera in (
            "head_rgb",
            "left_wrist_rgb",
            "right_wrist_rgb",
        ):
            video = (
                dataset
                / "videos"
                / "chunk-000"
                / f"observation.images.{camera}"
                / f"episode_{ep:06d}.mp4"
            )

            if not video.is_file():
                raise RuntimeError(
                    f"validation video missing: {video}"
                )

    print("=" * 100)
    print("πR² CHECKPOINT VALIDATION")
    print("=" * 100)
    print("checkpoint =", checkpoint)
    print("dataset    =", dataset)
    print("split      =", split_json)
    print("val eps    =", val_eps)
    print("outdir     =", outdir)
    print("denoise    =", args.denoise)
    print("quick      =", args.quick)
    print("=" * 100)

    # =========================================================
    # Load existing trusted evaluator
    # =========================================================

    spec = importlib.util.spec_from_file_location(
        "eval_22d_full_horizon_val_pir2",
        str(eval_script),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot import evaluator: {eval_script}"
        )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # =========================================================
    # Override experiment-specific globals
    # =========================================================

    mod.CHECKPOINT = checkpoint
    mod.DATASET = dataset
    mod.SPLIT_JSON = split_json
    mod.OUTDIR = outdir

    mod.DENOISING_STEPS = args.denoise

    # CUDA_VISIBLE_DEVICES decides the physical GPU.
    mod.DEVICE = "cuda:0"

    # Prevent evaluator argparse from parsing wrapper args.
    sys.argv = [str(eval_script)]

    if args.quick:
        sys.argv.append("--quick")

    mod.main()


if __name__ == "__main__":
    main()
