#!/usr/bin/env python3
"""
Merge (FGART + DDR_crop) 4cls YOLOv12l 학습 스크립트.

사용법:
    python scripts/merge_yolo12.py --device 0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# YOLO 학습은 COCO 폴더가 아니라 labels/가 포함된 YOLO 변환본을 사용해야 한다.
DEFAULT_DATA_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection/data/Merge_crop_yolo_4cls")
NAMES = ["MA", "HE", "EX", "SE"]

VAL_FOLDER = "val"


def write_data_yaml(out_path: Path, data_root: Path, names: list[str], val_folder: str) -> None:
    out_path.write_text(
        "\n".join(
            [
                f"path: {data_root}",
                "train: train/images",
                f"val: {val_folder}/images",
                "test: test_fgart/images",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_ultralytics(args: argparse.Namespace, data_yaml: Path) -> None:
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=args.exist_ok,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=0.01,
        momentum=args.momentum,
        weight_decay=0.0005,
        cos_lr=True,
        amp=True,
        cache="disk",
        patience=50,
        mosaic=0,
        close_mosaic=25,
        mixup=0.0,
        scale=0.2,
        translate=0.1,
        degrees=0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
    )

    if args.eval_after:
        best = Path(args.project) / args.name / "weights" / "best.pt"
        model = YOLO(str(best))
        eval_dir = Path(args.results_dir) / args.name
        eval_dir.mkdir(parents=True, exist_ok=True)

        metrics = model.val(
            data=str(data_yaml),
            split=args.eval_split,
            iou=args.iou,
            plots=True,
            project=str(Path(args.results_dir)),
            name=args.name,
            exist_ok=True,
        )

        cm = metrics.confusion_matrix.matrix
        nc = cm.shape[0] - 1
        tp_correct_class = cm[:nc, :nc].diagonal().sum()
        total_gt = cm[:nc, :].sum()
        total_detected = cm[:nc, :nc].sum()

        detection_acc = total_detected / total_gt if total_gt > 0 else 0
        classification_acc = tp_correct_class / total_detected if total_detected > 0 else 0
        overall_acc = tp_correct_class / total_gt if total_gt > 0 else 0

        pd.DataFrame(
            {
                "metric": ["mAP50", "mAP50-95", "detection_acc", "classification_acc", "overall_acc"],
                "value": [metrics.box.map50, metrics.box.map, detection_acc, classification_acc, overall_acc],
            }
        ).to_csv(eval_dir / "metrics.csv", index=False)

        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.eval.metrics_utils import compute_per_class_prf_ap

        names = [model.names[i] for i in sorted(model.names.keys())]
        df_prf_ap = compute_per_class_prf_ap(metrics, cm, names)
        df_prf_ap.to_csv(eval_dir / "per_class_ap.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge 4cls YOLOv12l training launcher")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "weights" / "yolov12l.pt"))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default=str(PROJECT_ROOT / "runs" / "merge_crop"))
    parser.add_argument("--name", type=str, default="yolo12")
    parser.add_argument("--results-dir", type=str, default=str(PROJECT_ROOT / "results" / "merge_crop"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.0015)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--eval-split", type=str, default="test_fgart")
    parser.add_argument("--conf", type=float, default=0.25, help="predict용; val은 기본 0.001")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise SystemExit(f"data root not found: {data_root}")

    data_yaml = data_root / "merge_4cls.yaml"
    write_data_yaml(data_yaml, data_root, NAMES, VAL_FOLDER)

    args.project = str(Path(args.project).resolve())
    args.results_dir = str(Path(args.results_dir).resolve())

    tb_proc = None
    if args.tensorboard:
        tb_proc = subprocess.Popen(
            ["tensorboard", "--logdir", args.project, "--port", "6006", "--bind_all"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)

    try:
        run_ultralytics(args, data_yaml)
    finally:
        if tb_proc is not None:
            tb_proc.terminate()


if __name__ == "__main__":
    main()
