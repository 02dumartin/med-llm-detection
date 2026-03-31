#!/usr/bin/env python3
"""
FGART YOLOv12l 학습 스크립트 (4cls / 1cls 공용)

사용법:
    # 4cls 학습
    python scripts/fgart_yolo12.py --variant 4cls --device 0

    # 1cls 학습
    python scripts/fgart_yolo12.py --variant 1cls --device 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
DATA_ROOT_BASE = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.yolo.training import (
    save_eval_outputs,
    start_tensorboard,
    stop_process,
    write_data_yaml as write_yolo_data_yaml,
)

VARIANT_CFG = {
    "4cls": {
        "data_root": DATA_ROOT_BASE / "FGART_yolo_4cls",
        "names": ["MA", "HE", "EX", "SE"],
        "yaml": "fgart_4cls.yaml",
        "default_name": "yolo12",
    },
    "1cls": {
        "data_root": DATA_ROOT_BASE / "FGART_yolo_1cls",
        "names": ["lesion"],
        "yaml": "fgart_1cls.yaml",
        "default_name": "yolo12_1cls",
    },
}


def write_data_yaml(out_path: Path, data_root: Path, names: list[str], val_folder: str = "val") -> None:
    write_yolo_data_yaml(out_path, data_root, names, val_folder=val_folder)


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
        # optimizer
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=0.01,
        momentum=args.momentum,
        weight_decay=0.0005,
        cos_lr=True,
        # 학습 효율
        amp=True,
        cache="disk",
        patience=50,
        # 작은 객체용 증강 (MA 등)
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
        save_eval_outputs(
            project_dir=args.project,
            results_dir=args.results_dir,
            run_name=args.name,
            data_yaml=data_yaml,
            eval_split=args.eval_split,
            iou=args.iou,
            metrics_mode="multi",
            predict_after=args.predict_after,
            data_root=args.data_root,
            conf=args.conf,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FGART YOLOv12l training launcher (4cls / 1cls)")
    parser.add_argument("--variant", choices=["4cls", "1cls"], default="4cls")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "weights" / "yolov12l.pt"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default=str(PROJECT_ROOT / "runs" / "fgart"))
    parser.add_argument("--name", type=str, default=None, help="미지정 시 variant에 따라 자동 설정")
    parser.add_argument("--results-dir", type=str, default=str(PROJECT_ROOT / "results" / "fgart"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.0015)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--conf", type=float, default=0.25, help="predict용; val은 기본 0.001")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--predict-after", action="store_true")
    args = parser.parse_args()

    cfg = VARIANT_CFG[args.variant]
    data_root = cfg["data_root"]
    names = cfg["names"]
    yaml_name = cfg["yaml"]

    args.data_root = str(data_root)
    if not data_root.exists():
        raise SystemExit(f"data root not found: {data_root}")

    if args.name is None:
        args.name = cfg["default_name"]

    data_yaml = data_root / yaml_name
    write_data_yaml(data_yaml, data_root, names)

    args.project = str(Path(args.project).resolve())
    args.results_dir = str(Path(args.results_dir).resolve())

    tb_proc = None
    if args.tensorboard:
        tb_proc = start_tensorboard(args.project)

    try:
        run_ultralytics(args, data_yaml)
    finally:
        stop_process(tb_proc)


if __name__ == "__main__":
    main()
