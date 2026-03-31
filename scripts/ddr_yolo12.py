#!/usr/bin/env python3
"""
DDR YOLOv12l 학습 스크립트 (4cls / 1cls 공용).

- DDR은 해상도가 다양함(35종). YOLO는 imgsz로 고정 후 letterbox 리사이즈하므로
  별도 처리 없이 학습 가능. (이미지마다 비율 유지 + 패딩으로 동일 크기 입력)

사용법:
    python scripts/ddr_yolo12.py --variant 4cls --device 2
    python scripts/ddr_yolo12.py --variant 1cls --device 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.yolo.training import (
    run_yolov12_repo as run_yolov12_repo_helper,
    save_eval_outputs,
    start_tensorboard,
    stop_process,
    write_data_yaml as write_yolo_data_yaml,
)

VARIANT_CFG = {
    "4cls": {
        "data_root": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_yolo_4cls"),
        "names": ["MA", "HE", "EX", "SE"],
        "yaml": "ddr_4cls.yaml",
        "default_name": "yolo12",
        "metrics_mode": "multi",
    },
    "1cls": {
        "data_root": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_yolo_1cls"),
        "names": ["lesion"],
        "yaml": "ddr_1cls.yaml",
        "default_name": "yolo12_1cls",
        "metrics_mode": "single",
    },
}

VAL_FOLDER = "valid"


def write_data_yaml(out_path: Path, data_root: Path, names: list[str], val_folder: str) -> None:
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
            metrics_mode=args.metrics_mode,
            predict_after=args.predict_after,
            data_root=args.data_root,
            conf=args.conf,
        )


def run_yolov12_repo(args: argparse.Namespace, data_yaml: Path) -> None:
    run_yolov12_repo_helper(args, data_yaml)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDR YOLOv12 training launcher (4cls / 1cls)")
    parser.add_argument("--variant", choices=["4cls", "1cls"], default="4cls")
    parser.add_argument("--backend", choices=["ultralytics", "yolov12"], default="ultralytics")
    parser.add_argument("--repo", type=str, default=None, help="YOLOv12 repo path (backend=yolov12)")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "weights" / "yolov12l.pt"))
    parser.add_argument("--imgsz", type=int, default=1920, help="DDR 해상도 다양 → letterbox로 이 크기로 통일")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default=str(PROJECT_ROOT / "runs" / "ddr"))
    parser.add_argument("--name", type=str, default=None, help="default: yolo12")
    parser.add_argument("--results-dir", type=str, default=str(PROJECT_ROOT / "results" / "ddr"))
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

    args.metrics_mode = cfg["metrics_mode"]
    args.data_root = str(data_root)
    if not data_root.exists():
        raise SystemExit(f"data root not found: {data_root}")

    if args.name is None:
        args.name = cfg["default_name"]

    data_yaml = data_root / yaml_name
    write_data_yaml(data_yaml, data_root, names, VAL_FOLDER)

    args.project = str(Path(args.project).resolve())
    args.results_dir = str(Path(args.results_dir).resolve())

    tb_proc = None
    if args.tensorboard:
        tb_proc = start_tensorboard(args.project)

    try:
        if args.backend == "ultralytics":
            run_ultralytics(args, data_yaml)
        else:
            if args.eval_after:
                print("[WARN] --eval-after is only supported for backend=ultralytics")
            run_yolov12_repo(args, data_yaml)
    finally:
        stop_process(tb_proc)


if __name__ == "__main__":
    main()
