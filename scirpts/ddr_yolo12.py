#!/usr/bin/env python3
"""
DDR 4cls YOLOv12l 학습 스크립트.

- DDR은 해상도가 다양함(35종). YOLO는 imgsz로 고정 후 letterbox 리사이즈하므로
  별도 처리 없이 학습 가능. (이미지마다 비율 유지 + 패딩으로 동일 크기 입력)

사용법:
    python scirpts/ddr_yolo12.py --device 2
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
DATA_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_yolo_4cls")
NAMES = ["MA", "HE", "EX", "SE"]

VAL_FOLDER = "valid"


def write_data_yaml(out_path: Path, data_root: Path, names: list[str], val_folder: str) -> None:
    out_path.write_text(
        "\n".join(
            [
                f"path: {data_root}",
                "train: train/images",
                f"val: {val_folder}/images",
                "test: test/images",
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
        patience=100,
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

        results_df = pd.DataFrame(
            {
                "metric": [
                    "mAP50",
                    "mAP50-95",
                    "detection_acc",
                    "classification_acc",
                    "overall_acc",
                ],
                "value": [
                    metrics.box.map50,
                    metrics.box.map,
                    detection_acc,
                    classification_acc,
                    overall_acc,
                ],
            }
        )
        results_df.to_csv(eval_dir / "metrics.csv", index=False)

        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.eval.metrics_utils import compute_per_class_prf_ap

        names = [model.names[i] for i in sorted(model.names.keys())]
        df_prf_ap = compute_per_class_prf_ap(metrics, cm, names)
        df_prf_ap.to_csv(eval_dir / "per_class_ap.csv", index=False)

        if args.predict_after:
            test_images = Path(args.data_root) / "test" / "images"
            pred_dir = eval_dir / "prediction"
            pred_dir.mkdir(parents=True, exist_ok=True)
            model.predict(
                source=str(test_images),
                save=True,
                save_txt=True,
                save_conf=True,
                project=str(pred_dir.parent),
                name="prediction",
                exist_ok=True,
                conf=args.conf,
                iou=args.iou,
            )


def run_yolov12_repo(args: argparse.Namespace, data_yaml: Path) -> None:
    if args.repo is None:
        raise SystemExit("--repo is required for backend=yolov12")
    repo = Path(args.repo)
    train_py = repo / "train.py"
    if not train_py.exists():
        raise SystemExit(f"train.py not found in repo: {train_py}")

    cmd = [
        "python",
        str(train_py),
        "--data",
        str(data_yaml),
        "--weights",
        args.model,
        "--img",
        str(args.imgsz),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--project",
        str(args.project),
        "--name",
        str(args.name),
        "--seed",
        str(args.seed),
    ]
    if args.exist_ok:
        cmd.append("--exist-ok")
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))


def main() -> None:
    parser = argparse.ArgumentParser(description="DDR 4cls YOLOv12 training launcher")
    parser.add_argument("--backend", choices=["ultralytics", "yolov12"], default="ultralytics")
    parser.add_argument("--repo", type=str, default=None, help="YOLOv12 repo path (backend=yolov12)")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "weights" / "yolov12l.pt"))
    parser.add_argument("--imgsz", type=int, default=1920, help="DDR 해상도 다양 → letterbox로 이 크기로 통일")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="2")
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

    args.data_root = str(DATA_ROOT)
    if not DATA_ROOT.exists():
        raise SystemExit(f"data root not found: {DATA_ROOT}")

    if args.name is None:
        args.name = "yolo12"

    data_yaml = DATA_ROOT / "ddr_4cls.yaml"
    write_data_yaml(data_yaml, DATA_ROOT, NAMES, VAL_FOLDER)

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
        if args.backend == "ultralytics":
            run_ultralytics(args, data_yaml)
        else:
            if args.eval_after:
                print("[WARN] --eval-after is only supported for backend=ultralytics")
            run_yolov12_repo(args, data_yaml)
    finally:
        if tb_proc is not None:
            tb_proc.terminate()


if __name__ == "__main__":
    main()
