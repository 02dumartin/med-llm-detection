#!/usr/bin/env python3
"""
E-ophtha 2cls YOLOv12l 학습 스크립트.

 - E-ophtha: MA/EX만 사용하는 2cls 설정
 - 1cls 비교와 맞추기 위해 기본 하이퍼파라미터를 동일하게 설정

사용법:
    python scripts/eophtha_yolo12.py --device 6
    # 여러 GPU (DDP) — 둘 다 동일
    python scripts/eophtha_yolo12.py --device 0 1 2 --batch 4
    python scripts/eophtha_yolo12.py --device 0,1,2 --batch 4
    # 끊긴 학습 이어가기 (기본: runs/eophtha/<name>/weights/last.pt)
    python scripts/eophtha_yolo12.py --device 5 --resume
    python scripts/eophtha_yolo12.py --device 0 1 2 --resume
    python scripts/eophtha_yolo12.py --device 5 --resume /path/to/last.pt
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

# E-ophtha YOLO 데이터 (eophtha_prepare_yolo.py 출력)
# 실제 폴더명은 yolo_4cls지만, JSON remap_2to1 이후 MA(0), EX(1)만 사용한다.
DATA_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_4cls")
NAMES = ["MA", "EX"]
VAL_FOLDER = "val"


def device_parts_to_ultralytics(parts: list[str]) -> str:
    """--device 인자(여러 토큰 또는 '0,1' 한 토큰)를 Ultralytics용 '0,1,2' 문자열로 합친다."""
    out: list[str] = []
    for p in parts:
        for seg in p.split(","):
            seg = seg.strip()
            if seg:
                out.append(seg)
    if not out:
        raise SystemExit("--device: GPU ID가 비었습니다.")
    return ",".join(out)


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


def validate_label_classes(data_root: Path, allowed_ids: set[int]) -> None:
    seen_ids: set[int] = set()
    invalid: list[str] = []

    for split in ("train", "val", "test"):
        label_dir = data_root / split / "labels"
        for label_path in sorted(label_dir.glob("*.txt")):
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for idx, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    invalid.append(f"{split}/{label_path.name}:{idx} invalid field count")
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    invalid.append(f"{split}/{label_path.name}:{idx} parse error")
                    continue
                seen_ids.add(cls_id)
                if cls_id not in allowed_ids:
                    invalid.append(f"{split}/{label_path.name}:{idx} unexpected class id {cls_id}")

    if invalid:
        preview = "\n".join(f"  - {row}" for row in invalid[:20])
        more = "" if len(invalid) <= 20 else f"\n  ... {len(invalid) - 20} more"
        raise SystemExit(
            f"E-ophtha 2cls label validation failed. Expected class ids {sorted(allowed_ids)}.\n"
            f"{preview}{more}"
        )

    print(f"[Eophtha 2cls] validated class ids: {sorted(seen_ids)}")


def run_ultralytics(args: argparse.Namespace, data_yaml: Path) -> None:
    train_kw = dict(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=args.exist_ok if args.resume_path is None else True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=0.01,
        momentum=args.momentum,
        weight_decay=0.0005,
        cos_lr=True,
        amp=args.amp,
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
    if args.resume_path is not None:
        model = YOLO(str(args.resume_path))
        model.train(resume=True, **train_kw)
    else:
        model = YOLO(args.model)
        model.train(**train_kw)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="E-ophtha 2cls YOLOv12 training launcher")
    parser.add_argument("--backend", choices=["ultralytics", "yolov12"], default="ultralytics")
    parser.add_argument("--repo", type=str, default=None, help="YOLOv12 repo path (backend=yolov12)")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "weights" / "yolov12l.pt"))
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--device",
        nargs="+",
        default=["0"],
        metavar="ID",
        help="CUDA 장치. 예: --device 5 | DDP: --device 0 1 2 또는 --device 0,1,2",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default=str(PROJECT_ROOT / "runs" / "eophtha"))
    parser.add_argument("--name", type=str, default=None, help="default: yolo12")
    parser.add_argument("--results-dir", type=str, default=str(PROJECT_ROOT / "results" / "eophtha"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--amp", action="store_true", help="enable AMP mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="disable AMP mixed precision")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--eval-after", action="store_true")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--predict-after", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="이어서 학습: last.pt 경로. 플래그만 주면 project/name/weights/last.pt",
    )
    parser.set_defaults(amp=False)
    args = parser.parse_args()
    args.device = device_parts_to_ultralytics(args.device)

    args.data_root = str(DATA_ROOT)
    if not DATA_ROOT.exists():
        raise SystemExit(f"data root not found: {DATA_ROOT}")

    if args.name is None:
        args.name = "yolo12"

    if args.resume is not None:
        ckpt = Path(args.resume) if args.resume else Path(args.project) / args.name / "weights" / "last.pt"
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        args.resume_path = ckpt
    else:
        args.resume_path = None

    validate_label_classes(DATA_ROOT, allowed_ids={0, 1})

    data_yaml = DATA_ROOT / "eophtha_2cls.yaml"
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
        run_ultralytics(args, data_yaml)
    finally:
        if tb_proc is not None:
            tb_proc.terminate()


if __name__ == "__main__":
    main()
