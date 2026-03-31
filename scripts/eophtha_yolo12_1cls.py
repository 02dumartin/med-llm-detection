#!/usr/bin/env python3
"""
E-ophtha 1cls YOLOv12l 학습 스크립트.

- E-ophtha lesion (EX+MA 통합) 1cls

사용법:
    python scripts/eophtha_yolo12_1cls.py --device 5,6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_ROOTS = [
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_1cls"),
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/E-OPTHA_yolo_1cls"),
]
NAMES = ["lesion"]
VAL_FOLDER = "val"


def normalize_device(device: str) -> str:
    s = device.strip()
    if "," not in s:
        return s
    return ",".join(p.strip() for p in s.split(",") if p.strip())


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


def resolve_data_root(cli_path: str | None) -> Path:
    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser().resolve())
    candidates.extend(DEFAULT_DATA_ROOTS)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        if path not in seen:
            ordered.append(path)
            seen.add(path)

    for path in ordered:
        if path.exists():
            return path

    checked = "\n".join(f"  - {path}" for path in ordered)
    raise SystemExit(f"data root not found. Checked:\n{checked}")


def inspect_labels(data_root: Path) -> None:
    class_counter: Counter[int] = Counter()
    split_stats: dict[str, dict[str, int]] = {}
    dense_files: list[tuple[int, str]] = []
    issues: list[str] = []

    for split in ("train", "val", "test"):
        label_dir = data_root / split / "labels"
        label_files = sorted(label_dir.glob("*.txt"))
        nonempty = 0
        backgrounds = 0
        objects = 0

        for label_path in label_files:
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                backgrounds += 1
                dense_files.append((0, f"{split}/{label_path.name}"))
                continue

            lines = [line for line in text.splitlines() if line.strip()]
            nonempty += 1
            objects += len(lines)
            dense_files.append((len(lines), f"{split}/{label_path.name}"))

            for idx, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"{split}/{label_path.name}:{idx} invalid field count -> {line}")
                    continue

                try:
                    cls_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    issues.append(f"{split}/{label_path.name}:{idx} parse error -> {line}")
                    continue

                class_counter[cls_id] += 1
                if cls_id != 0:
                    issues.append(f"{split}/{label_path.name}:{idx} invalid class id {cls_id}")
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    issues.append(f"{split}/{label_path.name}:{idx} out-of-range box -> {line}")
                if x - w / 2 < 0 or x + w / 2 > 1 or y - h / 2 < 0 or y + h / 2 > 1:
                    issues.append(f"{split}/{label_path.name}:{idx} box extends outside image -> {line}")

        split_stats[split] = {
            "files": len(label_files),
            "nonempty": nonempty,
            "backgrounds": backgrounds,
            "objects": objects,
        }

    if issues:
        preview = "\n".join(f"  - {issue}" for issue in issues[:20])
        more = "" if len(issues) <= 20 else f"\n  ... {len(issues) - 20} more"
        raise SystemExit(f"label sanity check failed:\n{preview}{more}")

    print("[Eophtha 1cls] label summary")
    for split, stats in split_stats.items():
        print(
            f"  {split}: files={stats['files']} nonempty={stats['nonempty']} "
            f"backgrounds={stats['backgrounds']} objects={stats['objects']}"
        )
    print(f"  classes: {dict(sorted(class_counter.items()))}")
    print("  densest files:")
    for count, name in sorted(dense_files, reverse=True)[:8]:
        print(f"    {name}: {count} objects")


def run_ultralytics(args: argparse.Namespace, data_yaml: Path) -> None:
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=normalize_device(args.device),
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=args.exist_ok,
        single_cls=True,
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
        total_gt = cm[:nc, :].sum()
        total_detected = cm[:nc, :nc].sum()
        detection_acc = total_detected / total_gt if total_gt > 0 else 0

        results_df = pd.DataFrame(
            {
                "metric": ["mAP50", "mAP50-95", "detection_acc"],
                "value": [metrics.box.map50, metrics.box.map, detection_acc],
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
    parser = argparse.ArgumentParser(description="E-ophtha 1cls YOLOv12 training launcher")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "weights" / "yolov12l.pt"))
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA 장치. 단일: 0 | DDP 다중 GPU: 0,1,2",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default=str(PROJECT_ROOT / "runs" / "eophtha"))
    parser.add_argument("--name", type=str, default=None, help="default: yolo12_1cls")
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
    parser.set_defaults(amp=False)
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    inspect_labels(data_root)
    args.data_root = str(data_root)

    if args.name is None:
        args.name = "yolo12_1cls"

    data_yaml = data_root / "eophtha_1cls.yaml"
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
