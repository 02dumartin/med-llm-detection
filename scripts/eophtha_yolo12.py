#!/usr/bin/env python3
"""
E-ophtha YOLOv12l 학습 스크립트 (2cls / 1cls 공용).

사용법:
    python scripts/eophtha_yolo12.py --variant 2cls --device 6
    python scripts/eophtha_yolo12.py --variant 1cls --device 5,6
    python scripts/eophtha_yolo12.py --variant 2cls --device 0 1 2 --batch 4
    python scripts/eophtha_yolo12.py --variant 2cls --device 5 --resume
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.yolo.training import (
    device_parts_to_ultralytics,
    save_eval_outputs,
    start_tensorboard,
    stop_process,
    write_data_yaml as write_yolo_data_yaml,
)


DEFAULT_DATA_ROOT_2CLS = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_4cls")
DEFAULT_DATA_ROOTS_1CLS = [
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_1cls"),
    Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/E-OPTHA_yolo_1cls"),
]
VAL_FOLDER = "val"
VARIANT_CFG = {
    "2cls": {
        "names": ["MA", "EX"],
        "yaml": "eophtha_2cls.yaml",
        "default_name": "yolo12",
        "metrics_mode": "multi",
        "single_cls": False,
    },
    "1cls": {
        "names": ["lesion"],
        "yaml": "eophtha_1cls.yaml",
        "default_name": "yolo12_1cls",
        "metrics_mode": "single",
        "single_cls": True,
    },
}


def write_data_yaml(out_path: Path, data_root: Path, names: list[str], val_folder: str) -> None:
    write_yolo_data_yaml(out_path, data_root, names, val_folder=val_folder)


def resolve_data_root(variant: str, cli_path: str | None) -> Path:
    if variant == "2cls":
        data_root = Path(cli_path).expanduser().resolve() if cli_path else DEFAULT_DATA_ROOT_2CLS
        return data_root

    candidates: list[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser().resolve())
    candidates.extend(DEFAULT_DATA_ROOTS_1CLS)

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


def validate_2cls_labels(data_root: Path, allowed_ids: set[int]) -> None:
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
            f"E-ophtha 2cls label validation failed. Expected class ids {sorted(allowed_ids)}.\n{preview}{more}"
        )

    print(f"[Eophtha 2cls] validated class ids: {sorted(seen_ids)}")


def inspect_1cls_labels(data_root: Path) -> None:
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
        single_cls=args.single_cls,
    )
    if args.resume_path is not None:
        model = YOLO(str(args.resume_path))
        model.train(resume=True, **train_kw)
    else:
        model = YOLO(args.model)
        model.train(**train_kw)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="E-ophtha YOLOv12 training launcher (2cls / 1cls)")
    parser.add_argument("--variant", choices=["2cls", "1cls"], default="2cls")
    parser.add_argument("--backend", choices=["ultralytics", "yolov12"], default="ultralytics")
    parser.add_argument("--repo", type=str, default=None, help="reserved for backend=yolov12")
    parser.add_argument("--data-root", type=str, default=None)
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
    parser.add_argument("--name", type=str, default=None)
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

    if args.backend != "ultralytics":
        raise SystemExit("backend=yolov12 is not supported for eophtha launcher yet")

    args.device = device_parts_to_ultralytics(args.device)
    cfg = VARIANT_CFG[args.variant]
    data_root = resolve_data_root(args.variant, args.data_root)
    if not data_root.exists():
        raise SystemExit(f"data root not found: {data_root}")

    args.data_root = str(data_root)
    args.metrics_mode = cfg["metrics_mode"]
    args.single_cls = cfg["single_cls"]

    if args.name is None:
        args.name = cfg["default_name"]

    if args.resume is not None:
        ckpt = Path(args.resume) if args.resume else Path(args.project) / args.name / "weights" / "last.pt"
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        args.resume_path = ckpt
    else:
        args.resume_path = None

    if args.variant == "2cls":
        validate_2cls_labels(data_root, allowed_ids={0, 1})
    else:
        inspect_1cls_labels(data_root)

    data_yaml = data_root / cfg["yaml"]
    write_data_yaml(data_yaml, data_root, cfg["names"], VAL_FOLDER)

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
