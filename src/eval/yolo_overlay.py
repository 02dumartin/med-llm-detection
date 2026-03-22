#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.fgart_overlay import draw_fgart_overlay
from src.visualization.ddr_overlay import draw_ddr_overlay, ANN_ROOT as DDR_ANN_ROOT


DATASETS = {
    "fgart": {
        "root_4cls": PROJECT_ROOT / "data" / "FGART_yolo_4cls",
        "root_1cls": PROJECT_ROOT / "data" / "FGART_yolo_1cls",
        "val_folder": "val",
        "overlay": "fgart",
    },
    "ddr": {
        "root_4cls": PROJECT_ROOT / "data" / "DDR_yolo_4cls",
        "root_1cls": PROJECT_ROOT / "data" / "DDR_yolo_1cls",
        "val_folder": "valid",
        "overlay": "ddr",
    },
    "idrid": {
        "root_4cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_4cls"),
        "root_1cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_1cls"),
        "val_folder": "val",
        "overlay": None,
    },
    "e-optha": {
        "root_4cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_4cls"),
        "root_1cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_1cls"),
        "val_folder": "val",
        "overlay": None,
    },
}

CLASS_COLORS = {
    "MA": "#00FF00",
    "HE": "#FF0000",
    "EX": "#FFFF00",
    "SE": "#00FFFF",
}


def infer_train_model_from_weights(weights: Path) -> tuple[str | None, str | None]:
    # 기대 경로: .../runs/<train>/<model>/weights/best.pt
    parts = weights.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if len(parts) >= idx + 4:
            train_name = parts[idx + 1]
            model_name = parts[idx + 2]
            return train_name, model_name
    return None, None


def find_ddr_xml_by_stem(stem: str) -> Path | None:
    for split in ["train", "valid", "test"]:
        p = DDR_ANN_ROOT / split / f"{stem}.xml"
        if p.exists():
            return p
    return None


def save_gt_overlays(test_images_dir: Path, out_dir: Path, overlay_type: str | None) -> None:
    if overlay_type is None:
        print("[WARN] GT overlay는 지원되지 않는 데이터셋입니다.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in test_images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    for p in imgs:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if overlay_type == "fgart":
            draw_fgart_overlay(ax, p.name, legend=False)
            out_path = out_dir / p.name
        elif overlay_type == "ddr":
            xml_path = find_ddr_xml_by_stem(p.stem)
            if xml_path is None:
                plt.close(fig)
                continue
            draw_ddr_overlay(ax, xml_path, legend=False)
            out_path = out_dir / f"{p.stem}.png"
        else:
            plt.close(fig)
            continue
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def save_pred_overlays(
    model,
    test_images_dir: Path,
    out_dir: Path,
    class_names: list[str],
    conf: float,
    iou: float,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(test_images_dir),
        conf=conf,
        iou=iou,
        device=device,
        save=False,
        verbose=False,
    )

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for r in results:
        p = Path(r.path)
        im = Image.open(p).convert("RGB")
        draw = ImageDraw.Draw(im)

        boxes = r.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.tolist()
            cls_ids = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            for (x1, y1, x2, y2), cls_id, score in zip(xyxy, cls_ids, confs):
                cls_id = int(cls_id)
                name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
                color = "#FF00FF" if len(class_names) == 1 else CLASS_COLORS.get(name.upper(), "#FFFFFF")
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                if font is not None:
                    draw.text((x1 + 2, y1 + 2), f"{name} {score:.2f}", fill=color, font=font)

        im.save(out_dir / p.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay 전용 스크립트 (pred/gt overlay 저장)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["4cls", "1cls"], default="4cls")
    parser.add_argument("--train-data", choices=DATASETS.keys(), default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--test-data", choices=DATASETS.keys(), required=True)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="2")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="결과 저장 루트. 미지정 시 PROJECT_ROOT/results/{train_data}/{model_name}/{eval_name}")
    parser.add_argument("--save-pred-overlay", action="store_true", default=True)
    parser.add_argument("--no-pred-overlay", dest="save_pred_overlay", action="store_false")
    parser.add_argument("--save-gt-overlay", action="store_true", default=False)
    args = parser.parse_args()

    weights = Path(args.weights)
    infer_train, infer_model = infer_train_model_from_weights(weights)
    train_data = args.train_data or infer_train
    model_name = args.model_name or infer_model

    if train_data is None:
        raise SystemExit("train-data를 지정해주세요")
    if model_name is None:
        raise SystemExit("model-name를 지정해주세요")

    test_cfg = DATASETS[args.test_data]
    data_root = test_cfg["root_4cls"] if args.variant == "4cls" else test_cfg["root_1cls"]
    overlay_type = test_cfg["overlay"]
    names = ["MA", "HE", "EX", "SE"] if args.variant == "4cls" else ["lesion"]

    if not data_root.exists():
        raise SystemExit(f"test data root not found: {data_root}")

    eval_name = f"{args.test_data}_{args.conf}_{args.iou}"
    if args.results_dir is not None:
        results_dir = Path(args.results_dir) / model_name / eval_name
    else:
        results_dir = PROJECT_ROOT / "results" / train_data / model_name / eval_name
    results_dir.mkdir(parents=True, exist_ok=True)

    test_images_dir = data_root / args.eval_split / "images"

    if args.save_gt_overlay:
        save_gt_overlays(test_images_dir, results_dir / "overlay_test_gt", overlay_type)

    if args.save_pred_overlay:
        model = YOLO(str(weights))
        save_pred_overlays(
            model,
            test_images_dir,
            results_dir / "overlay_test_pred",
            names,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )


if __name__ == "__main__":
    main()
