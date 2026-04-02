#!/usr/bin/env python3
from __future__ import annotations

"""
Processed test GT overlay generator.

Usage examples:
  # FGART 4cls test GT overlay from COCO annotations
  /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection/src/visualization/gt_overlay.py \
  --dataset fgart --variant 4cls --gt-source coco --split test

  # IDRiD 1cls test GT overlay from YOLO txt annotations
  /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection/.venv/bin/python \
  /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection/src/visualization/gt_overlay.py \
  --dataset idrid --variant 1cls --gt-source txt --split test --legend

  # Save only 10 images to a custom output directory
  /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection/.venv/bin/python \
  /home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection/src/visualization/gt_overlay.py \
  --dataset eophtha --variant 4cls --gt-source coco --limit 10 --out-dir /tmp/eophtha_gt_overlay
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv


PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.yolo.common import (  # noqa: E402
    DATASET_ALIASES,
    DATASETS,
    canonical_dataset_name,
    class_colors_for_variant,
    class_names_for_variant,
    get_default_gt_split,
    get_gt_coco_root,
    get_gt_yolo_root,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class GTRoot:
    dataset: str
    variant: str
    source: str
    split: str
    images_dir: Path
    json_path: Path | None = None
    labels_dir: Path | None = None


def _empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        confidence=np.zeros((0,), dtype=np.float32),
        class_id=np.zeros((0,), dtype=np.int32),
    )


def _xywh_norm_to_xyxy_pixels(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    x1 = (x_center - width / 2.0) * img_w
    y1 = (y_center - height / 2.0) * img_h
    x2 = (x_center + width / 2.0) * img_w
    y2 = (y_center + height / 2.0) * img_h
    return x1, y1, x2, y2


def resolve_gt_dataset_root(dataset: str, variant: str, source: str, split: str) -> GTRoot:
    dataset = canonical_dataset_name(dataset)

    if source == "coco":
        root = get_gt_coco_root(dataset, variant)
        images_dir = root / split / "images"
        json_path = root / split / f"{split}.json"
        return GTRoot(dataset=dataset, variant=variant, source=source, split=split, images_dir=images_dir, json_path=json_path)

    root = get_gt_yolo_root(dataset, variant)
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    return GTRoot(dataset=dataset, variant=variant, source=source, split=split, images_dir=images_dir, labels_dir=labels_dir)


def validate_gt_root(gt_root: GTRoot) -> None:
    if not gt_root.images_dir.exists():
        raise SystemExit(f"image directory not found: {gt_root.images_dir}")
    if gt_root.source == "coco":
        assert gt_root.json_path is not None
        if not gt_root.json_path.exists():
            raise SystemExit(f"COCO json not found: {gt_root.json_path}")
    else:
        assert gt_root.labels_dir is not None
        if not gt_root.labels_dir.exists():
            raise SystemExit(f"YOLO labels directory not found: {gt_root.labels_dir}")


def load_coco_index(json_path: Path) -> dict[str, dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_name = {img["file_name"]: img for img in coco.get("images", [])}
    annotations_by_image_id: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        image_id = ann["image_id"]
        annotations_by_image_id.setdefault(image_id, []).append(ann)

    return {
        "images_by_name": images_by_name,
        "annotations_by_image_id": annotations_by_image_id,
    }


def detections_from_coco_record(file_name: str, coco_index: dict[str, dict], num_classes: int, variant: str) -> sv.Detections:
    images_by_name = coco_index["images_by_name"]
    annotations_by_image_id = coco_index["annotations_by_image_id"]

    img_info = images_by_name.get(file_name)
    if img_info is None:
        return _empty_detections()

    xyxy_list: list[list[float]] = []
    class_ids: list[int] = []
    confs: list[float] = []

    for ann in annotations_by_image_id.get(img_info["id"], []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox[:4]
        class_id = 0 if variant == "1cls" else int(ann.get("category_id", 0))
        vis_cls = max(0, min(class_id, max(0, num_classes - 1)))
        xyxy_list.append([float(x), float(y), float(x + w), float(y + h)])
        class_ids.append(vis_cls)
        confs.append(1.0)

    if not xyxy_list:
        return _empty_detections()

    return sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.array(class_ids, dtype=np.int32),
    )


def detections_from_txt(label_path: Path, img_w: int, img_h: int, num_classes: int) -> sv.Detections:
    if not label_path.exists():
        return _empty_detections()

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return _empty_detections()

    xyxy_list: list[list[float]] = []
    class_ids: list[int] = []
    confs: list[float] = []

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        raw_cls = int(float(parts[0]))
        vis_cls = max(0, min(raw_cls, max(0, num_classes - 1)))
        x1, y1, x2, y2 = _xywh_norm_to_xyxy_pixels(
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            img_w,
            img_h,
        )
        xyxy_list.append([x1, y1, x2, y2])
        class_ids.append(vis_cls)
        confs.append(1.0)

    if not xyxy_list:
        return _empty_detections()

    return sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.array(class_ids, dtype=np.int32),
    )


def render_gt_overlay(image_bgr: np.ndarray, detections: sv.Detections, class_colors: list[str]) -> np.ndarray:
    palette = sv.ColorPalette.from_hex(class_colors)
    box_annotator = sv.BoxAnnotator(color=palette, thickness=2)
    return box_annotator.annotate(scene=image_bgr.copy(), detections=detections)


def draw_legend(image_bgr: np.ndarray, class_names: list[str], class_colors: list[str]) -> np.ndarray:
    rendered = image_bgr.copy()
    pad = 12
    swatch = 18
    gap = 10
    line_h = 26
    box_w = 150
    box_h = pad * 2 + line_h * len(class_names)
    x2 = rendered.shape[1] - pad
    x1 = max(0, x2 - box_w)
    y1 = pad
    y2 = min(rendered.shape[0], y1 + box_h)

    cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)

    for idx, (name, hex_color) in enumerate(zip(class_names, class_colors)):
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        bgr = (rgb[2], rgb[1], rgb[0])
        top = y1 + pad + idx * line_h
        cv2.rectangle(rendered, (x1 + pad, top), (x1 + pad + swatch, top + swatch), bgr, thickness=-1)
        cv2.rectangle(rendered, (x1 + pad, top), (x1 + pad + swatch, top + swatch), (0, 0, 0), thickness=1)
        cv2.putText(
            rendered,
            name,
            (x1 + pad + swatch + gap, top + swatch - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return rendered


def iter_image_paths(images_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if limit is not None:
        return paths[:limit]
    return paths


def save_gt_overlays(
    gt_root: GTRoot,
    out_dir: Path,
    class_names: list[str],
    class_colors: list[str],
    limit: int | None = None,
    legend: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = iter_image_paths(gt_root.images_dir, limit=limit)
    coco_index = load_coco_index(gt_root.json_path) if gt_root.source == "coco" else None

    print(f"[GT] dataset={gt_root.dataset} variant={gt_root.variant} source={gt_root.source} split={gt_root.split}")
    print(f"[GT] images={len(image_paths)} out_dir={out_dir}")

    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] image open failed: {image_path}")
            continue

        if gt_root.source == "coco":
            assert coco_index is not None
            detections = detections_from_coco_record(image_path.name, coco_index, len(class_names), gt_root.variant)
            if image_path.name not in coco_index["images_by_name"]:
                print(f"[WARN] image not found in COCO json: {image_path.name}")
        else:
            assert gt_root.labels_dir is not None
            h, w = image.shape[:2]
            label_path = gt_root.labels_dir / f"{image_path.stem}.txt"
            detections = detections_from_txt(label_path, w, h, len(class_names))
            if not label_path.exists():
                print(f"[WARN] label not found: {label_path}")

        rendered = render_gt_overlay(image, detections, class_colors)
        if legend:
            rendered = draw_legend(rendered, class_names, class_colors)

        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), rendered)

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"[GT] saved {idx}/{len(image_paths)}")


def default_output_dir(dataset: str, variant: str, source: str) -> Path:
    return PROJECT_ROOT / "results" / "gt_overlay" / dataset / variant / source


def parse_args() -> argparse.Namespace:
    dataset_choices = sorted(set(DATASETS.keys()) | set(DATASET_ALIASES.keys()))
    parser = argparse.ArgumentParser(description="Processed test GT overlay generator")
    parser.add_argument("--dataset", choices=dataset_choices, required=True)
    parser.add_argument("--variant", choices=["4cls", "1cls"], default="4cls")
    parser.add_argument("--gt-source", choices=["coco", "txt"], default="coco")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--legend", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = canonical_dataset_name(args.dataset)
    split = args.split or get_default_gt_split(dataset)
    gt_root = resolve_gt_dataset_root(dataset, args.variant, args.gt_source, split)
    validate_gt_root(gt_root)

    class_names = class_names_for_variant(args.variant)
    class_colors = class_colors_for_variant(args.variant, class_names)
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_output_dir(dataset, args.variant, args.gt_source)

    save_gt_overlays(
        gt_root,
        out_dir=out_dir,
        class_names=class_names,
        class_colors=class_colors,
        limit=args.limit,
        legend=args.legend,
    )


if __name__ == "__main__":
    main()
