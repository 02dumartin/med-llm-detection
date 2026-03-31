#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from ultralytics import YOLO


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.yolo.common import (  # noqa: E402
    DATASET_ALIASES,
    DATASETS,
    canonical_dataset_name,
    class_colors_for_variant,
    class_names_for_variant,
    evaluation_dir,
    get_data_root,
    get_default_eval_split,
    get_overlay_type,
    infer_train_model_from_weights,
    overlay_dir,
    resolve_run_root,
)
from src.visualization.ddr_overlay import ANN_ROOT as DDR_ANN_ROOT  # noqa: E402
from src.visualization.ddr_overlay import draw_ddr_overlay  # noqa: E402
from src.visualization.fgart_overlay import draw_fgart_overlay  # noqa: E402


def find_ddr_xml_by_stem(stem: str) -> Path | None:
    for split in ["train", "valid", "test"]:
        path = DDR_ANN_ROOT / split / f"{stem}.xml"
        if path.exists():
            return path
    return None


def save_gt_overlays(test_images_dir: Path, out_dir: Path, overlay_type: str | None) -> None:
    if overlay_type is None:
        print("[WARN] GT overlay is not supported for this dataset")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in test_images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    for path in imgs:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if overlay_type == "fgart":
            draw_fgart_overlay(ax, path.name, legend=False)
            out_path = out_dir / path.name
        elif overlay_type == "ddr":
            xml_path = find_ddr_xml_by_stem(path.stem)
            if xml_path is None:
                plt.close(fig)
                continue
            draw_ddr_overlay(ax, xml_path, legend=False)
            out_path = out_dir / f"{path.stem}.png"
        else:
            plt.close(fig)
            continue
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


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


def detections_from_txt(label_path: Path, img_w: int, img_h: int, num_classes: int) -> tuple[sv.Detections, list[str]]:
    if not label_path.exists():
        return _empty_detections(), []

    xyxy_list: list[list[float]] = []
    class_ids: list[int] = []
    confs: list[float] = []
    labels: list[str] = []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return _empty_detections(), []

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
        score = float(parts[5]) if len(parts) >= 6 else 1.0
        xyxy_list.append([x1, y1, x2, y2])
        class_ids.append(vis_cls)
        confs.append(score)
        labels.append(f"{raw_cls} {score:.2f}")

    if not xyxy_list:
        return _empty_detections(), []

    detections = sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.array(class_ids, dtype=np.int32),
    )
    return detections, labels


def detections_from_result(result, num_classes: int) -> tuple[sv.Detections, list[str]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return _empty_detections(), []

    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().astype(np.float32)
    raw_cls = boxes.cls.cpu().numpy().astype(np.int32)
    vis_cls = np.clip(raw_cls, 0, max(0, num_classes - 1))
    labels = [f"{int(cls_id)} {float(score):.2f}" for cls_id, score in zip(raw_cls, conf)]
    detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=vis_cls)
    return detections, labels


def render_overlay(
    image_bgr: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
    class_colors: list[str],
) -> np.ndarray:
    palette = sv.ColorPalette.from_hex(class_colors)
    box_annotator = sv.BoxAnnotator(color=palette, thickness=2)
    return box_annotator.annotate(scene=image_bgr.copy(), detections=detections)


def save_pred_overlays_from_txt(
    test_images_dir: Path,
    pred_labels_dir: Path,
    out_dir: Path,
    class_names: list[str],
    class_colors: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in test_images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    for path in imgs:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        detections, labels = detections_from_txt(pred_labels_dir / f"{path.stem}.txt", w, h, len(class_names))
        if labels:
            labels = [f"{class_names[detections.class_id[i]]} {detections.confidence[i]:.2f}" for i in range(len(labels))]
        rendered = render_overlay(image, detections, labels, class_colors)
        cv2.imwrite(str(out_dir / path.name), rendered)


def save_pred_overlays_live(
    model,
    test_images_dir: Path,
    out_dir: Path,
    class_names: list[str],
    class_colors: list[str],
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
    for result in results:
        path = Path(result.path)
        image = cv2.imread(str(path))
        if image is None:
            continue
        detections, labels = detections_from_result(result, len(class_names))
        if labels:
            labels = [f"{class_names[detections.class_id[i]]} {detections.confidence[i]:.2f}" for i in range(len(labels))]
        rendered = render_overlay(image, detections, labels, class_colors)
        cv2.imwrite(str(out_dir / path.name), rendered)


def main() -> None:
    dataset_choices = sorted(set(DATASETS.keys()) | set(DATASET_ALIASES.keys()))
    parser = argparse.ArgumentParser(description="Overlay runner (sv based)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["4cls", "1cls"], default="4cls")
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--test-data", choices=dataset_choices, required=True)
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--source", choices=["txt", "live"], default="txt")
    parser.add_argument("--pred-dir", type=str, default=None, help="prediction txt directory when --source txt")
    parser.add_argument("--save-pred-overlay", action="store_true", default=True)
    parser.add_argument("--no-pred-overlay", dest="save_pred_overlay", action="store_false")
    parser.add_argument("--save-gt-overlay", action="store_true", default=False)
    parser.add_argument("--use-saved-pred-txt", action="store_true", default=False, help="legacy alias for --source txt")
    parser.add_argument("--pred-label-dir", type=str, default=None, help="legacy alias for --pred-dir")
    args = parser.parse_args()

    if args.use_saved_pred_txt:
        args.source = "txt"
    if args.pred_label_dir is not None and args.pred_dir is None:
        args.pred_dir = args.pred_label_dir

    test_data = canonical_dataset_name(args.test_data)
    weights = Path(args.weights)
    infer_train, infer_model = infer_train_model_from_weights(weights)
    train_data = canonical_dataset_name(args.train_data or infer_train or "")
    model_name = args.model_name or infer_model

    if not train_data:
        raise SystemExit("train-data is required")
    if not model_name:
        raise SystemExit("model-name is required")

    data_root = get_data_root(test_data, args.variant)
    if not data_root.exists():
        raise SystemExit(f"test data root not found: {data_root}")

    eval_split = args.eval_split or get_default_eval_split(test_data)
    class_names = class_names_for_variant(args.variant)
    class_colors = class_colors_for_variant(args.variant, class_names)
    overlay_type = get_overlay_type(test_data)

    run_root = resolve_run_root(args.results_dir, train_data, model_name, test_data, args.conf, args.iou)
    eval_dir = evaluation_dir(run_root)
    overlay_root = overlay_dir(run_root)
    pred_overlay_dir = overlay_root / "pred"
    gt_overlay_dir = overlay_root / "gt"
    overlay_root.mkdir(parents=True, exist_ok=True)

    test_images_dir = data_root / eval_split / "images"

    if args.save_gt_overlay:
        save_gt_overlays(test_images_dir, gt_overlay_dir, overlay_type)

    if args.save_pred_overlay:
        if args.source == "txt":
            pred_dir = Path(args.pred_dir) if args.pred_dir is not None else eval_dir / "labels"
            if not pred_dir.exists():
                raise SystemExit(f"saved prediction txt directory not found: {pred_dir}")
            save_pred_overlays_from_txt(test_images_dir, pred_dir, pred_overlay_dir, class_names, class_colors)
        else:
            model = YOLO(str(weights))
            save_pred_overlays_live(
                model,
                test_images_dir,
                pred_overlay_dir,
                class_names,
                class_colors,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
            )


if __name__ == "__main__":
    main()
