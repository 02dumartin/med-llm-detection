"""YOLO metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

FPPI_POINTS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def compute_class_accuracy(cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    nc = len(class_names)
    rows: list[dict] = []

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        total_gt = cm[i, : nc + 1].sum()
        total_detected = cm[i, :nc].sum()
        rows.append(
            {
                "class": name,
                "detection_acc": (total_detected / total_gt) if total_gt > 0 else 0.0,
                "classification_acc": (tp / total_detected) if total_detected > 0 else 0.0,
                "overall_acc": (tp / total_gt) if total_gt > 0 else 0.0,
            }
        )

    tp_total = cm[:nc, :nc].diagonal().sum()
    total_gt_all = cm[:nc, :].sum()
    total_det_all = cm[:nc, :nc].sum()
    rows.append(
        {
            "class": "overall",
            "detection_acc": total_det_all / total_gt_all if total_gt_all > 0 else 0.0,
            "classification_acc": tp_total / total_det_all if total_det_all > 0 else 0.0,
            "overall_acc": tp_total / total_gt_all if total_gt_all > 0 else 0.0,
        }
    )
    return pd.DataFrame(rows)


def compute_class_prediction(metrics, cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    nc = len(class_names)
    box = metrics.box

    ap_class_index = np.array(getattr(box, "ap_class_index", []), dtype=int)
    p_raw = np.array(getattr(box, "p", []), dtype=float)
    r_raw = np.array(getattr(box, "r", []), dtype=float)
    f1_raw = np.array(getattr(box, "f1", []), dtype=float)
    ap50_raw = np.array(getattr(box, "ap50", []), dtype=float)
    ap_raw = np.array(getattr(box, "ap", []), dtype=float)

    p = np.zeros(nc, dtype=float)
    r = np.zeros(nc, dtype=float)
    f1 = np.zeros(nc, dtype=float)
    ap50 = np.zeros(nc, dtype=float)
    ap = np.zeros(nc, dtype=float)

    for src_idx, cls_idx in enumerate(ap_class_index):
        if 0 <= cls_idx < nc:
            if src_idx < len(p_raw):
                p[cls_idx] = p_raw[src_idx]
            if src_idx < len(r_raw):
                r[cls_idx] = r_raw[src_idx]
            if src_idx < len(f1_raw):
                f1[cls_idx] = f1_raw[src_idx]
            if src_idx < len(ap50_raw):
                ap50[cls_idx] = ap50_raw[src_idx]
            if src_idx < len(ap_raw):
                ap[cls_idx] = ap_raw[src_idx]

    rows = []
    for i, name in enumerate(class_names):
        rows.append(
            {
                "class": name,
                "AP@0.5": round(float(ap50[i]), 6),
                "AP@0.5:0.95": round(float(ap[i]), 6),
                "precision": round(float(p[i]), 6),
                "recall": round(float(r[i]), 6),
                "f1": round(float(f1[i]), 6),
            }
        )

    f1_mean = float(f1_raw.mean()) if len(f1_raw) else 0.0
    rows.append(
        {
            "class": "overall",
            "AP@0.5": round(float(metrics.box.map50), 6),
            "AP@0.5:0.95": round(float(metrics.box.map), 6),
            "precision": round(float(getattr(box, "mp", 0.0)), 6),
            "recall": round(float(getattr(box, "mr", 0.0)), 6),
            "f1": round(f1_mean, 6),
        }
    )
    return pd.DataFrame(rows)


def compute_per_class_prf_ap(metrics, cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    return compute_class_prediction(metrics, cm, class_names)


def compute_froc(
    all_preds: list[tuple[float, str, tuple, int]],
    gt_by_stem: dict[str, list[tuple[int, tuple]]],
    num_images: int,
    class_names: list[str],
    iou_thresh: float = 0.3,
) -> pd.DataFrame:
    nc = len(class_names)

    total_gt_cls = [0] * nc
    total_gt = 0
    for boxes in gt_by_stem.values():
        for cls, _ in boxes:
            if 0 <= cls < nc:
                total_gt_cls[cls] += 1
                total_gt += 1

    def froc_curve(target_cls: int | None = None) -> tuple[list[tuple[float, float]], int]:
        n_gt = total_gt if target_cls is None else total_gt_cls[target_cls]
        if n_gt == 0:
            return [], n_gt
        matched: dict[str, set] = {stem: set() for stem in gt_by_stem}
        tp = fp = 0
        curve: list[tuple[float, float]] = []
        for _, stem, bbox, cls in all_preds:
            if target_cls is not None and cls != target_cls:
                continue
            gts = gt_by_stem.get(stem, [])
            best_iou, best_idx = 0.0, -1
            for idx, (gt_cls, gt_bbox) in enumerate(gts):
                if target_cls is not None and gt_cls != target_cls:
                    continue
                if idx in matched[stem]:
                    continue
                iou = _iou_xyxy(bbox, gt_bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou >= iou_thresh and best_idx >= 0:
                matched[stem].add(best_idx)
                tp += 1
            else:
                fp += 1
            curve.append((fp / num_images, tp / n_gt))
        return curve, n_gt

    def sens_at_fppi(curve: list[tuple[float, float]], target: float) -> float:
        for fppi_val, sens in curve:
            if fppi_val >= target:
                return sens
        return curve[-1][1] if curve else 0.0

    rows: list[dict] = []
    for cls_idx, cls_name in enumerate(class_names):
        curve, n_gt = froc_curve(target_cls=cls_idx)
        sens_list = [sens_at_fppi(curve, fp) for fp in FPPI_POINTS]
        row: dict = {"class": cls_name, "n_gt": n_gt}
        for fp, sens in zip(FPPI_POINTS, sens_list):
            row[f"FPPI={fp}"] = round(sens, 4)
        row["avg_FROC"] = round(sum(sens_list) / len(sens_list), 4)
        rows.append(row)

    curve, n_gt = froc_curve(target_cls=None)
    sens_list = [sens_at_fppi(curve, fp) for fp in FPPI_POINTS]
    row = {"class": "overall", "n_gt": n_gt}
    for fp, sens in zip(FPPI_POINTS, sens_list):
        row[f"FPPI={fp}"] = round(sens, 4)
    row["avg_FROC"] = round(sum(sens_list) / len(sens_list), 4)
    rows.append(row)

    return pd.DataFrame(rows)


def compute_summary(
    df_accuracy: pd.DataFrame,
    df_prediction: pd.DataFrame,
    df_froc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = df_accuracy.merge(df_prediction, on="class", how="left")
    if df_froc is not None:
        froc_cols = [c for c in df_froc.columns if c != "n_gt"]
        merged = merged.merge(df_froc[froc_cols], on="class", how="left")
    return merged


def _iou_xyxy(b1: tuple, b2: tuple) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0
