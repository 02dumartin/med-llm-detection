# src/eval/metrics_utils.py
"""공통 메트릭 계산 (학습 eval-after, yolo_eval)"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_per_class_prf_ap(metrics, cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """
    Ultralytics metrics에서 클래스별 precision, recall, f1, AP50, AP50-95 계산.
    cm: rows=GT, cols=pred (마지막 col=background/FN)
    """
    nc = len(class_names)
    box = metrics.box

    # Ultralytics returns per-class arrays aligned to ap_class_index, not always 0..nc-1.
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
                "precision": round(float(p[i]), 6),
                "recall": round(float(r[i]), 6),
                "f1": round(float(f1[i]), 6),
                "AP@0.5": float(ap50[i]),
                "AP@0.5:0.95": float(ap[i]),
            }
        )

    # overall: use Ultralytics-provided macro means for P/R and macro mean of per-class F1.
    p_mean = float(getattr(box, "mp", 0.0))
    r_mean = float(getattr(box, "mr", 0.0))
    f1_mean = float(f1_raw.mean()) if len(f1_raw) else 0.0

    rows.append(
        {
            "class": "overall",
            "precision": round(p_mean, 6),
            "recall": round(r_mean, 6),
            "f1": round(f1_mean, 6),
            "AP@0.5": float(metrics.box.map50),
            "AP@0.5:0.95": float(metrics.box.map),
        }
    )
    return pd.DataFrame(rows)
