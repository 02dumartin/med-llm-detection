# src/eval/metrics_utils.py
"""공통 메트릭 계산 (학습 eval-after, yolo_eval)"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_per_class_prf_ap(metrics, cm: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """
    Confusion matrix + metrics에서 클래스별 precision, recall, f1, AP50, AP50-95 계산.
    cm: rows=GT, cols=pred (마지막 col=background/FN)
    """
    nc = len(class_names)
    ap50 = np.array(metrics.box.ap50) if hasattr(metrics.box, "ap50") else np.zeros(nc)
    ap = np.array(metrics.box.ap) if hasattr(metrics.box, "ap") else np.zeros(nc)

    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        total_gt = cm[i, :].sum()
        total_pred_as_i = cm[:, i].sum()

        precision = tp / total_pred_as_i if total_pred_as_i > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "class": name,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "AP@0.5": float(ap50[i]) if i < len(ap50) else 0.0,
                "AP@0.5:0.95": float(ap[i]) if i < len(ap) else 0.0,
            }
        )

    # overall (mean)
    tp_total = cm[:nc, :nc].diagonal().sum()
    total_gt_all = cm[:nc, :].sum()
    total_pred_all = cm[:nc, :nc].sum()
    p_mean = tp_total / total_pred_all if total_pred_all > 0 else 0.0
    r_mean = tp_total / total_gt_all if total_gt_all > 0 else 0.0
    f1_mean = 2 * p_mean * r_mean / (p_mean + r_mean) if (p_mean + r_mean) > 0 else 0.0

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
