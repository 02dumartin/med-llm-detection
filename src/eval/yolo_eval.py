#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATASETS = {
    "fgart": {
        "root_4cls": PROJECT_ROOT / "data" / "FGART_yolo_4cls",
        "root_1cls": PROJECT_ROOT / "data" / "FGART_yolo_1cls",
        "val_folder": "val",
    },
    "ddr": {
        "root_4cls": PROJECT_ROOT / "data" / "DDR_yolo_4cls",
        "root_1cls": PROJECT_ROOT / "data" / "DDR_yolo_1cls",
        "val_folder": "valid",
    },
    "idrid": {
        "root_4cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_4cls"),
        "root_1cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_1cls"),
        "val_folder": "val",
    },
    "e-optha": {
        "root_4cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_4cls"),
        "root_1cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_1cls"),
        "val_folder": "val",
    },
}

FPPI_POINTS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def infer_train_model_from_weights(weights: Path) -> tuple[str | None, str | None]:
    # 기대 경로: .../runs/<train>/<model>/weights/best.pt
    parts = weights.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if len(parts) >= idx + 4:
            return parts[idx + 1], parts[idx + 2]
    return None, None


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


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def _plot_cm(cm: np.ndarray, labels: list[str], out_path: Path, normalize: bool) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n + 2), max(5, n + 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fmt = ".2f" if normalize else ".0f"
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    labels = class_names + ["background"]
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")
    _plot_cm(cm, labels, out_dir / "confusion_matrix.png", normalize=False)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)
    _plot_cm(cm_norm, labels, out_dir / "confusion_matrix_normalized.png", normalize=True)


# ── Metrics 저장 ──────────────────────────────────────────────────────────────

def save_overall_metrics(metrics, cm: np.ndarray, out_dir: Path) -> None:
    nc = cm.shape[0] - 1
    tp_correct_class = cm[:nc, :nc].diagonal().sum()
    total_gt = cm[:nc, :].sum()
    total_detected = cm[:nc, :nc].sum()

    detection_acc = total_detected / total_gt if total_gt > 0 else 0
    classification_acc = tp_correct_class / total_detected if total_detected > 0 else 0
    overall_acc = tp_correct_class / total_gt if total_gt > 0 else 0

    # mAP는 per_class_ap.csv에 저장하므로 여기서는 det/cls/overall만 저장
    pd.DataFrame(
        {
            "metric": ["detection_acc", "classification_acc", "overall_acc"],
            "value": [detection_acc, classification_acc, overall_acc],
        }
    ).to_csv(out_dir / "metrics.csv", index=False)
    
    return detection_acc, classification_acc, overall_acc


def compute_per_class_metrics(cm: np.ndarray, class_names: list[str]) -> list[dict]:
    nc = len(class_names)
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        total_gt = cm[i, : nc + 1].sum()
        total_detected = cm[i, :nc].sum()
        detection_acc = (total_detected / total_gt) if total_gt > 0 else 0
        classification_acc = (tp / total_detected) if total_detected > 0 else 0
        overall_acc = (tp / total_gt) if total_gt > 0 else 0
        rows.append(
            {
                "class": name,
                "detection_acc": detection_acc,
                "classification_acc": classification_acc,
                "overall_acc": overall_acc,
            }
        )

    # overall 행 추가
    tp_total = cm[:nc, :nc].diagonal().sum()
    total_gt_all = cm[:nc, :].sum()
    total_det_all = cm[:nc, :nc].sum()
    rows.append(
        {
            "class": "overall",
            "detection_acc": total_det_all / total_gt_all if total_gt_all > 0 else 0,
            "classification_acc": tp_total / total_det_all if total_det_all > 0 else 0,
            "overall_acc": tp_total / total_gt_all if total_gt_all > 0 else 0,
        }
    )
    return rows


def save_per_class_ap(metrics, cm: np.ndarray, class_names: list[str], out_dir: Path) -> pd.DataFrame:
    """클래스별 precision, recall, f1, AP@0.5, AP@0.5:0.95 저장"""
    from src.eval.metrics_utils import compute_per_class_prf_ap  # type: ignore[reportMissingImports]

    df = compute_per_class_prf_ap(metrics, cm, class_names)
    df.to_csv(out_dir / "per_class_ap.csv", index=False)
    return df


def save_metrics_total(
    df_per_class: pd.DataFrame,
    df_ap: pd.DataFrame,
    out_dir: Path,
    df_froc: pd.DataFrame | None = None,
) -> None:
    """모든 메트릭을 클래스 기준으로 합쳐 metrics_total.csv 저장"""
    merged = df_per_class.merge(df_ap, on="class", how="left")
    if df_froc is not None:
        froc_cols = [c for c in df_froc.columns if c != "n_gt"]
        merged = merged.merge(df_froc[froc_cols], on="class", how="left")
    merged.to_csv(out_dir / "metrics_total.csv", index=False)


# ── FROC ─────────────────────────────────────────────────────────────────────

def _xywh_norm_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


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


def _load_yolo_label(path: Path) -> list[tuple[int, tuple]]:
    """YOLO txt 레이블 로드 → [(cls_id, (x1,y1,x2,y2) normalized), ...]"""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        bbox = _xywh_norm_to_xyxy(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        boxes.append((cls, bbox))
    return boxes


def compute_and_save_froc(
    model,
    test_images_dir: Path,
    test_labels_dir: Path,
    class_names: list[str],
    out_dir: Path,
    device: str,
    iou_thresh: float = 0.3,
) -> None:
    """
    FROC 분석: 7개 FPPI 포인트에서 sensitivity + avg FROC 저장
    iou_thresh: GT matching IoU threshold (의료영상 표준 0.3)
    """
    imgs = sorted(
        [p for p in test_images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    num_images = len(imgs)
    if num_images == 0:
        print("[FROC] 이미지 없음, 건너뜀")
        return

    # GT 로드
    gt_by_stem: dict[str, list[tuple[int, tuple]]] = {}
    for img_path in imgs:
        gt_by_stem[img_path.stem] = _load_yolo_label(test_labels_dir / (img_path.stem + ".txt"))

    nc = len(class_names)
    total_gt_cls = [0] * nc
    total_gt = 0
    for boxes in gt_by_stem.values():
        for cls, _ in boxes:
            if 0 <= cls < nc:
                total_gt_cls[cls] += 1
                total_gt += 1

    print(f"[FROC] GT 총 {total_gt}개 ({num_images}장), predict(conf=0.001) 실행 중...")

    # conf=0.001로 모든 후보 예측 수집
    results = model.predict(
        source=str(test_images_dir),
        conf=0.001,
        iou=0.5,
        device=device,
        save=False,
        verbose=False,
    )

    # 예측 수집: (confidence, stem, bbox_norm_xyxy, cls_id)
    all_preds: list[tuple[float, str, tuple, int]] = []
    for r in results:
        stem = Path(r.path).stem
        if r.boxes is None or len(r.boxes) == 0:
            continue
        orig_h, orig_w = r.orig_shape
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            bbox_norm = (x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h)
            all_preds.append((float(confs[i]), stem, bbox_norm, int(clss[i])))

    all_preds.sort(key=lambda x: -x[0])

    def _froc_curve(target_cls: int | None = None):
        n_gt = total_gt if target_cls is None else total_gt_cls[target_cls] if target_cls is not None else total_gt
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

    def _sens_at_fppi(curve: list, target: float) -> float:
        for fppi_val, sens in curve:
            if fppi_val >= target:
                return sens
        return curve[-1][1] if curve else 0.0

    rows = []

    # 클래스별 FROC
    for cls_idx, cls_name in enumerate(class_names):
        curve, n_gt = _froc_curve(target_cls=cls_idx)
        sens_list = [_sens_at_fppi(curve, fp) for fp in FPPI_POINTS]
        avg = sum(sens_list) / len(sens_list)
        row: dict = {"class": cls_name, "n_gt": n_gt}
        for fp, s in zip(FPPI_POINTS, sens_list):
            row[f"FPPI={fp}"] = round(s, 4)
        row["avg_FROC"] = round(avg, 4)
        rows.append(row)

    # overall FROC
    curve, n_gt = _froc_curve(target_cls=None)
    sens_list = [_sens_at_fppi(curve, fp) for fp in FPPI_POINTS]
    avg = sum(sens_list) / len(sens_list)
    row = {"class": "overall", "n_gt": n_gt}
    for fp, s in zip(FPPI_POINTS, sens_list):
        row[f"FPPI={fp}"] = round(s, 4)
    row["avg_FROC"] = round(avg, 4)
    rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "froc.csv", index=False)
    print(f"[FROC] 저장 완료: {out_dir / 'froc.csv'}  (overall avg FROC = {avg:.4f})")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified eval (metrics only)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["4cls", "1cls"], default="4cls")
    parser.add_argument("--train-data", choices=DATASETS.keys(), default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--test-data", choices=DATASETS.keys(), required=True)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--conf", type=float, default=0.25, help="폴더명용; val은 conf 미사용(기본 0.001)")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="2")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="결과 저장 루트. 미지정 시 PROJECT_ROOT/results/{train_data}/{model_name}/{eval_name}")
    parser.add_argument("--save-froc", action="store_true", default=False,
                        help="FROC 분석 실행 (conf=0.001로 predict 추가 실행)")
    parser.add_argument("--froc-iou", type=float, default=0.3,
                        help="FROC GT matching IoU threshold (기본 0.3)")
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
    val_folder = test_cfg["val_folder"]
    names = ["MA", "HE", "EX", "SE"] if args.variant == "4cls" else ["lesion"]

    if not data_root.exists():
        raise SystemExit(f"test data root not found: {data_root}")

    data_yaml = data_root / f"{args.test_data}_{args.variant}.yaml"
    write_data_yaml(data_yaml, data_root, names, val_folder)

    eval_name = f"{args.test_data}_{args.conf}_{args.iou}"
    if args.results_dir is not None:
        results_dir = Path(args.results_dir) / model_name / eval_name
    else:
        results_dir = PROJECT_ROOT / "results" / train_data / model_name / eval_name
    results_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=args.eval_split,
        iou=args.iou,
        device=args.device,
        plots=True,
        project=str(results_dir.parent),
        name=eval_name,
        exist_ok=True,
        save_json=False,
    )

    # ── metrics 저장 ──────────────────────────────────────────────────────────
    cm = metrics.confusion_matrix.matrix
    save_overall_metrics(metrics, cm, results_dir)       # metrics.csv: det/cls/overall
    save_confusion_matrix(cm, names, results_dir)
    df_per_class = pd.DataFrame(compute_per_class_metrics(cm, names))
    df_per_class.to_csv(results_dir / "per_class_metrics.csv", index=False)
    df_ap = save_per_class_ap(metrics, cm, names, results_dir)   # per_class_ap.csv

    # ── FROC (모델 메모리에 있는 동안 실행) ───────────────────────────────────
    df_froc = None
    if args.save_froc:
        test_images_dir = data_root / args.eval_split / "images"
        test_labels_dir = data_root / args.eval_split / "labels"
        if not test_labels_dir.exists():
            print(f"[FROC] labels 디렉토리 없음: {test_labels_dir}, 건너뜀")
        else:
            compute_and_save_froc(
                model,
                test_images_dir,
                test_labels_dir,
                names,
                results_dir,
                device=args.device,
                iou_thresh=args.froc_iou,
            )
            froc_path = results_dir / "froc.csv"
            if froc_path.exists():
                df_froc = pd.read_csv(froc_path)

    # ── 통합 메트릭 ───────────────────────────────────────────────────────────
    save_metrics_total(df_per_class, df_ap, results_dir, df_froc)  # metrics_total.csv

    # ── GPU 해제 ──────────────────────────────────────────────────────────────
    del metrics
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
