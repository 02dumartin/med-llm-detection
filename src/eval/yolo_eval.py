#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.common import (  # noqa: E402
    DATASET_ALIASES,
    DATASETS,
    canonical_dataset_name,
    class_names_for_variant,
    eval_name,
    evaluation_dir,
    get_data_root,
    get_default_eval_split,
    infer_train_model_from_weights,
    resolve_run_root,
)
from src.eval.metrics_utils import compute_per_class_prf_ap  # noqa: E402


FPPI_POINTS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def write_data_yaml(out_path: Path, data_root: Path, names: list[str], val_split: str) -> None:
    out_path.write_text(
        "\n".join(
            [
                f"path: {data_root}",
                "train: train/images",
                f"val: {val_split}/images",
                "test: test/images",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


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
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_outputs(cm: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    labels = class_names + ["background"]
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion.csv")
    _plot_cm(cm, labels, out_dir / "confusion.png", normalize=False)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)
    _plot_cm(cm_norm, labels, out_dir / "confusion-normalized.png", normalize=True)


def compute_class_metrics(cm: np.ndarray, class_names: list[str]) -> list[dict]:
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
    return rows

# CSV save function
def save_class_prediction(metrics, cm: np.ndarray, class_names: list[str], out_dir: Path) -> pd.DataFrame:
    df = compute_per_class_prf_ap(metrics, cm, class_names)
    df.to_csv(out_dir / "class-prediction.csv", index=False)
    return df

def save_overall_csv(df_class: pd.DataFrame, out_dir: Path) -> None:
    row = df_class[df_class["class"] == "overall"]
    if row.empty:
        return
    r = row.iloc[0]
    pd.DataFrame(
        {
            "metric": ["detection_acc", "classification_acc", "overall_acc"],
            "value": [r["detection_acc"], r["classification_acc"], r["overall_acc"]],
        }
    ).to_csv(out_dir / "overall.csv", index=False)


def save_summary(
    df_class_metrics: pd.DataFrame,
    df_class_pr: pd.DataFrame,
    out_dir: Path,
    df_froc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = df_class_metrics.merge(df_class_pr, on="class", how="left")
    if df_froc is not None:
        froc_cols = [c for c in df_froc.columns if c != "n_gt"]
        merged = merged.merge(df_froc[froc_cols], on="class", how="left")
    merged.to_csv(out_dir / "summary.csv", index=False)
    return merged


def _xywh_norm_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
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


def _load_yolo_label(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not path.exists():
        return []
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        bbox = _xywh_norm_to_xyxy(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        boxes.append((cls, bbox))
    return boxes


def collect_predictions(
    model,
    test_images_dir: Path,
    device: str,
    conf_thresh: float,
    pred_iou: float,
    save_pred_txt: bool,
    eval_dir: Path | None,
) -> list[tuple[float, str, tuple[float, float, float, float], int]]:
    predict_kwargs = dict(
        source=str(test_images_dir),
        conf=conf_thresh,
        iou=pred_iou,
        device=device,
        save=False,
        verbose=False,
    )
    if save_pred_txt:
        if eval_dir is None:
            raise ValueError("eval_dir must be set when save_pred_txt=True")
        eval_dir.mkdir(parents=True, exist_ok=True)
        predict_kwargs.update(
            save_txt=True,
            save_conf=True,
            project=str(eval_dir.parent),
            name=eval_dir.name,
            exist_ok=True,
        )

    results = model.predict(**predict_kwargs)

    all_preds: list[tuple[float, str, tuple[float, float, float, float], int]] = []
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
    return all_preds


def load_predictions_from_txt_dir(
    labels_dir: Path,
) -> list[tuple[float, str, tuple[float, float, float, float], int]]:
    all_preds: list[tuple[float, str, tuple[float, float, float, float], int]] = []
    if not labels_dir.exists():
        return all_preds
    for txt_path in sorted(labels_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        stem = txt_path.stem
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            bbox = _xywh_norm_to_xyxy(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            score = float(parts[5]) if len(parts) >= 6 else 1.0
            all_preds.append((score, stem, bbox, cls))
    all_preds.sort(key=lambda x: -x[0])
    return all_preds


def compute_and_save_froc(
    test_images_dir: Path,
    test_labels_dir: Path,
    class_names: list[str],
    out_dir: Path,
    all_preds: list[tuple[float, str, tuple[float, float, float, float], int]],
    iou_thresh: float = 0.3,
) -> pd.DataFrame:
    imgs = sorted([p for p in test_images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    num_images = len(imgs)
    if num_images == 0:
        raise RuntimeError(f"no images found: {test_images_dir}")

    gt_by_stem: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for img_path in imgs:
        gt_by_stem[img_path.stem] = _load_yolo_label(test_labels_dir / f"{img_path.stem}.txt")

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
        tp = 0
        fp = 0
        curve: list[tuple[float, float]] = []

        for _, stem, bbox, cls in all_preds:
            if target_cls is not None and cls != target_cls:
                continue
            gts = gt_by_stem.get(stem, [])
            best_iou = 0.0
            best_idx = -1
            for idx, (gt_cls, gt_bbox) in enumerate(gts):
                if target_cls is not None and gt_cls != target_cls:
                    continue
                if idx in matched[stem]:
                    continue
                iou = _iou_xyxy(bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
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

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "froc.csv", index=False)
    return df


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def load_existing_eval_tables(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_metrics_path = _first_existing([eval_dir / "class-metrics.csv", eval_dir / "per_class_metrics.csv"])
    class_pr_path = _first_existing([eval_dir / "class-pr.csv", eval_dir / "per_class_ap.csv"])
    if class_metrics_path is None or class_pr_path is None:
        missing = []
        if class_metrics_path is None:
            missing.append("class-metrics.csv")
        if class_pr_path is None:
            missing.append("class-pr.csv")
        raise SystemExit(f"froc-only requires existing evaluation outputs. missing: {', '.join(missing)}")
    return pd.read_csv(class_metrics_path), pd.read_csv(class_pr_path)


def main() -> None:
    dataset_choices = sorted(set(DATASETS.keys()) | set(DATASET_ALIASES.keys()))
    parser = argparse.ArgumentParser(description="Unified evaluation runner")
    parser.add_argument("--family", type=str, default="yolo", help="model family, currently only yolo is supported")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", choices=["4cls", "1cls"], default="4cls")
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--test-data", choices=dataset_choices, required=True)
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="results root. if set, outputs are stored under <results-dir>/<model>/<test_conf_iou>",
    )
    parser.add_argument("--pred-only", action="store_true", default=False)
    parser.add_argument("--metrics-only", action="store_true", default=False)
    parser.add_argument("--froc-only", action="store_true", default=False)
    parser.add_argument("--from-preds", type=str, default=None, help="reuse saved yolo txt from this labels directory")
    parser.add_argument("--no-froc", action="store_true", default=False, help="skip froc computation")
    parser.add_argument("--no-pred-txt", action="store_true", default=False, help="do not save prediction txt")
    parser.add_argument("--no-plots", action="store_true", default=False, help="disable ultralytics plot outputs")
    parser.add_argument("--save-froc", action="store_true", default=False, help="deprecated alias: enable froc")
    parser.add_argument("--save-pred-txt", action="store_true", default=False, help="deprecated alias: enable txt")
    parser.add_argument("--froc-iou", type=float, default=0.3)
    args = parser.parse_args()

    if args.family.lower() != "yolo":
        raise SystemExit(f"unsupported family: {args.family}")
    if args.pred_only and (args.metrics_only or args.froc_only):
        raise SystemExit("--pred-only cannot be combined with --metrics-only or --froc-only")
    if args.metrics_only and args.froc_only:
        raise SystemExit("--metrics-only cannot be combined with --froc-only")

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
    data_yaml = data_root / f"{test_data}_{args.variant}.yaml"
    write_data_yaml(data_yaml, data_root, class_names, eval_split)

    run_root = resolve_run_root(args.results_dir, train_data, model_name, test_data, args.conf, args.iou)
    eval_dir = evaluation_dir(run_root)
    eval_dir.mkdir(parents=True, exist_ok=True)

    save_pred_txt = args.save_pred_txt or (not args.no_pred_txt)
    save_froc = args.save_froc or (not args.no_froc)
    if args.metrics_only:
        save_pred_txt = False
        save_froc = False
    if args.pred_only:
        save_pred_txt = True

    test_images_dir = data_root / eval_split / "images"
    test_labels_dir = data_root / eval_split / "labels"
    model = YOLO(str(weights))

    try:
        if args.pred_only:
            collect_predictions(
                model,
                test_images_dir,
                device=args.device,
                conf_thresh=args.conf,
                pred_iou=args.iou,
                save_pred_txt=True,
                eval_dir=eval_dir,
            )
            (eval_dir / "labels.done").touch()
            return

        if args.froc_only:
            df_class, df_pr = load_existing_eval_tables(eval_dir)
            df_froc = None
            if save_froc:
                if args.from_preds:
                    all_preds = load_predictions_from_txt_dir(Path(args.from_preds))
                else:
                    default_labels_dir = eval_dir / "labels"
                    if default_labels_dir.exists() and any(default_labels_dir.glob("*.txt")):
                        all_preds = load_predictions_from_txt_dir(default_labels_dir)
                    else:
                        all_preds = collect_predictions(
                            model,
                            test_images_dir,
                            device=args.device,
                            conf_thresh=args.conf,
                            pred_iou=args.iou,
                            save_pred_txt=save_pred_txt,
                            eval_dir=eval_dir if save_pred_txt else None,
                        )
                        if save_pred_txt:
                            (eval_dir / "labels.done").touch()
                if test_labels_dir.exists():
                    df_froc = compute_and_save_froc(
                        test_images_dir,
                        test_labels_dir,
                        class_names,
                        eval_dir,
                        all_preds=all_preds,
                        iou_thresh=args.froc_iou,
                    )
            save_summary(df_class, df_pr, eval_dir, df_froc)
            (eval_dir / ".done").touch()
            return

        metrics = model.val(
            data=str(data_yaml),
            split=eval_split,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            plots=not args.no_plots,
            project=str(run_root),
            name="evaluation",
            exist_ok=True,
            save_json=False,
        )

        cm = metrics.confusion_matrix.matrix
        save_confusion_outputs(cm, class_names, eval_dir)
        df_class = pd.DataFrame(compute_class_metrics(cm, class_names))
        df_class.to_csv(eval_dir / "class-metrics.csv", index=False)
        save_overall_csv(df_class, eval_dir)
        df_pr = save_class_pr(metrics, cm, class_names, eval_dir)

        if args.metrics_only:
            save_summary(df_class, df_pr, eval_dir, None)
            (eval_dir / ".done").touch()
            return

        all_preds: list[tuple[float, str, tuple[float, float, float, float], int]] | None = None
        if args.from_preds:
            all_preds = load_predictions_from_txt_dir(Path(args.from_preds))
        elif save_pred_txt or save_froc:
            all_preds = collect_predictions(
                model,
                test_images_dir,
                device=args.device,
                conf_thresh=args.conf,
                pred_iou=args.iou,
                save_pred_txt=save_pred_txt,
                eval_dir=eval_dir if save_pred_txt else None,
            )
            if save_pred_txt:
                (eval_dir / "labels.done").touch()

        df_froc = None
        if save_froc and test_labels_dir.exists():
            if all_preds is None:
                all_preds = collect_predictions(
                    model,
                    test_images_dir,
                    device=args.device,
                    conf_thresh=args.conf,
                    pred_iou=args.iou,
                    save_pred_txt=False,
                    eval_dir=None,
                )
            df_froc = compute_and_save_froc(
                test_images_dir,
                test_labels_dir,
                class_names,
                eval_dir,
                all_preds=all_preds,
                iou_thresh=args.froc_iou,
            )

        save_summary(df_class, df_pr, eval_dir, df_froc)
        (eval_dir / ".done").touch()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
