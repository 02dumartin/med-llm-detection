# src/yolo/eval.py
"""단일 실험 실행 - 로드 / 저장 / CLI 전담. 계산은 src.yolo.metrics에 위임."""

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

from src.yolo.common import (  # noqa: E402
    DATASET_ALIASES,
    DATASETS,
    canonical_dataset_name,
    class_names_for_variant,
    evaluation_dir,
    get_data_root,
    get_default_eval_split,
    infer_train_model_from_weights,
    resolve_run_root,
)
from src.yolo.metrics import (  # noqa: E402
    compute_class_accuracy,
    compute_class_prediction,
    compute_froc,
    compute_summary,
)


def save_class_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_csv(out_dir / "class-accuracy.csv", index=False)


def save_class_prediction(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_csv(out_dir / "class-prediction.csv", index=False)


def save_froc(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_csv(out_dir / "froc.csv", index=False)


def save_summary(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_csv(out_dir / "summary.csv", index=False)


def save_confusion_outputs(cm: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    labels = class_names + ["background"]
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion.csv")
    _plot_cm(cm, labels, out_dir / "confusion.png", normalize=False)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums != 0)
    _plot_cm(cm_norm, labels, out_dir / "confusion-normalized.png", normalize=True)


def load_ground_truth(
    test_labels_dir: Path,
    imgs: list[Path],
) -> dict[str, list[tuple[int, tuple[float, float, float, float]]]]:
    gt_by_stem: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for img_path in imgs:
        gt_by_stem[img_path.stem] = _load_yolo_label(test_labels_dir / f"{img_path.stem}.txt")
    return gt_by_stem


def load_images(test_images_dir: Path) -> list[Path]:
    return sorted([p for p in test_images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def load_predictions_from_txt_dir(labels_dir: Path) -> list[tuple[float, str, tuple[float, float, float, float], int]]:
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


def load_existing_eval_tables(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    accuracy_path = _first_existing([eval_dir / "class-accuracy.csv", eval_dir / "class-metrics.csv"])
    prediction_path = _first_existing([eval_dir / "class-prediction.csv", eval_dir / "class-pr.csv"])
    if accuracy_path is None or prediction_path is None:
        missing = []
        if accuracy_path is None:
            missing.append("class-accuracy.csv")
        if prediction_path is None:
            missing.append("class-prediction.csv")
        raise SystemExit(f"froc-only requires existing outputs. missing: {', '.join(missing)}")
    return pd.read_csv(accuracy_path), pd.read_csv(prediction_path)


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
    for result in results:
        stem = Path(result.path).stem
        if result.boxes is None or len(result.boxes) == 0:
            continue
        orig_h, orig_w = result.orig_shape
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for idx in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[idx]
            bbox_norm = (x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h)
            all_preds.append((float(confs[idx]), stem, bbox_norm, int(clss[idx])))
    all_preds.sort(key=lambda x: -x[0])
    return all_preds


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


def _load_yolo_label(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not path.exists():
        return []
    boxes = []
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


def _xywh_norm_to_xyxy(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> None:
    dataset_choices = sorted(set(DATASETS.keys()) | set(DATASET_ALIASES.keys()))
    parser = argparse.ArgumentParser(description="Unified evaluation runner")
    parser.add_argument("--family", type=str, default="yolo")
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
    parser.add_argument("--pred-only", action="store_true", default=False)
    parser.add_argument("--metrics-only", action="store_true", default=False)
    parser.add_argument("--froc-only", action="store_true", default=False)
    parser.add_argument("--from-preds", type=str, default=None)
    parser.add_argument("--no-froc", action="store_true", default=False)
    parser.add_argument("--save-pred-txt", action="store_true", default=False)
    parser.add_argument("--no-pred-txt", action="store_true", default=False)
    parser.add_argument("--no-plots", action="store_true", default=False)
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

    save_pred_txt = args.save_pred_txt and not args.no_pred_txt
    save_froc_flag = not args.no_froc
    if args.metrics_only:
        save_pred_txt = False
        save_froc_flag = False

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
            df_accuracy, df_prediction = load_existing_eval_tables(eval_dir)
            df_froc = None
            if save_froc_flag:
                if args.from_preds:
                    all_preds = load_predictions_from_txt_dir(Path(args.from_preds))
                else:
                    labels_dir = eval_dir / "labels"
                    if labels_dir.exists() and any(labels_dir.glob("*.txt")):
                        all_preds = load_predictions_from_txt_dir(labels_dir)
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
                    imgs = load_images(test_images_dir)
                    gt_by_stem = load_ground_truth(test_labels_dir, imgs)
                    df_froc = compute_froc(all_preds, gt_by_stem, len(imgs), class_names, args.froc_iou)
                    save_froc(df_froc, eval_dir)

            df_summary = compute_summary(df_accuracy, df_prediction, df_froc)
            save_summary(df_summary, eval_dir)
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

        df_accuracy = compute_class_accuracy(cm, class_names)
        df_prediction = compute_class_prediction(metrics, cm, class_names)
        save_class_accuracy(df_accuracy, eval_dir)
        save_class_prediction(df_prediction, eval_dir)

        if args.metrics_only:
            df_summary = compute_summary(df_accuracy, df_prediction, None)
            save_summary(df_summary, eval_dir)
            (eval_dir / ".done").touch()
            return

        all_preds = None
        if args.from_preds:
            all_preds = load_predictions_from_txt_dir(Path(args.from_preds))
        elif save_pred_txt or save_froc_flag:
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
        if save_froc_flag and test_labels_dir.exists():
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
            imgs = load_images(test_images_dir)
            gt_by_stem = load_ground_truth(test_labels_dir, imgs)
            df_froc = compute_froc(all_preds, gt_by_stem, len(imgs), class_names, args.froc_iou)
            save_froc(df_froc, eval_dir)

        df_summary = compute_summary(df_accuracy, df_prediction, df_froc)
        save_summary(df_summary, eval_dir)
        (eval_dir / ".done").touch()

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
