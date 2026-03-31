from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from src.yolo.metrics import compute_per_class_prf_ap


PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")


def write_data_yaml(
    out_path: Path,
    data_root: Path,
    names: list[str],
    val_folder: str = "val",
    test_folder: str = "test",
) -> None:
    out_path.write_text(
        "\n".join(
            [
                f"path: {data_root}",
                "train: train/images",
                f"val: {val_folder}/images",
                f"test: {test_folder}/images",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def normalize_device(device: str) -> str:
    return ",".join(part.strip() for part in device.split(",") if part.strip())


def device_parts_to_ultralytics(parts: list[str]) -> str:
    normalized = normalize_device(",".join(parts))
    if not normalized:
        raise SystemExit("--device: GPU ID가 비었습니다.")
    return normalized


def start_tensorboard(logdir: str | Path) -> subprocess.Popen | None:
    proc = subprocess.Popen(
        ["tensorboard", "--logdir", str(logdir), "--port", "6006", "--bind_all"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)
    return proc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is not None:
        proc.terminate()


def run_yolov12_repo(args, data_yaml: Path) -> None:
    if getattr(args, "repo", None) is None:
        raise SystemExit("--repo is required for backend=yolov12")
    repo = Path(args.repo)
    train_py = repo / "train.py"
    if not train_py.exists():
        raise SystemExit(f"train.py not found in repo: {train_py}")

    cmd = [
        "python",
        str(train_py),
        "--data",
        str(data_yaml),
        "--weights",
        args.model,
        "--img",
        str(args.imgsz),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--project",
        str(args.project),
        "--name",
        str(args.name),
        "--seed",
        str(args.seed),
    ]
    if getattr(args, "exist_ok", False):
        cmd.append("--exist-ok")
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo))


def save_eval_outputs(
    *,
    project_dir: str | Path,
    results_dir: str | Path,
    run_name: str,
    data_yaml: Path,
    eval_split: str,
    iou: float,
    metrics_mode: str = "multi",
    predict_after: bool = False,
    data_root: str | Path | None = None,
    conf: float = 0.25,
) -> None:
    best = Path(project_dir) / run_name / "weights" / "best.pt"
    model = YOLO(str(best))
    eval_dir = Path(results_dir) / run_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=str(data_yaml),
        split=eval_split,
        iou=iou,
        plots=True,
        project=str(Path(results_dir)),
        name=run_name,
        exist_ok=True,
    )

    cm = metrics.confusion_matrix.matrix
    nc = cm.shape[0] - 1
    total_gt = cm[:nc, :].sum()
    total_detected = cm[:nc, :nc].sum()
    detection_acc = total_detected / total_gt if total_gt > 0 else 0

    if metrics_mode == "single":
        metric_rows = {
            "metric": ["mAP50", "mAP50-95", "detection_acc"],
            "value": [metrics.box.map50, metrics.box.map, detection_acc],
        }
    else:
        tp_correct_class = cm[:nc, :nc].diagonal().sum()
        classification_acc = tp_correct_class / total_detected if total_detected > 0 else 0
        overall_acc = tp_correct_class / total_gt if total_gt > 0 else 0
        metric_rows = {
            "metric": ["mAP50", "mAP50-95", "detection_acc", "classification_acc", "overall_acc"],
            "value": [metrics.box.map50, metrics.box.map, detection_acc, classification_acc, overall_acc],
        }

    pd.DataFrame(metric_rows).to_csv(eval_dir / "metrics.csv", index=False)

    names = [model.names[i] for i in sorted(model.names.keys())]
    df_prf_ap = compute_per_class_prf_ap(metrics, cm, names)
    df_prf_ap.to_csv(eval_dir / "per_class_ap.csv", index=False)

    if predict_after:
        if data_root is None:
            raise SystemExit("predict_after requires data_root")
        test_images = Path(data_root) / "test" / "images"
        pred_dir = eval_dir / "prediction"
        pred_dir.mkdir(parents=True, exist_ok=True)
        model.predict(
            source=str(test_images),
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(pred_dir.parent),
            name="prediction",
            exist_ok=True,
            conf=conf,
            iou=iou,
        )
