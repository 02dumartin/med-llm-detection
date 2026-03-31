#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUT = PROJECT_ROOT / "results" / "eval_summary.csv"

EVAL_NAME_RE = re.compile(r"^(?P<test_data>.+)_(?P<conf>[^_]+)_(?P<iou>[^_]+)$")
TEST_LABELS = {
    "fgart": "FGART",
    "ddr": "DDR",
    "eophtha": "eophtha",
    "idrid": "IDRiD",
    "diaretdb1": "DIARETDB1",
}
CLASS_LABELS = {
    "overall": "Total",
}


def infer_model_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "yolo-world" in lowered or "yolo_world" in lowered or "yoloworld" in lowered:
        return "YOLO-World"
    if "rtdetr" in lowered or "rt-detr" in lowered or "rt_detr" in lowered:
        return "RT-DETR"
    if "yolo" in lowered:
        return "YOLO"
    return "Unknown"


def infer_variant(model_name: str, classes: list[str]) -> str:
    uniq = {c.lower() for c in classes}
    if uniq <= {"lesion", "overall", "total"}:
        return "1cls"
    if {"ma", "he", "ex", "se"} & uniq:
        return "4cls"
    if model_name.endswith("_1cls"):
        return "1cls"
    return "4cls"


def load_optional_froc(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rename_map = {
        "FPPI=0.125": "FPPI 0.125",
        "FPPI=0.25": "FPPI 0.25",
        "FPPI=0.5": "FPPI 0.5",
        "FPPI=1.0": "FPPI 1",
        "FPPI=2.0": "FPPI 2",
        "FPPI=4.0": "FPPI 4",
        "FPPI=8.0": "FPPI 8",
        "avg_FROC": "AVG",
    }
    return df.rename(columns=rename_map)


def _iter_summary_files(results_root: Path) -> list[Path]:
    files: list[Path] = []
    files.extend(sorted(results_root.rglob("evaluation/summary.csv")))
    files.extend(sorted(results_root.rglob("metrics_total.csv")))
    return files


def collect_summary_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for summary_path in _iter_summary_files(results_root):
        rel = summary_path.relative_to(results_root)

        if summary_path.name == "summary.csv":
            if len(rel.parts) < 5:
                continue
            train_data = rel.parts[0]
            model_name = rel.parts[1]
            eval_name = rel.parts[2]
            metrics_total = pd.read_csv(summary_path)
        else:
            if len(rel.parts) < 4:
                continue
            train_data = rel.parts[0]
            model_name = rel.parts[1]
            eval_name = rel.parts[-2]
            metrics_total = pd.read_csv(summary_path)
            froc = load_optional_froc(summary_path.with_name("froc.csv"))
            if froc is not None:
                metrics_total = metrics_total.merge(froc, on="class", how="left")

        if metrics_total.empty:
            continue

        match = EVAL_NAME_RE.match(eval_name)
        if not match:
            continue
        key = (train_data, model_name, eval_name)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        test_data = match.group("test_data")
        conf = match.group("conf")
        iou = match.group("iou")

        variant = infer_variant(model_name, metrics_total["class"].tolist())

        for _, row in metrics_total.iterrows():
            cls = str(row["class"])
            rows.append(
                {
                    "train_data": train_data,
                    "model_name": model_name,
                    "model_family": infer_model_family(model_name),
                    "variant": variant,
                    "test_data": test_data,
                    "dataset_block": TEST_LABELS.get(test_data, test_data),
                    "conf": conf,
                    "iou": iou,
                    "class": CLASS_LABELS.get(cls, cls),
                    "AP 0.5": row.get("AP@0.5"),
                    "AP 0.5:0.95": row.get("AP@0.5:0.95"),
                    "Det acc": row.get("detection_acc"),
                    "Cls acc": row.get("classification_acc"),
                    "Ovr acc": row.get("overall_acc"),
                    "FPPI 0.125": row.get("FPPI 0.125"),
                    "FPPI 0.25": row.get("FPPI 0.25"),
                    "FPPI 0.5": row.get("FPPI 0.5"),
                    "FPPI 1": row.get("FPPI 1"),
                    "FPPI 2": row.get("FPPI 2"),
                    "FPPI 4": row.get("FPPI 4"),
                    "FPPI 8": row.get("FPPI 8"),
                    "AVG": row.get("AVG"),
                    "source_dir": str(summary_path.parent),
                }
            )
    return rows


def apply_filters(
    df: pd.DataFrame,
    train_data: str | None,
    model_name: str | None,
    model_family: str | None,
    variant: str | None,
    conf: str | None,
    iou: str | None,
    tests: list[str] | None,
) -> pd.DataFrame:
    if train_data:
        df = df[df["train_data"] == train_data]
    if model_name:
        df = df[df["model_name"] == model_name]
    if model_family:
        df = df[df["model_family"].str.lower() == model_family.lower()]
    if variant:
        df = df[df["variant"] == variant]
    if conf:
        df = df[df["conf"] == conf]
    if iou:
        df = df[df["iou"] == iou]
    if tests:
        df = df[df["test_data"].isin(tests)]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model-agnostic evaluation summary CSV")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-family", type=str, default=None, help="e.g. YOLO, RT-DETR, YOLO-World")
    parser.add_argument("--variant", choices=["4cls", "1cls"], default=None)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--iou", type=str, default=None)
    parser.add_argument("--tests", nargs="*", default=None)
    args = parser.parse_args()

    rows = collect_summary_rows(args.results_root)
    if not rows:
        raise SystemExit(f"No metrics_total.csv found under: {args.results_root}")

    df = pd.DataFrame(rows)
    df = apply_filters(
        df,
        train_data=args.train_data,
        model_name=args.model_name,
        model_family=args.model_family,
        variant=args.variant,
        conf=args.conf,
        iou=args.iou,
        tests=args.tests,
    )
    if df.empty:
        raise SystemExit("No rows left after filtering.")

    test_order = {name: idx for idx, name in enumerate(["fgart", "ddr", "eophtha", "idrid", "diaretdb1"])}
    class_order = {name: idx for idx, name in enumerate(["MA", "HE", "EX", "SE", "lesion", "Total"])}
    df["_test_order"] = df["test_data"].map(lambda x: test_order.get(x, 999))
    df["_class_order"] = df["class"].map(lambda x: class_order.get(x, 999))
    df = df.sort_values(
        by=["train_data", "model_family", "model_name", "variant", "_test_order", "_class_order", "class"],
        kind="stable",
    ).drop(columns=["_test_order", "_class_order"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[DONE] wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
