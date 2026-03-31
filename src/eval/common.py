from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")

DATASETS = {
    "fgart": {
        "root_4cls": PROJECT_ROOT / "data" / "FGART_yolo_4cls",
        "root_1cls": PROJECT_ROOT / "data" / "FGART_yolo_1cls",
        "eval_split": "val",
        "overlay": "fgart",
    },
    "ddr": {
        "root_4cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_yolo_4cls"),
        "root_1cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DDR_crop_yolo_1cls"),
        "eval_split": "val",
        "overlay": "ddr",
    },
    "idrid": {
        "root_4cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_4cls"),
        "root_1cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/IDRiD_yolo_1cls"),
        "eval_split": "test",
        "overlay": None,
    },
    "eophtha": {
        "root_4cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_4cls"),
        "root_1cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/Eophtha_yolo_1cls"),
        "eval_split": "val",
        "overlay": None,
    },
    "diaretdb1": {
        "root_4cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DIARETDB1_yolo_4cls"),
        "root_1cls": Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-data/DIARETDB1_yolo_1cls"),
        "eval_split": "test",
        "overlay": None,
    },
}

DATASET_ALIASES = {
    "e-optha": "eophtha",
}

CLASS_COLORS = {
    "MA": "#00FF00",
    "HE": "#FF0000",
    "EX": "#FFFF00",
    "SE": "#00FFFF",
    "lesion": "#FF00FF",
}


def canonical_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def get_dataset_config(name: str) -> dict:
    key = canonical_dataset_name(name)
    if key not in DATASETS:
        raise KeyError(f"unknown dataset: {name}")
    return DATASETS[key]


def get_data_root(name: str, variant: str) -> Path:
    cfg = get_dataset_config(name)
    return cfg["root_4cls"] if variant == "4cls" else cfg["root_1cls"]


def get_default_eval_split(name: str) -> str:
    cfg = get_dataset_config(name)
    return cfg["eval_split"]


def get_overlay_type(name: str) -> str | None:
    cfg = get_dataset_config(name)
    return cfg["overlay"]


def class_names_for_variant(variant: str) -> list[str]:
    if variant == "4cls":
        return ["MA", "HE", "EX", "SE"]
    return ["lesion"]


def class_colors_for_variant(variant: str, class_names: list[str]) -> list[str]:
    if variant == "4cls":
        return [CLASS_COLORS[n] for n in class_names]
    return [CLASS_COLORS["lesion"]]


def infer_train_model_from_weights(weights: Path) -> tuple[str | None, str | None]:
    parts = weights.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if len(parts) >= idx + 4:
            return parts[idx + 1], parts[idx + 2]
    return None, None


def eval_name(test_data: str, conf: float, iou: float) -> str:
    return f"{canonical_dataset_name(test_data)}_{conf}_{iou}"


def resolve_run_root(
    results_root_arg: str | None,
    train_data: str,
    model_name: str,
    test_data: str,
    conf: float,
    iou: float,
) -> Path:
    name = eval_name(test_data, conf, iou)
    if results_root_arg is not None:
        return Path(results_root_arg) / model_name / name
    return PROJECT_ROOT / "results" / train_data / model_name / name


def evaluation_dir(run_root: Path) -> Path:
    return run_root / "evaluation"


def overlay_dir(run_root: Path) -> Path:
    return run_root / "overlay"
