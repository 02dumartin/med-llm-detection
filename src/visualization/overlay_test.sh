#!/usr/bin/env bash
# Batch GT overlay runner for 4cls test split.
# Runs:
#   fgart, ddr, eophtha, idrid, diaretdb1
# using src/visualization/gt_overlay.py
set -euo pipefail

BASE="/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection"
VENV_PY="$BASE/.venv/bin/python"
GT_OVERLAY_PY="$BASE/src/visualization/gt_overlay.py"

VARIANT="4cls"
SPLIT="test"
GT_SOURCE="coco"
OUT_ROOT="${BASE}/results/gt_overlay"
LIMIT=""
LEGEND=false
DRY_RUN=false
ONLY_DATASETS=""

DATASETS=("fgart" "ddr" "eophtha" "idrid" "diaretdb1")

usage() {
    cat <<EOF
Usage:
  bash scripts/overlay_test.sh [options]

Options:
  --gt-source <coco|txt>     GT annotation source (default: ${GT_SOURCE})
  --split <name>             split to render (default: ${SPLIT})
  --out-root <dir>           output root (default: ${OUT_ROOT})
  --limit <n>                render only first n images per dataset
  --legend                   draw legend on each image
  --only-datasets <csv>      subset, e.g. fgart,ddr,idrid
  --dry-run                  print commands without executing
  -h, --help                 show this help

Examples:
  bash scripts/overlay_test.sh
  bash scripts/overlay_test.sh --gt-source txt --legend
  bash scripts/overlay_test.sh --only-datasets fgart,ddr --limit 10
EOF
}

contains_csv_item() {
    local csv="$1" item="$2"
    if [[ -z "$csv" ]]; then
        return 0
    fi
    local token
    IFS=',' read -r -a tokens <<< "$csv"
    for token in "${tokens[@]}"; do
        if [[ "$token" == "$item" ]]; then
            return 0
        fi
    done
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gt-source) GT_SOURCE="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --legend) LEGEND=true; shift ;;
        --only-datasets) ONLY_DATASETS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "$GT_SOURCE" != "coco" && "$GT_SOURCE" != "txt" ]]; then
    echo "[ERROR] --gt-source must be one of: coco, txt" >&2
    exit 1
fi

if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] Python not found: $VENV_PY" >&2
    exit 1
fi

if [[ ! -f "$GT_OVERLAY_PY" ]]; then
    echo "[ERROR] Script not found: $GT_OVERLAY_PY" >&2
    exit 1
fi

run_dataset() {
    local dataset="$1"
    local out_dir="$OUT_ROOT/$dataset/$VARIANT/$GT_SOURCE"
    local cmd=(
        "$VENV_PY" "$GT_OVERLAY_PY"
        --dataset "$dataset"
        --variant "$VARIANT"
        --gt-source "$GT_SOURCE"
        --split "$SPLIT"
        --out-dir "$out_dir"
    )

    if [[ -n "$LIMIT" ]]; then
        cmd+=(--limit "$LIMIT")
    fi
    if $LEGEND; then
        cmd+=(--legend)
    fi

    echo ""
    echo "============================================================"
    echo "[RUN] dataset=${dataset} variant=${VARIANT} source=${GT_SOURCE} split=${SPLIT}"
    echo "  out_dir: ${out_dir}"
    printf '  cmd    :'
    printf ' %q' "${cmd[@]}"
    echo ""
    echo "============================================================"

    if $DRY_RUN; then
        return 0
    fi

    "${cmd[@]}"
}

for dataset in "${DATASETS[@]}"; do
    if ! contains_csv_item "$ONLY_DATASETS" "$dataset"; then
        continue
    fi
    run_dataset "$dataset"
done

echo ""
echo "[DONE] 4cls GT overlay batch completed."
