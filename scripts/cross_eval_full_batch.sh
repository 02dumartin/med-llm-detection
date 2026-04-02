#!/usr/bin/env bash
set -euo pipefail

BASE="/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection"
VENV_PY="$BASE/.venv/bin/python"
DEVICE="7"
IOU="0.5"
FIRST_CONF="0.05"
SECOND_CONF="0.01"
FORCE=false
DRY_RUN=false
ONLY_MODELS=""
ONLY_TESTS=""

usage() {
    cat <<EOF
Usage:
  bash scripts/cross_eval_full_batch.sh [options]

Runs two full cross-eval rounds in sequence:
  1. conf=0.05, iou=0.5 -> evaluation/report -> overlay
  2. conf=0.01, iou=0.5 -> evaluation/report -> overlay

Options:
  --device <gpu>           GPU id (default: ${DEVICE})
  --iou <float>            IoU threshold (default: ${IOU})
  --force                  rerun completed evaluation/overlay jobs
  --only-models <csv>      subset, e.g. fgart_4cls,merge_1cls
  --only-tests <csv>       subset, e.g. fgart,ddr,eophtha
  --dry-run                print planned commands only
  Example:
  bash scripts/cross_eval_full_batch.sh --device 7 --force
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --iou) IOU="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        --only-models) ONLY_MODELS="$2"; shift 2 ;;
        --only-tests) ONLY_TESTS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

cd "$BASE"

run_round() {
    local conf="$1"
    local reports_dir="$BASE/reports/conf_${conf}_iou_${IOU}"
    local eval_args=(
        --device "$DEVICE"
        --conf "$conf"
        --iou "$IOU"
        --save-froc
        --save-pred-txt
    )
    local overlay_args=(
        --device "$DEVICE"
        --conf "$conf"
        --iou "$IOU"
        --use-saved-pred-txt
    )

    if $FORCE; then
        eval_args+=(--force)
        overlay_args+=(--force)
    fi
    if [[ -n "$ONLY_MODELS" ]]; then
        eval_args+=(--only-models "$ONLY_MODELS")
        overlay_args+=(--only-models "$ONLY_MODELS")
    fi
    if [[ -n "$ONLY_TESTS" ]]; then
        eval_args+=(--only-tests "$ONLY_TESTS")
        overlay_args+=(--only-tests "$ONLY_TESTS")
    fi
    if $DRY_RUN; then
        eval_args+=(--dry-run)
        overlay_args+=(--dry-run)
    fi

    echo ""
    echo "============================================================"
    echo "[ROUND] conf=$conf iou=$IOU"
    echo "  1) evaluation"
    echo "  2) report export"
    echo "  3) overlay from saved prediction txt"
    echo "  device      : $DEVICE"
    echo "  force       : $FORCE"
    echo "  only_models : ${ONLY_MODELS:-<all>}"
    echo "  only_tests  : ${ONLY_TESTS:-<all>}"
    echo "  reports_dir : $reports_dir"
    echo "============================================================"

    bash scripts/yolo_eval_batch.sh "${eval_args[@]}"

    if $DRY_RUN; then
        echo "[DRY-RUN] report export"
        echo "          $VENV_PY -m src.yolo.reporting.reports --conf $conf --iou $IOU --reports-dir $reports_dir"
    else
        "$VENV_PY" -m src.yolo.reporting.reports \
            --conf "$conf" \
            --iou "$IOU" \
            --reports-dir "$reports_dir"
    fi

    bash scripts/yolo_overlay_batch.sh "${overlay_args[@]}"
}

run_round "$FIRST_CONF"
run_round "$SECOND_CONF"

echo ""
echo "[DONE] cross_eval_full_batch finished."
echo "       rounds: conf=${FIRST_CONF}, conf=${SECOND_CONF} (iou=${IOU})"
