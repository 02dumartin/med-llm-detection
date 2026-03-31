#!/usr/bin/env bash
set -euo pipefail

BASE="/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection"
DEVICE="7"
IOU="0.5"
FIRST_CONF="0.05"
SECOND_CONF="0.01"

usage() {
    cat <<EOF
Usage:
  bash scripts/cross_eval_full_batch.sh [--device <gpu>] [--iou <float>]

Runs two full cross-eval rounds in sequence:
  1. conf=0.05, iou=0.5 -> evaluation/report -> overlay
  2. conf=0.01, iou=0.5 -> evaluation/report -> overlay

Example:
  bash scripts/cross_eval_full_batch.sh --device 7
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --iou) IOU="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

cd "$BASE"

run_round() {
    local conf="$1"

    echo ""
    echo "============================================================"
    echo "[ROUND] conf=$conf iou=$IOU"
    echo "  1) evaluation + report"
    echo "  2) overlay from saved prediction txt"
    echo "============================================================"

    bash scripts/cross_eval_report_batch.sh \
        --device "$DEVICE" \
        --conf "$conf" \
        --iou "$IOU"

    bash scripts/yolo_overlay_batch.sh \
        --device "$DEVICE" \
        --conf "$conf" \
        --iou "$IOU" \
        --use-saved-pred-txt
}

# run_round "$FIRST_CONF"
run_round "$SECOND_CONF"

echo ""
echo "[DONE] cross_eval_full_batch finished."
