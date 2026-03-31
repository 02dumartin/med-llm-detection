#!/usr/bin/env bash
set -euo pipefail

BASE="/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection"
DEVICE="7"
CONF=""
IOU=""

usage() {
    cat <<EOF
Usage:
  bash scripts/cross_eval_stage1.sh --conf <float> --iou <float> [--device <gpu>]

Stage 1:
  Single-threshold full evaluation run
  -> metrics + FROC + prediction txt

Examples:
  bash scripts/cross_eval_stage1.sh --device 7 --conf 0.01 --iou 0.5
  bash scripts/cross_eval_stage1.sh --conf 0.05 --iou 0.5
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --conf) CONF="$2"; shift 2 ;;
        --iou) IOU="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$CONF" || -z "$IOU" ]]; then
    echo "[ERROR] --conf and --iou are required." >&2
    usage
    exit 1
fi

cd "$BASE"

echo "[STAGE1] full evaluation start"
echo "         device: $DEVICE"
echo "         conf  : $CONF"
echo "         iou   : $IOU"
echo "         output: metrics + FROC + prediction txt"

bash scripts/yolo_eval_batch.sh \
    --device "$DEVICE" \
    --conf "$CONF" \
    --iou "$IOU" \
    --save-froc \
    --save-pred-txt

echo "[DONE] Stage 1 finished for conf=$CONF, iou=$IOU."
