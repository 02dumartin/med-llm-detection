#!/usr/bin/env bash
set -euo pipefail

BASE="/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection"
VENV_PY="$BASE/.venv/bin/python"
DEVICE="7"
CONF="0.01"
IOU="0.5"
REPORTS_DIR=""

usage() {
    cat <<EOF
Usage:
  bash scripts/cross_eval_report_batch.sh [--device <gpu>] [--conf <float>] [--iou <float>] [--reports-dir <path>]

Runs all 35 cross-eval jobs and then exports aggregate CSV/XLSX reports.
Outputs:
  1. prediction txt
  2. evaluation CSVs
  3. all_eval_summary.csv
  4. master.xlsx and per-train workbooks

Example:
  bash scripts/cross_eval_report_batch.sh --device 7 --conf 0.01 --iou 0.5
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --conf) CONF="$2"; shift 2 ;;
        --iou) IOU="$2"; shift 2 ;;
        --reports-dir) REPORTS_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$REPORTS_DIR" ]]; then
    REPORTS_DIR="$BASE/reports/conf_${CONF}_iou_${IOU}"
fi

cd "$BASE"

echo "[BATCH] evaluation start"
echo "        device      : $DEVICE"
echo "        conf / iou  : $CONF / $IOU"
echo "        reports_dir : $REPORTS_DIR"

bash scripts/yolo_eval_batch.sh \
    --device "$DEVICE" \
    --conf "$CONF" \
    --iou "$IOU" \
    --save-froc \
    --save-pred-txt

echo "[BATCH] exporting aggregate reports"
"$VENV_PY" -m src.yolo.reporting.reports \
    --conf "$CONF" \
    --iou "$IOU" \
    --reports-dir "$REPORTS_DIR"

echo "[DONE] cross_eval_report_batch finished."
