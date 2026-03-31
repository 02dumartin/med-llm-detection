#!/usr/bin/env bash
set -euo pipefail

BASE="/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection"
VENV_PY="$BASE/.venv/bin/python"
DEVICE="7"

usage() {
    cat <<EOF
Usage:
  bash scripts/cross_eval_stage2.sh [--device <gpu>]

Prerequisite:
  notebook/overlay.ipynb 에서 conf=0.05, conf=0.01 결과의 저장된 prediction txt를 데이터셋별로 검증 완료

Stage 2:
  1. conf=0.05, iou=0.5 -> overlay + report
  2. conf=0.01, iou=0.5 -> overlay + report
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

cd "$BASE"

echo "[STAGE2] 0.05/0.5 overlay"
bash scripts/yolo_overlay_batch.sh --device "$DEVICE" --conf 0.05 --iou 0.5 --use-saved-pred-txt

echo "[STAGE2] 0.05/0.5 report"
"$VENV_PY" scripts/export_eval_reports.py \
    --conf 0.05 \
    --iou 0.5 \
    --reports-dir reports/conf_0.05_iou_0.5

echo "[STAGE2] 0.01/0.5 overlay"
bash scripts/yolo_overlay_batch.sh --device "$DEVICE" --conf 0.01 --iou 0.5 --use-saved-pred-txt

echo "[STAGE2] 0.01/0.5 report"
"$VENV_PY" scripts/export_eval_reports.py \
    --conf 0.01 \
    --iou 0.5 \
    --reports-dir reports/conf_0.01_iou_0.5

echo "[DONE] Stage 2 finished."
