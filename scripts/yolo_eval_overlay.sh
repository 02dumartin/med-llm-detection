#!/usr/bin/env bash
# yolo_eval.py (metrics) -> yolo_overlay.py (overlay) 순차 실행
# Usage:
#   bash yolo_eval_with_overlay.sh \
#     --weights <path> \
#     --variant <4cls|1cls> \
#     --train-data <fgart|ddr|...> \
#     --model-name <name> \
#     --test-data <fgart|ddr|...> \
#     --conf <float> \
#     --iou <float> \
#     --device <int> \
#     [--results-dir <path>] \
#     [--save-gt-overlay]
set -euo pipefail

BASE=/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection
VENV_PY="$BASE/.venv/bin/python"
EVAL_PY="$BASE/src/eval/yolo_eval.py"
OVERLAY_PY="$BASE/src/eval/yolo_overlay.py"

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
WEIGHTS=""
VARIANT="4cls"
TRAIN_DATA=""
MODEL_NAME=""
TEST_DATA=""
CONF="0.25"
IOU="0.7"
DEVICE="2"
RESULTS_DIR=""
SAVE_GT_OVERLAY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights)      WEIGHTS="$2";      shift 2 ;;
        --variant)      VARIANT="$2";      shift 2 ;;
        --train-data)   TRAIN_DATA="$2";   shift 2 ;;
        --model-name)   MODEL_NAME="$2";   shift 2 ;;
        --test-data)    TEST_DATA="$2";    shift 2 ;;
        --conf)         CONF="$2";         shift 2 ;;
        --iou)          IOU="$2";          shift 2 ;;
        --device)       DEVICE="$2";       shift 2 ;;
        --results-dir)  RESULTS_DIR="$2";  shift 2 ;;
        --save-gt-overlay) SAVE_GT_OVERLAY=true; shift ;;
        *) echo "[ERROR] 알 수 없는 인자: $1"; exit 1 ;;
    esac
done

if [[ -z "$WEIGHTS" || -z "$TEST_DATA" ]]; then
    echo "[ERROR] --weights, --test-data 는 필수입니다."
    exit 1
fi

# ── 공통 인자 구성 ──────────────────────────────────────────────────────────────
COMMON_ARGS=(
    --weights     "$WEIGHTS"
    --variant     "$VARIANT"
    --test-data   "$TEST_DATA"
    --conf        "$CONF"
    --iou         "$IOU"
    --device      "$DEVICE"
)
[[ -n "$TRAIN_DATA"   ]] && COMMON_ARGS+=(--train-data  "$TRAIN_DATA")
[[ -n "$MODEL_NAME"   ]] && COMMON_ARGS+=(--model-name  "$MODEL_NAME")
[[ -n "$RESULTS_DIR"  ]] && COMMON_ARGS+=(--results-dir "$RESULTS_DIR")

# ── 1단계: val + metrics (GPU 사용 후 프로세스 종료로 VRAM 완전 해제) ───────────
echo "=== [1/2] yolo_eval.py (metrics) ==="
"$VENV_PY" "$EVAL_PY" "${COMMON_ARGS[@]}"
echo "=== metrics 완료 ==="

# ── 2단계: overlay (새 프로세스 → 깨끗한 CUDA context) ──────────────────────────
echo "=== [2/2] yolo_overlay.py (overlay) ==="
OVERLAY_ARGS=("${COMMON_ARGS[@]}")
$SAVE_GT_OVERLAY && OVERLAY_ARGS+=(--save-gt-overlay)

"$VENV_PY" "$OVERLAY_PY" "${OVERLAY_ARGS[@]}"
echo "=== overlay 완료 ==="
