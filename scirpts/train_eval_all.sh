#!/usr/bin/env bash
# ============================================================
# train_eval_all.sh
# FGART, DDR, Merge 각 4cls+1cls 학습
#
# 학습 구조:
#   Round 1: DDR 4cls (GPU 5)     +  FGART 4cls (GPU 6)  ← 병렬
#   Round 2: DDR 1cls (GPU 5)    +  FGART 1cls (GPU 6)  ← 병렬
#   Round 3: Merge 4cls (GPU 5)   +  Merge 1cls (GPU 6)  ← 병렬
#   Round 4: Eophtha 4cls (GPU 5) +  Eophtha 1cls (GPU 6)  ← 병렬
#   Round 5: Eophtha 4cls (GPU 5) +  Eophtha 1cls (GPU 6) ← 병렬
#
# DDR/Merge: imgsz=1920, batch=4 (스크립트 기본값)
# FGART: imgsz=1280, batch=8
#
# 현재: train만 실행 (test/eval 비활성화)
#
# 사용법:
#   bash scirpts/train_eval_all.sh                  # train만
#   bash scirpts/train_eval_all.sh --skip-train    # 학습 건너뛰기
#   # eval 실행: 아래 2단계 각주 해제 후 --eval 옵션 추가
# ============================================================

set -euo pipefail

# 프로젝트 루트 (스크립트: <repo>/scirpts/train_eval_all.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$BASE/logs"
TMUX_LOG_DIR="$BASE/tmux_log"

mkdir -p "$LOG_DIR"
mkdir -p "$TMUX_LOG_DIR"

# tmux 로그: 스크립트 전체 출력 → tmux_log/
TMUX_LOG="$TMUX_LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$TMUX_LOG") 2>&1
echo "[tmux_log] $TMUX_LOG"

VENV_PY="$BASE/.venv/bin/python"

FGART_PY="$BASE/scirpts/fgart_yolo12.py"
DDR4_PY="$BASE/scirpts/ddr_yolo12.py"
DDR1_PY="$BASE/scirpts/ddr_yolo12_1cls.py"
MERGE4_PY="$BASE/scirpts/merge_yolo12.py"
MERGE1_PY="$BASE/scirpts/merge_yolo12_1cls.py"
EOPTHA4_PY="$BASE/scirpts/eophtha_yolo12.py"
EOPTHA1_PY="$BASE/scirpts/eophtha_yolo12_1cls.py"
EVAL_PY="$BASE/src/eval/yolo_eval.py"
OVERLAY_PY="$BASE/src/eval/yolo_overlay.py"

# ── GPU 설정 ────────────────────────────────────────────────
GPU_DDR=5             # DDR, Merge 학습용 단일 GPU
GPU_FGART=6            # FGART 학습용 멀티 GPU (DDP)
GPU_EVAL_FGART=5      # FGART 모델 eval GPU
GPU_EVAL_DDR=6        # DDR 모델 eval GPU

# ── eval 파라미터 ────────────────────────────────────────────
CONF="${CONF:-0.25}"
IOU="${IOU:-0.5}"

# ── 플래그 파싱 ──────────────────────────────────────────────
SKIP_TRAIN=false
DO_EVAL=false   # train만 실행 (test/eval 비활성화) → --eval 시 eval+overlay 실행
DO_FROC=true
DO_OVERLAY=true

for arg in "$@"; do
    case "$arg" in
        --skip-train) SKIP_TRAIN=true ;;
        --eval)       DO_EVAL=true ;;
        --no-froc)    DO_FROC=false ;;
        --no-overlay) DO_OVERLAY=false ;;
        *) echo "[WARN] 알 수 없는 옵션: $arg" ;;
    esac
done

# ── 공통 함수 ────────────────────────────────────────────────
log() { echo ""; echo "══════════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════════"; }

# 백그라운드 학습: Python 프로세스만 띄움 ($! = 학습 PID, 파이프로 awk 등 추가 PID 없음)
# stdout/stderr 전부 logs/에 기록 (tmux_log에는 안 남김)
# PID는 전역 변수 LAST_BG_PID 에 저장 (서브쉘 방지)
train_bg() {
    local label="$1"; shift
    local log_file="$LOG_DIR/${label// /_}.log"
    echo "[시작] $label  →  로그: $log_file"
    : >"$log_file"
    PYTHONUNBUFFERED=1 "$VENV_PY" "$@" >>"$log_file" 2>&1 &
    LAST_BG_PID=$!
}
wait_pids() {
    local label1="$1" pid1="$2" label2="$3" pid2="$4"
    local ok=true
    if wait "$pid1"; then
        echo "[완료] $label1" >&2
    else
        echo "[FAIL] $label1  (로그: $LOG_DIR/${label1// /_}.log)" >&2
        ok=false
    fi
    if wait "$pid2"; then
        echo "[완료] $label2" >&2
    else
        echo "[FAIL] $label2  (로그: $LOG_DIR/${label2// /_}.log)" >&2
        ok=false
    fi
    $ok || exit 1
}

run_eval() {
    local label="$1" weights="$2" variant="$3" train_data="$4" model_name="$5" test_data="$6" device="$7"
    local extra_args=()
    $DO_FROC && extra_args+=(--save-froc)

    log "[$label] → test: $test_data"
    "$VENV_PY" "$EVAL_PY" \
        --weights     "$weights" \
        --variant     "$variant" \
        --train-data  "$train_data" \
        --model-name  "$model_name" \
        --test-data   "$test_data" \
        --conf        "$CONF" \
        --iou         "$IOU" \
        --device      "$device" \
        "${extra_args[@]}"
}

run_overlay() {
    local label="$1" weights="$2" variant="$3" train_data="$4" model_name="$5" test_data="$6" device="$7"

    log "[$label] overlay → test: $test_data (GPU $device)"
    "$VENV_PY" "$OVERLAY_PY" \
        --weights     "$weights" \
        --variant     "$variant" \
        --train-data  "$train_data" \
        --model-name  "$model_name" \
        --test-data   "$test_data" \
        --conf        "$CONF" \
        --iou         "$IOU" \
        --device      "$device"
}

# ════════════════════════════════════════════════════════════
# 1단계: 학습 (2개씩 병렬, 3라운드)
# ════════════════════════════════════════════════════════════
if ! $SKIP_TRAIN; then

    # ── Round 1: DDR 4cls (GPU $GPU_DDR)  +  FGART 4cls (GPU $GPU_FGART) ──
    log "학습 Round 1: DDR 4cls (GPU $GPU_DDR)  +  FGART 4cls (GPU $GPU_FGART)"

    train_bg "DDR_4cls"   "$DDR4_PY"  --device $GPU_DDR;         PID_D4=$LAST_BG_PID
    train_bg "FGART_4cls" "$FGART_PY" --variant 4cls --device "$GPU_FGART"; PID_F4=$LAST_BG_PID

    wait_pids "DDR 4cls 학습" $PID_D4 "FGART 4cls 학습" $PID_F4

    # ── Round 2: DDR 1cls (GPU $GPU_DDR)  +  FGART 1cls (GPU $GPU_FGART) ──
    log "학습 Round 2: DDR 1cls (GPU $GPU_DDR)  +  FGART 1cls (GPU $GPU_FGART)"

    train_bg "DDR_1cls"   "$DDR1_PY"  --device $GPU_DDR;         PID_D1=$LAST_BG_PID
    train_bg "FGART_1cls" "$FGART_PY" --variant 1cls --device "$GPU_FGART"; PID_F1=$LAST_BG_PID

    wait_pids "DDR 1cls 학습" $PID_D1 "FGART 1cls 학습" $PID_F1

    # ── Round 3: Merge 4cls (GPU $GPU_DDR)  +  Merge 1cls (GPU $GPU_FGART) ──
    log "학습 Round 3: Merge 4cls (GPU $GPU_DDR)  +  Merge 1cls (GPU $GPU_FGART)"

    train_bg "Merge_4cls" "$MERGE4_PY" --device $GPU_DDR;        PID_M4=$LAST_BG_PID
    train_bg "Merge_1cls" "$MERGE1_PY" --device "$GPU_FGART";    PID_M1=$LAST_BG_PID

    wait_pids "Merge 4cls 학습" $PID_M4 "Merge 1cls 학습" $PID_M1

    # ── Round 4: Eophtha 4cls (GPU $GPU_DDR)  +  Eophtha 1cls (GPU $GPU_FGART) ──
    log "학습 Round 4: Eophtha 4cls (GPU $GPU_DDR)  +  Eophtha 1cls (GPU $GPU_FGART)"

    train_bg "Eophtha_4cls" "$EOPTHA4_PY" --device $GPU_DDR;     PID_E4=$LAST_BG_PID
    train_bg "Eophtha_1cls" "$EOPTHA1_PY" --device "$GPU_FGART"; PID_E1=$LAST_BG_PID

    wait_pids "Eophtha 4cls 학습" $PID_E4 "Eophtha 1cls 학습" $PID_E1

    log "학습 전체 완료"
fi

# ════════════════════════════════════════════════════════════
# 2단계: Cross-Evaluation + Overlay (아래 각주 해제 후 --eval 시 실행)
# ════════════════════════════════════════════════════════════
# LABELS=(    "FGART_4cls" "FGART_1cls" "DDR_4cls" "DDR_1cls" "Merge_4cls" "Merge_1cls" )
# WEIGHTS=(
#     "$BASE/runs/fgart/yolo12/weights/best.pt"
#     "$BASE/runs/fgart/yolo12_1cls/weights/best.pt"
#     "$BASE/runs/ddr/yolo12/weights/best.pt"
#     "$BASE/runs/ddr/yolo12_1cls/weights/best.pt"
#     "$BASE/runs/merge/yolo12/weights/best.pt"
#     "$BASE/runs/merge/yolo12_1cls/weights/best.pt"
# )
# VARIANTS=(   "4cls"  "1cls"  "4cls"  "1cls"  "4cls"  "1cls" )
# TRAIN_DATAS=("fgart" "fgart" "ddr"   "ddr"   "merge" "merge")
# MODEL_NAMES=("yolo12" "yolo12_1cls" "yolo12" "yolo12_1cls" "yolo12" "yolo12_1cls")
# EVAL_GPUS=(  $GPU_EVAL_FGART $GPU_EVAL_FGART $GPU_EVAL_DDR $GPU_EVAL_DDR $GPU_SINGLE $GPU_SINGLE )
# TEST_SETS=(
#     "fgart ddr"   # FGART 4cls
#     "fgart ddr"   # FGART 1cls
#     "fgart ddr"   # DDR 4cls
#     "fgart ddr"   # DDR 1cls
#     "fgart ddr"   # Merge 4cls
#     "fgart ddr"   # Merge 1cls
# )
#
# if $DO_EVAL; then
#     log "Cross-Evaluation 시작 (conf=$CONF, iou=$IOU)"
#     for i in 0 1 2 3 4 5; do
#         label="${LABELS[$i]}"
#         weights="${WEIGHTS[$i]}"
#         variant="${VARIANTS[$i]}"
#         train_data="${TRAIN_DATAS[$i]}"
#         model_name="${MODEL_NAMES[$i]}"
#         gpu="${EVAL_GPUS[$i]}"
#         if [[ ! -f "$weights" ]]; then
#             echo "[SKIP] weights 없음: $weights"
#             continue
#         fi
#         for test_data in ${TEST_SETS[$i]}; do
#             run_eval "$label" "$weights" "$variant" "$train_data" "$model_name" "$test_data" "$gpu"
#             $DO_OVERLAY && run_overlay "$label" "$weights" "$variant" "$train_data" "$model_name" "$test_data" "$gpu"
#         done
#     done
# fi

log "전체 완료 ✓"
echo ""
echo "  학습 로그:"
echo "    $LOG_DIR/FGART_4cls.log"
echo "    $LOG_DIR/FGART_1cls.log"
echo "    $LOG_DIR/DDR_4cls.log"
echo "    $LOG_DIR/DDR_1cls.log"
echo "    $LOG_DIR/Merge_4cls.log"
echo "    $LOG_DIR/Merge_1cls.log"
echo "    $LOG_DIR/Eophtha_4cls.log"
echo "    $LOG_DIR/Eophtha_1cls.log"
