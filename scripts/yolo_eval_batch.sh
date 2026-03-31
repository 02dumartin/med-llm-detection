#!/usr/bin/env bash
# Batch cross-dataset evaluation runner.
# Reuses src/yolo/eval.py and writes evaluation CSVs under results/<...>/evaluation/.
#
# Default jobs:
#   fgart_4cls, ddr_4cls, merge_4cls
#   fgart_1cls, ddr_1cls, merge_1cls, eophtha_1cls
#
# Default test datasets:
#   fgart, ddr, eophtha, idrid, diaretdb1
#
# Resume behavior:
#   - If a job already produced the expected evaluation outputs, it is skipped.
#   - Re-running the same command continues from the first incomplete job.
#   - Use --force to re-run completed jobs.
set -euo pipefail

BASE="/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection"
VENV_PY="$BASE/.venv/bin/python"
EVAL_PY="$BASE/src/yolo/eval.py"
LOG_DIR="$BASE/logs/eval_batch"
STATUS_TSV="$LOG_DIR/status.tsv"

CONF="0.05"
IOU="0.5"
DEVICE="7"
SAVE_FROC=true
FROC_ONLY=false
PRED_ONLY=false
SAVE_PRED_TXT=false
NO_PRED_TXT=false
FORCE=false
DRY_RUN=false
ONLY_MODELS=""
ONLY_TESTS=""

pick_first_existing() {
    local path
    for path in "$@"; do
        if [[ -f "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    printf '%s\n' "$1"
}

usage() {
    cat <<EOF
Usage:
  bash scripts/yolo_eval_batch.sh [options]

Options:
  --device <gpu>           GPU id for evaluation (default: ${DEVICE})
  --conf <float>           confidence threshold passed to yolo_eval.py (default: ${CONF})
  --iou <float>            IoU threshold passed to yolo_eval.py (default: ${IOU})
  --save-froc              force enabling FROC
  --no-froc                disable FROC
  --froc-only              reuse existing eval results and compute only FROC + updated metrics_total.csv
  --pred-only              save prediction txt(+conf) only, without metrics/FROC
  --save-pred-txt          save prediction txt(+conf) to each result folder's labels/ directory
  --no-pred-txt            disable prediction txt save in normal evaluation mode
  --force                  rerun jobs even if result files already exist
  --only-models <csv>      comma-separated subset, e.g. fgart_4cls,merge_1cls
  --only-tests <csv>       comma-separated subset, e.g. fgart,ddr,eophtha
  --dry-run                print planned jobs only
  -h, --help               show this help

Resume:
  Run the same command again. Completed jobs are skipped automatically.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --conf) CONF="$2"; shift 2 ;;
        --iou) IOU="$2"; shift 2 ;;
        --save-froc) SAVE_FROC=true; shift ;;
        --no-froc) SAVE_FROC=false; shift ;;
        --froc-only) FROC_ONLY=true; shift ;;
        --pred-only) PRED_ONLY=true; shift ;;
        --save-pred-txt) SAVE_PRED_TXT=true; shift ;;
        --no-pred-txt) NO_PRED_TXT=true; shift ;;
        --force) FORCE=true; shift ;;
        --only-models) ONLY_MODELS="$2"; shift 2 ;;
        --only-tests) ONLY_TESTS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
if [[ ! -f "$STATUS_TSV" ]]; then
    printf "timestamp\tstatus\tmodel_label\ttest_data\tweights\tresults_dir\tlog_file\n" > "$STATUS_TSV"
fi

WEIGHT_FGART_4CLS="${WEIGHT_FGART_4CLS:-$BASE/runs/fgart/yolo12/weights/best.pt}"
WEIGHT_DDR_4CLS="${WEIGHT_DDR_4CLS:-$BASE/runs/ddr/yolo12/weights/best.pt}"
WEIGHT_MERGE_4CLS="${WEIGHT_MERGE_4CLS:-$(pick_first_existing "$BASE/runs/merge_crop/yolo12/weights/best.pt" "$BASE/runs/merge_crop/yolo123/weights/best.pt")}"
WEIGHT_FGART_1CLS="${WEIGHT_FGART_1CLS:-$BASE/runs/fgart/yolo12_1cls/weights/best.pt}"
WEIGHT_DDR_1CLS="${WEIGHT_DDR_1CLS:-$BASE/runs/ddr/yolo12_1cls/weights/best.pt}"
WEIGHT_MERGE_1CLS="${WEIGHT_MERGE_1CLS:-$BASE/runs/merge_crop/yolo12_1cls/weights/best.pt}"
WEIGHT_EOPHTHA_1CLS="${WEIGHT_EOPHTHA_1CLS:-$BASE/runs/eophtha/yolo12_1cls/weights/best.pt}"

MODEL_LABELS=(
    "fgart_4cls"
    "ddr_4cls"
    "merge_4cls"
    "fgart_1cls"
    "ddr_1cls"
    "merge_1cls"
    "eophtha_1cls"
)

TEST_DATASETS=("fgart" "ddr" "eophtha" "idrid" "diaretdb1")

declare -A MODEL_VARIANT=(
    [fgart_4cls]="4cls"
    [ddr_4cls]="4cls"
    [merge_4cls]="4cls"
    [fgart_1cls]="1cls"
    [ddr_1cls]="1cls"
    [merge_1cls]="1cls"
    [eophtha_1cls]="1cls"
)

declare -A MODEL_WEIGHTS=(
    [fgart_4cls]="$WEIGHT_FGART_4CLS"
    [ddr_4cls]="$WEIGHT_DDR_4CLS"
    [merge_4cls]="$WEIGHT_MERGE_4CLS"
    [fgart_1cls]="$WEIGHT_FGART_1CLS"
    [ddr_1cls]="$WEIGHT_DDR_1CLS"
    [merge_1cls]="$WEIGHT_MERGE_1CLS"
    [eophtha_1cls]="$WEIGHT_EOPHTHA_1CLS"
)

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

infer_train_data() {
    local weights="$1"
    local parts=()
    local i
    IFS='/' read -r -a parts <<< "$weights"
    for ((i = 0; i < ${#parts[@]}; i++)); do
        if [[ "${parts[$i]}" == "runs" && $((i + 2)) -lt ${#parts[@]} ]]; then
            printf '%s\n' "${parts[$((i + 1))]}"
            return 0
        fi
    done
    return 1
}

infer_model_name() {
    local weights="$1"
    local parts=()
    local i
    IFS='/' read -r -a parts <<< "$weights"
    for ((i = 0; i < ${#parts[@]}; i++)); do
        if [[ "${parts[$i]}" == "runs" && $((i + 2)) -lt ${#parts[@]} ]]; then
            printf '%s\n' "${parts[$((i + 2))]}"
            return 0
        fi
    done
    return 1
}

results_dir_for_job() {
    local weights="$1" test_data="$2"
    local train_data model_name
    train_data="$(infer_train_data "$weights")"
    model_name="$(infer_model_name "$weights")"
    printf '%s\n' "$BASE/results/$train_data/$model_name/${test_data}_${CONF}_${IOU}"
}

is_job_complete() {
    local out_dir="$1"
    local eval_dir="$out_dir/evaluation"
    if $PRED_ONLY; then
        [[ -f "$eval_dir/labels.done" ]] || [[ -d "$eval_dir/labels" ]]
        return
    fi
    if $FROC_ONLY; then
        [[ -f "$eval_dir/froc.csv" ]]
        return
    fi
    [[ -f "$eval_dir/.done" ]] || {
        [[ -f "$eval_dir/class-accuracy.csv" ]] &&
        [[ -f "$eval_dir/class-prediction.csv" ]] &&
        [[ -f "$eval_dir/summary.csv" ]]
    }
}

append_status() {
    local status="$1" model_label="$2" test_data="$3" weights="$4" results_dir="$5" log_file="$6"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date -u +%FT%TZ)" "$status" "$model_label" "$test_data" "$weights" "$results_dir" "$log_file" \
        >> "$STATUS_TSV"
}

run_job() {
    local model_label="$1" test_data="$2"
    local weights variant results_dir log_file
    local extra_args=()

    if ! contains_csv_item "$ONLY_MODELS" "$model_label"; then
        return 0
    fi
    if ! contains_csv_item "$ONLY_TESTS" "$test_data"; then
        return 0
    fi

    weights="${MODEL_WEIGHTS[$model_label]}"
    variant="${MODEL_VARIANT[$model_label]}"
    results_dir="$(results_dir_for_job "$weights" "$test_data")"
    log_file="$LOG_DIR/${model_label}__${test_data}.log"

    if [[ ! -f "$weights" ]]; then
        echo "[SKIP] missing weights: $model_label -> $weights"
        append_status "SKIP_MISSING_WEIGHTS" "$model_label" "$test_data" "$weights" "$results_dir" "$log_file"
        return 0
    fi

    if ! $FORCE && is_job_complete "$results_dir"; then
        echo "[SKIP] already done: $model_label -> $test_data"
        append_status "SKIP_DONE" "$model_label" "$test_data" "$weights" "$results_dir" "$log_file"
        return 0
    fi

    if $DRY_RUN; then
        echo "[DRY-RUN] $model_label -> $test_data"
        echo "          weights: $weights"
        echo "          results: $results_dir"
        return 0
    fi

    mkdir -p "$(dirname "$results_dir")"
    rm -f "$results_dir/evaluation/.done"

    echo ""
    echo "============================================================"
    echo "[RUN] $model_label -> $test_data"
    echo "  variant : $variant"
    echo "  weights : $weights"
    echo "  device  : $DEVICE"
    echo "  results : $results_dir"
    echo "  log     : $log_file"
    echo "============================================================"

    append_status "START" "$model_label" "$test_data" "$weights" "$results_dir" "$log_file"

    if $PRED_ONLY; then
        extra_args+=(--pred-only)
    elif $FROC_ONLY; then
        extra_args+=(--froc-only)
    elif ! $SAVE_FROC; then
        extra_args+=(--no-froc)
    fi
    if $PRED_ONLY; then
        extra_args+=(--save-pred-txt)
    else
        $SAVE_PRED_TXT && extra_args+=(--save-pred-txt)
        $NO_PRED_TXT && extra_args+=(--no-pred-txt)
    fi

    if "$VENV_PY" "$EVAL_PY" \
        --weights "$weights" \
        --variant "$variant" \
        --test-data "$test_data" \
        --conf "$CONF" \
        --iou "$IOU" \
        --device "$DEVICE" \
        "${extra_args[@]}" \
        > "$log_file" 2>&1; then
        mkdir -p "$results_dir/evaluation"
        if ! $FROC_ONLY && ! $PRED_ONLY; then
            touch "$results_dir/evaluation/.done"
        fi
        echo "[DONE] $model_label -> $test_data"
        append_status "DONE" "$model_label" "$test_data" "$weights" "$results_dir" "$log_file"
    else
        echo "[FAIL] $model_label -> $test_data"
        echo "       log: $log_file"
        append_status "FAIL" "$model_label" "$test_data" "$weights" "$results_dir" "$log_file"
        return 1
    fi
}

echo "[INFO] Batch eval start"
echo "  device      : $DEVICE"
echo "  conf / iou  : $CONF / $IOU"
echo "  save_froc   : $SAVE_FROC"
echo "  froc_only   : $FROC_ONLY"
echo "  pred_only   : $PRED_ONLY"
echo "  save_pred   : $SAVE_PRED_TXT"
echo "  no_pred_txt : $NO_PRED_TXT"
echo "  force       : $FORCE"
echo "  only_models : ${ONLY_MODELS:-<all>}"
echo "  only_tests  : ${ONLY_TESTS:-<all>}"
echo "  logs        : $LOG_DIR"
echo "  status      : $STATUS_TSV"
echo ""
echo "[WARN] E-ophtha 4cls test labels were previously remapped in-place to 2cls."
echo "       4cls-on-E-ophtha metrics are only semantically valid if original 4cls labels are restored."

for model_label in "${MODEL_LABELS[@]}"; do
    for test_data in "${TEST_DATASETS[@]}"; do
        run_job "$model_label" "$test_data"
    done
done

echo ""
echo "[DONE] Batch evaluation finished."
echo "       Re-run the same command to resume from incomplete jobs."
echo "       Status log: $STATUS_TSV"
