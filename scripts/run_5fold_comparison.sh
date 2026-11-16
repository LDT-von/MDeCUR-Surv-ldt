#!/usr/bin/env bash
# Five-fold BLCA multimodal comparison: PIBD, CAMR, DeReF and DIMAF.
# DCT numbers are an archived historical reference, never a training arm.
#
# The script is deliberately conservative:
#   * it never touches an existing run directory, log, plan or summary;
#   * it refuses to start when the required inputs are missing;
#   * it writes into a NEW batch directory, so an earlier training queue keeps
#     running untouched;
#   * it does not touch the outer test partition;
#   * the DCT reference is loaded from the committed archive, never re-run.
#
# Run it manually on the server, one fold at a time, after checking that the
# requested GPU is free.  Nothing here starts itself.
set -euo pipefail
set -o noclobber
export PYTHONHASHSEED=0

INPUTS="${COMPARISON_INPUTS_ROOT:-/data1/decur_runs/PARITY_DCT_FOLD0}"
PROJECT="${COMPARISON_PROJECT_ROOT:-/data1/MDeCUR-Surv-ldt}"
PY="${COMPARISON_PYTHON:-/home/ubuntu/.conda/envs/trisurv/bin/python}"
RUN_PARENT="${COMPARISON_RUN_ROOT:-/data1/decur_runs}"
BATCH_OUT="${COMPARISON_BATCH_OUT:-$RUN_PARENT/fivefold_comparison_$(date -u +%Y%m%dT%H%M%SZ)}"
DEVICE="${COMPARISON_DEVICE:-cuda:0}"
EPOCHS="${COMPARISON_EPOCHS:-30}"
FOLDS="${COMPARISON_FOLDS:-0 1 2 3 4}"
MODELS="${COMPARISON_MODELS:-pibd_2m camr_2m deref_2m dimaf_2m}"
DEREF_SIGNATURES="${COMPARISON_DEREF_SIGNATURES:-/data1/dataset_csv/signatures/deref_signatures.csv}"

for input in manifest.csv survival_reference.csv genes.txt combine_signatures.gmt; do
  if [[ ! -f "$INPUTS/$input" ]]; then
    printf 'Missing required input: %s\n' "$INPUTS/$input" >&2
    exit 2
  fi
done

cd -- "$PROJECT"
if [[ -e "$BATCH_OUT" || -L "$BATCH_OUT" ]]; then
  printf 'Refusing an existing batch output: %s\n' "$BATCH_OUT" >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$BATCH_OUT")"
mkdir -- "$BATCH_OUT"

{
  printf 'protocol=blca_5fold_dss_uni_v1_p4096_dnll_v1\n'
  printf 'PYTHONHASHSEED=0\n'
  printf 'epochs=%s\n' "$EPOCHS"
  printf 'device=%s\n' "$DEVICE"
  printf 'models=%s\n' "$MODELS"
  printf 'folds=%s\n' "$FOLDS"
  printf 'outer_test=not_loaded\n'
} > "$BATCH_OUT/protocol.txt"
if revision=$(git rev-parse HEAD 2>/dev/null); then
  printf 'git_commit=%s\n' "$revision" >> "$BATCH_OUT/protocol.txt"
else
  printf 'git_commit=unavailable\n' >> "$BATCH_OUT/protocol.txt"
fi

run_one() {
  local fold=$1 model=$2
  local tag="fold${fold}_${model}"
  local plan="$BATCH_OUT/plan_${tag}.json"
  local output="$BATCH_OUT/run_${tag}"
  local log="$BATCH_OUT/run_${tag}.log"
  local command_file="$BATCH_OUT/run_${tag}.command.txt"
  if [[ -e "$plan" || -L "$plan" || -e "$output" || -L "$output" \
     || -e "$log" || -L "$log" || -e "$command_file" || -L "$command_file" ]]; then
    printf 'Refusing an existing plan/output/log/command: %s\n' "$tag" >&2
    return 2
  fi
  local -a plan_command=(
    "$PY" comparison_5fold.py plan
    --fold "$fold"
    --manifest "$INPUTS/manifest.csv"
    --survival-reference "$INPUTS/survival_reference.csv"
    --genes "$INPUTS/genes.txt"
    --gmt "$INPUTS/combine_signatures.gmt"
    --deref-signatures "$DEREF_SIGNATURES"
    --device "$DEVICE"
    --out "$plan"
  )
  printf '%q ' "${plan_command[@]}" > "$command_file"
  printf '\n' >> "$command_file"
  "${plan_command[@]}" >> "$log" 2>&1

  local plan_sha
  plan_sha=$("$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$plan")
  local -a run_command=(
    "$PY" comparison_5fold.py run
    --plan "$plan"
    --plan-sha256 "$plan_sha"
    --model "$model"
    --device "$DEVICE"
    --epochs "$EPOCHS"
    --out "$output"
  )
  printf '%q ' "${run_command[@]}" >> "$command_file"
  printf '\n' >> "$command_file"
  "${run_command[@]}" >> "$log" 2>&1
}

for fold in $FOLDS; do
  for model in $MODELS; do
    printf 'Running fold %s model %s\n' "$fold" "$model"
    run_one "$fold" "$model"
  done
done

SUMMARY="$BATCH_OUT/summary.json"
if [[ -e "$SUMMARY" ]]; then
  printf 'Refusing an existing summary: %s\n' "$SUMMARY" >&2
  exit 2
fi
"$PY" comparison_5fold.py summary \
  --runs "$BATCH_OUT"/run_fold*_*/ \
  --dct-reference reference/dct_5fold_archive.json \
  --out "$SUMMARY"

printf 'Five-fold comparison completed in new batch: %s\n' "$BATCH_OUT"
printf 'Summary: %s\n' "$SUMMARY"
