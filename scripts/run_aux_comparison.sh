#!/usr/bin/env bash
# New legacy-architecture comparison under the corrected per-patient NLL.
# Historical scores used the old broadcast-affected NLL and are not overwritten.
# Both arms share batch size AND tail minimum; archive actual schedules/gradients
# before attributing a score difference to the auxiliary objectives.
# Run manually only after checking that the selected devices are available.
set -euo pipefail
set -o noclobber
export PYTHONHASHSEED=0

INPUTS="${PARITY_INPUTS_ROOT:-/data1/decur_runs/PARITY_DCT_FOLD0}"
PROJECT="${PARITY_PROJECT_ROOT:-/data1/MDeCUR-Surv-ldt}"
PY="${PARITY_PYTHON:-/home/ubuntu/.conda/envs/trisurv/bin/python}"
RUN_PARENT="${PARITY_RUN_ROOT:-/data1/decur_runs}"
BATCH_OUT="${PARITY_BATCH_OUT:-$RUN_PARENT/aux_comparison_corrected_nll_$(date -u +%Y%m%dT%H%M%SZ)}"
DEVICE_0="${PARITY_DEVICE_0:-cuda:0}"
DEVICE_1="${PARITY_DEVICE_1:-cuda:1}"

for input in manifest.csv survival_reference.csv genes.txt combine_signatures.gmt; do
  if [[ ! -f "$INPUTS/$input" ]]; then
    printf 'Missing required input: %s\n' "$input" >&2
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
printf 'protocol=corrected_per_patient_nll_legacy_aux_comparison\nPYTHONHASHSEED=0\nbatch_tail_minimum=17\n' > "$BATCH_OUT/protocol.txt"
if revision=$(git rev-parse HEAD 2>/dev/null); then
  printf 'git_commit=%s\n' "$revision" >> "$BATCH_OUT/protocol.txt"
else
  printf 'git_commit=unavailable\n' >> "$BATCH_OUT/protocol.txt"
fi

run() {
  local tag=$1 device=$2
  shift 2
  local output="$BATCH_OUT/$tag" log="$BATCH_OUT/$tag.log" command_file="$BATCH_OUT/$tag.command.txt"
  if [[ -e "$output" || -L "$output" || -e "$log" || -L "$log" || -e "$command_file" || -L "$command_file" ]]; then
    printf 'Refusing an existing run output/log/command: %s\n' "$tag" >&2
    return 2
  fi
  local -a command=(
    "$PY" train_pathway_parity.py
    --manifest "$INPUTS/manifest.csv"
    --survival-reference "$INPUTS/survival_reference.csv"
    --genes "$INPUTS/genes.txt"
    --pathways "$INPUTS/combine_signatures.gmt"
    --out "$output"
    --num-patches 4096 --n-bins 4 --alpha-surv 0.5 --epochs 30
    --patient-batch-size 17 --batch-tail-minimum 17 --patch-chunk-size 4096
    --legacy-architecture
    --rna-scale-policy none --lr 0.0005 --weight-decay 0.0005 --seed 3
    --endpoint DSS --time-unit months
    --split-name "5fold/blca/fold_0 (legacy_val)"
    --device "$device" "$@"
  )
  printf '%q ' "${command[@]}" > "$command_file"
  printf '\n' >> "$command_file"
  "${command[@]}" > "$log" 2>&1
}

PIDS=()
TAGS=()
wait_for_jobs() {
  local failed=0 index
  for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
      printf 'Completed: %s\n' "${TAGS[$index]}"
    else
      printf 'Failed: %s (see its new log)\n' "${TAGS[$index]}" >&2
      failed=1
    fi
  done
  PIDS=()
  TAGS=()
  return "$failed"
}

run aux_on "$DEVICE_0" &
PIDS+=("$!")
TAGS+=(aux_on)
run aux_off "$DEVICE_1" --disable-auxiliary &
PIDS+=("$!")
TAGS+=(aux_off)
if ! wait_for_jobs; then
  exit 1
fi
printf 'Auxiliary comparison completed in new batch: %s\n' "$BATCH_OUT"
