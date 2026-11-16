#!/bin/bash
# Three decisive controls for the BLCA fold0 parity run.
#
# 1. null       : the training labels are permuted.  The validation score must
#                 collapse to chance.  If it does not, the pipeline reads
#                 something other than the training outcome.
# 2. rna_only   : the pathology is removed from both training and evaluation.
# 3. path_only  : the RNA is removed from both training and evaluation.
#
# Everything else matches the reference run exactly.
set -u
BASE=/data1/decur_runs/PARITY_DCT_FOLD0
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
cd /data1/MDeCUR-Surv-ldt

run () {
  local tag=$1; shift
  rm -rf "$BASE/ctrl_$tag"
  $PY train_pathway_parity.py \
    --manifest $BASE/manifest.csv \
    --survival-reference $BASE/survival_reference.csv \
    --genes $BASE/genes.txt \
    --pathways $BASE/combine_signatures.gmt \
    --out $BASE/ctrl_$tag \
    --num-patches 4096 --n-bins 4 --alpha-surv 0.5 --epochs 30 \
    --patient-batch-size 8 --patch-chunk-size 4096 --disable-auxiliary \
    --legacy-architecture \
    --rna-scale-policy none --lr 0.0005 --weight-decay 0.0005 --seed 3 \
    --endpoint DSS --time-unit months \
    --split-name "5fold/blca/fold_0 (legacy_val)" --device cuda "$@" \
    > $BASE/ctrl_$tag.log 2>&1
}

run null       --shuffle-train-labels &
run rna_only   --freeze-modality pathology &
wait
run path_only  --freeze-modality rna &
wait
echo "ALL CONTROLS DONE"
