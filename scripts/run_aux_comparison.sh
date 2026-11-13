#!/bin/bash
# Does the auxiliary decoupling objective change the score?
#
# The auxiliary path has a rank bound: the identity-correlation terms need at
# least hidden+1 = 17 patients per batch, while the earlier controls used a
# batch of 8.  The batch size is therefore a confound, and both arms of this
# comparison use the same batch of 17.  The only difference between the arms is
# whether the eleven auxiliary correlation terms contribute gradients.
#
#   aux_on   : the default weights of this repository's method
#   aux_off  : every auxiliary weight forced to zero, common mode off
#
# Everything else matches the reference parity run: the same manifest, the same
# survival reference, the same bins, the same patch budget, the same seed and
# the same checkpoint selection rule.
set -u
BASE=/data1/decur_runs/PARITY_DCT_FOLD0
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
cd /data1/MDeCUR-Surv-ldt

run () {
  local tag=$1; local device=$2; shift 2
  rm -rf "$BASE/$tag"
  $PY train_pathway_parity.py \
    --manifest $BASE/manifest.csv \
    --survival-reference $BASE/survival_reference.csv \
    --genes $BASE/genes.txt \
    --pathways $BASE/combine_signatures.gmt \
    --out $BASE/$tag \
    --num-patches 4096 --n-bins 4 --alpha-surv 0.5 --epochs 30 \
    --patient-batch-size 17 --patch-chunk-size 4096 \
    --rna-scale-policy none --lr 0.0005 --weight-decay 0.0005 --seed 3 \
    --endpoint DSS --time-unit months \
    --split-name "5fold/blca/fold_0 (legacy_val)" --device "$device" "$@" \
    > $BASE/$tag.log 2>&1
}

run aux_on  cuda:0 &
run aux_off cuda:1 --disable-auxiliary &
wait
echo "AUX COMPARISON DONE"
