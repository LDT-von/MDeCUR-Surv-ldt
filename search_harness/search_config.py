"""Shared configuration for the pre-registered pathway hyperparameter search.

Protocol: HYPERPARAM_SEARCH_PROTOCOL_20261112.md

Every path is overridable through environment variables so the harness can be
run on a different machine (the server) without editing the source.  The
defaults match the local layout under /data1.

Environment variables:
    SEARCH_RUN_DIR      run root that holds queue.json, logs/ and one
                        directory per configuration
    SEARCH_FOLD_DIR     directory holding the internal development fold CSVs
    SEARCH_FOLD         single fold used by Stage 1 (default fold_0.csv)
    SEARCH_GENES        genes.txt for the pathway encoding
    SEARCH_PYTHON       python interpreter used for the training subprocess
    SEARCH_MEASURE      runner script that wraps train_pathway.py and prints
                        MEASURE statistics; defaults to the in-repo copy
    SEARCH_REPO         repository root that contains train_pathway.py
    SEARCH_LD_LIBRARY   LD_LIBRARY_PATH prefix for the training subprocess
"""
import os
from pathlib import Path

REPO = Path(os.environ.get('SEARCH_REPO', '/data1/MDeCUR-Surv-ldt'))
RUN = Path(os.environ.get('SEARCH_RUN_DIR',
                          '/data1/decur_runs/PATHWAY_HYPERPARAM_SEARCH_20261112'))
FOLD_DIR = Path(os.environ.get('SEARCH_FOLD_DIR',
                               '/data1/decur_runs/DEVELOPMENT_INNER_FOLDS_SEED17_20261110/folds'))
FOLD = Path(os.environ.get('SEARCH_FOLD', str(FOLD_DIR / 'fold_0.csv')))
GENES = os.environ.get('SEARCH_GENES',
                        '/data1/decur_runs/BLCA_SINGLE_VIEW_NEW_INPUTS/genes.txt')
PY = os.environ.get('SEARCH_PYTHON', '/home/ubuntu/.conda/envs/trisurv/bin/python')
MEASURE = os.environ.get('SEARCH_MEASURE', str(REPO / 'run_pathway_with_measure.py'))
LD_LIBRARY = os.environ.get('SEARCH_LD_LIBRARY',
                             '/home/ubuntu/.conda/envs/trisurv/lib')

# Protocol revision (HYPERPARAM_SEARCH_PROTOCOL_20261112.md section 3.0): the
# screening uses patch_chunk_size 8192 for every run, including the baseline.
# The historical chunk 2048 five-fold result is reported as a reference arm but
# is not the paired comparison target.
PATCH_CHUNK_SIZE = '8192'
