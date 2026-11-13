"""Cross-check the parity metrics against the reference implementation.

The reference pipeline lives in a sibling repository and is not a dependency
of this one.  These checks therefore run only when that repository is present;
they are skipped otherwise, and the skip is explicit in the report.
"""
import csv
import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np
import torch

from cancer_parity import (dct_bin_edges, discrete_survival, dct_risk, metric_time_grid,
                           parity_metrics)


REFERENCE_ROOT = Path(os.environ.get('DCT_REG_ROOT', '/data1/DCT-Reg'))
REFERENCE_PATH = REFERENCE_ROOT/'survot_rank/research/legacy/slotspe_runtime'
PARITY_INPUTS = Path(os.environ.get('PARITY_INPUTS_ROOT',
                                    '/data1/decur_runs/PARITY_DCT_FOLD0'))


def _load_reference():
    """Import the reference metric helpers without importing the whole package."""
    if not (REFERENCE_PATH/'utils/core_utils.py').is_file():
        return None
    import sys
    added = [str(REFERENCE_PATH), str(REFERENCE_ROOT)]
    for path in added:
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        spec = importlib.util.spec_from_file_location(
            'reference_core_utils', REFERENCE_PATH/'utils/core_utils.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


class ReferenceCrossCheckTest(unittest.TestCase):
    """The parity metrics must reproduce the reference numbers exactly."""

    @classmethod
    def setUpClass(cls):
        cls.reference = _load_reference()
        if cls.reference is None:
            raise unittest.SkipTest('The reference repository is not available')

    def _cohort(self):
        manifest = PARITY_INPUTS/'manifest.csv'
        reference = PARITY_INPUTS/'survival_reference.csv'
        if not manifest.is_file() or not reference.is_file():
            raise unittest.SkipTest('The parity inputs have not been built')
        rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
        values = {row['patient_id']: (float(row['time']), float(row['event']))
                  for row in csv.DictReader(reference.open(encoding='utf-8-sig'))}
        train = [row['patient_id'] for row in rows if row['split'] == 'train']
        val = [row['patient_id'] for row in rows if row['split'] == 'val']
        return (np.array([values[i][0] for i in train]),
                np.array([values[i][1] for i in train]),
                np.array([values[i][0] for i in val]),
                np.array([values[i][1] for i in val]))

    def test_bin_edges_match_the_reference(self):
        import pandas as pd
        train_time, train_event, _, _ = self._cohort()
        split = pd.read_csv('/data1/dataset_csv/splits/5fold/blca/fold_0.csv')
        clinical = pd.read_csv('/data1/dataset_csv/clinical/all/blca.csv')
        clinical = clinical[['case id', 'survival_months_dss', 'censorship_dss', 'wsi']]
        clinical = clinical.dropna().drop_duplicates(subset=['case id'], keep='first')
        train_ids = set(split['train'].dropna().astype(str))
        rows = clinical[clinical['case id'].isin(train_ids)]
        uncensored = rows[rows['censorship_dss'] < 1]
        _, edges = pd.qcut(uncensored['survival_months_dss'], q=4, retbins=True,
                           labels=False, duplicates='drop')
        edges = edges.astype(float)
        edges[0], edges[-1] = -np.inf, np.inf
        mine = dct_bin_edges(train_time, train_event, 4)
        self.assertTrue(np.array_equal(edges, mine))

    def test_metric_grid_matches_the_reference(self):
        train_time, train_event, val_time, _ = self._cohort()
        edges = dct_bin_edges(train_time, train_event, 4)
        predictions = np.random.default_rng(0).uniform(.1, .9, (len(val_time), 4))
        reference_times, _ = self.reference._select_valid_metric_time_grid(
            val_time, {'time': train_time}, edges, predictions)
        grid, _ = metric_time_grid(val_time, train_time, edges, 4)
        self.assertTrue(np.array_equal(reference_times, grid))

    def test_every_metric_matches_the_reference(self):
        from sksurv.util import Surv
        train_time, train_event, val_time, val_event = self._cohort()
        edges = dct_bin_edges(train_time, train_event, 4)

        class Factory:
            bins = edges
            n_bins = 4

        torch.manual_seed(7)
        survival = discrete_survival(torch.randn(len(val_time), 4)).numpy()
        risk = dct_risk(survival)
        # The reference stores a censorship column where 1 means censored.
        train_censorship = 1-train_event
        val_censorship = 1-val_event
        train_surv = Surv.from_arrays(event=(1-train_censorship).astype(bool), time=train_time)
        reference = self.reference._calculate_metrics(
            None, Factory(), train_surv, risk, val_censorship, val_time, survival)
        mine = parity_metrics(train_time, train_event, val_time, val_event, survival, edges)
        self.assertEqual(mine['c_index'], reference[0])
        self.assertEqual(mine['c_index_ipcw'], reference[1])
        self.assertEqual(mine['ibs'], reference[3])
        self.assertEqual(mine['iauc'], reference[4])
        self.assertTrue(np.array_equal(np.asarray(mine['brier']), np.asarray(reference[2])))


if __name__ == '__main__':
    unittest.main()
