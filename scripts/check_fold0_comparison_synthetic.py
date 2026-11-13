"""Exercise both 30-epoch comparison arms on generated data, never patient inputs."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import comparison_fold0 as compare
from comparison_camr import CAMR2M
from test_comparison_fold0 import Fold0ComparisonTest


def check(out):
    case = Fold0ComparisonTest()
    case.setUpClass()
    case.setUp()
    try:
        plan, sha = case.plan()
        results, paths = {}, []
        for name in compare.MODELS:
            directory = case.root/name
            paths.append(directory)
            with redirect_stdout(io.StringIO()):
                result = compare.run_model(plan, sha, name, directory, device='cpu', smoke=False)
            assert result['epochs_completed'] == result['last_epoch'] == 30
            # Reload each checkpoint and independently reproduce its saved validation predictions.
            cohort, reference, edges, original, _ = compare.load_inputs(case.manifest, case.reference, case.ref_run)
            try:
                bags = [compare.BudgetedPatientBag(b, 4096, 'eval', 3) for b in cohort['val']['bags']]
                labels = torch.from_numpy(compare.discretize_labels(reference['val'][0], edges))
                model = (compare.PathwayLocalSurvival(**original['model_kwargs']) if name == 'mdecur_aux_off'
                         else CAMR2M(6, 8, bins=4))
                states = {}
                for selection in ('best', 'last'):
                    checkpoint = torch.load(directory/f'checkpoint_{selection}.pt', weights_only=True)
                    states[selection] = checkpoint['model_state']
                    assert checkpoint['epoch'] == result[f'{selection}_epoch']
                    model.load_state_dict(checkpoint['model_state'])
                    metrics, survival = compare.evaluate(name, model, cohort['val'], bags, cohort['val']['rna'],
                                                         labels, reference, edges, torch.device('cpu'))
                    assert metrics == result[f'{selection}_validation']
                    saved = np.loadtxt(directory/f'val_predictions_{selection}.csv', delimiter=',',
                                       skiprows=1, usecols=(2, 3, 4, 5))
                    np.testing.assert_array_equal(saved, survival.astype(float))
                distinct = any(not torch.equal(v, states['last'][k]) for k, v in states['best'].items())
                results[name] = {'epochs': 30, 'best_epoch': result['best_epoch'], 'last_epoch': 30,
                                 'best_and_last_weights_differ': distinct,
                                 'both_checkpoint_predictions_and_metrics_reproduced': True,
                                 'training_sampling_sha256': result['training_sampling_sha256']}
            finally:
                compare.close(cohort)
        summary = compare.summarize(paths, case.root/'comparison.json')
        assert len(summary['rows']) == 4
        assert results['mdecur_aux_off']['training_sampling_sha256'] == results['camr_2m']['training_sampling_sha256']
        receipt = {'scope': 'generated synthetic 12 train/4 validation only; no patient or CUDA evidence',
                   'results': results, 'common_plan_sha256': sha, 'source_sha256': compare.sources(),
                   'four_row_summary_verified': True}
        compare.write_new(out, receipt)
        return receipt
    finally:
        case.doCleanups()
        case.tearDownClass()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    print(json.dumps(check(parser.parse_args().out), indent=2))
