"""Known-answer CPU checks for survival numerics and ablation random streams."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

from cancer_metrics import ipcw_brier
from cancer_model import survival_at
from cancer_survival_diagnostics import METRIC_CONVENTIONS, horizon_risk_audit
from model import stability_weighted_common


def inverse_rate(rates):
    """Inverse of the project's positive-rate link for small known test rates."""
    return torch.log(torch.expm1(rates-1e-8))


def saturation_case():
    rates = torch.tensor([[3.], [2.]], dtype=torch.float32)
    logits, starts = inverse_rate(rates), torch.tensor([0.])
    time, event = torch.tensor([1., 2.]), torch.ones(2)
    return {str(h): horizon_risk_audit(logits, time, event, starts, h) for h in (1., 12.)}


def crossing_case():
    rates = torch.tensor([[.08, .015], [.015, .08]], dtype=torch.float64)
    starts = torch.tensor([0., 12.], dtype=torch.float64)
    horizons = torch.tensor([12., 24., 36.], dtype=torch.float64)
    survival = survival_at(inverse_rate(rates), horizons, starts)
    return {'rates': rates.tolist(), 'starts': starts.tolist(), 'horizons': horizons.tolist(),
            'survival': survival.tolist(),
            'interpretation': 'Known two-interval curves cross at 24; single-interval positive rates cannot cross.'}


def rng_case():
    # All views held fixed, so only weight shuffling may consume randomness.
    gen = torch.Generator().manual_seed(3)
    views = [torch.randn(20, 4, generator=gen) for _ in range(4)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(33)
        stability_weighted_common(*views)
        reference = torch.randn(4)
        torch.manual_seed(33)
        stability_weighted_common(*views, shuffle_weights=True)
        legacy_next = torch.randn(4)
        torch.manual_seed(33)
        stability_weighted_common(*views, shuffle_weights=True,
                                  shuffle_generator=torch.Generator().manual_seed(492))
        isolated_next = torch.randn(4)
    return {'legacy_next_noise_matches_reference': bool(torch.equal(reference, legacy_next)),
            'legacy_max_next_noise_difference': float((reference-legacy_next).abs().max()),
            'isolated_next_noise_matches_reference': bool(torch.equal(reference, isolated_next)),
            'isolated_max_next_noise_difference': float((reference-isolated_next).abs().max())}


def run():
    legacy = ipcw_brier([1, 1, 2], [1, 0, 1], [1, 2], [1, 1], [.2, .7], 1.5)
    nan_rejected = False
    try:
        ipcw_brier([1, 3], [1, 1], [1, 3], [1, 1], [.2, .8], float('nan'))
    except ValueError:
        nan_rejected = True
    files = ('research_survival_audit.py', 'cancer_survival_diagnostics.py', 'cancer_metrics.py',
             'cancer_model.py', 'model.py', 'cancer_pathway_stream.py', 'train_pathway.py')
    return {'evidence': 'known-answer CPU synthetic audit; no patient data or server access',
            'probability_saturation': saturation_case(), 'crossing_hazards': crossing_case(),
            'random_stream': rng_case(), 'nan_horizon_rejected': nan_rejected,
            'ipcw_tie_example': {'legacy_computed': legacy['value'], 'legacy_expected': .0875,
                                 'reverse_km_left_event_expected': .11,
                                 'sksurv_028_right_event_expected': .13,
                                 'external_library_executed': False,
                                 'external_expected_source': 'https://github.com/sebp/scikit-survival/blob/v0.28.0/sksurv/metrics.py'},
            'metric_conventions': METRIC_CONVENTIONS,
            'source_sha256': {f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in files},
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
            'limits': ['Examples establish numerical and protocol issues, not their prevalence in existing patient runs.',
                       'The two-interval example is prescribed, not a fitted model or a patient performance improvement.',
                       'Legacy metric tie definitions are preserved; external values are hand calculations from the cited implementation.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    data = run()
    with Path(args.out).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
