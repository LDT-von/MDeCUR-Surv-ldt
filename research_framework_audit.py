"""CPU mechanism audits with constructed data; no patient files or server access.

Runs the actual matching-score/pooling and correlation functions. The null
examples prove allowed failure modes, not that a fitted patient model uses them.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform

import torch
from torch.nn import functional as F

from cancer_pathway import PathwayLocalSurvival, pathway_correlation_loss
from cancer_pathway_stream import _feature_scores, _pool_summary
from cancer_stability_candidates import scaled_difference_weights
from model import stability_weighted_common


def correlation(a, b, eps=1e-4):
    a, b = a-a.mean(0), b-b.mean(0)
    return (a/(a.square().mean(0)+eps).sqrt()).T @ (b/(b.square().mean(0)+eps).sqrt())/len(a)


def correlation_summary(a, b):
    matrix = correlation(a, b)
    loss, info = pathway_correlation_loss(a[:, None, :], b[:, None, :], 1)
    diag = matrix.diagonal()
    off = matrix-torch.diag(diag)
    return {'diagonal_mean': diag.mean().item(), 'off_diagonal_max_abs': off.abs().max().item(),
            'identity_loss': loss.item(), 'low_variance_coordinates': info['low_variance_coordinates'],
            'path_centered_rank': int(torch.linalg.matrix_rank(a-a.mean(0)).item()),
            'path_coordinate_variances': a.var(0, unbiased=False).tolist()}


def matching_null(seed, dim=4, temperature=.2):
    """Cartesian product makes pathology amplitude, RNA and donor query independent."""
    torch.manual_seed(seed)
    model = PathwayLocalSurvival(dim, dim, [list(range(dim))], hidden=dim, common=dim//2,
                                local_dim=2*dim, local_common=dim,
                                match_temperature=temperature).double().eval()
    with torch.no_grad():
        model.match_path.weight.copy_(torch.eye(dim, dtype=torch.float64))
        model.match_rna.weight.copy_(torch.eye(dim, dtype=torch.float64))
        # A control with nonzero coordinate variance; it still has rank one.
        # Random tiny initial queries could make all variances fall below eps.
        model.pathway_queries.fill_(4.)
    basis = torch.cat((torch.eye(dim, dtype=torch.float64), -torch.eye(dim, dtype=torch.float64)))
    amplitudes = torch.linspace(.9, 1.1, 9, dtype=torch.float64)
    observations = list(itertools.product(range(2*dim), range(2*dim), range(len(amplitudes))))
    rna = torch.stack([basis[r] for r, _, _ in observations])
    donor = torch.stack([basis[q] for _, q, _ in observations])
    amp = torch.stack([amplitudes[a] for _, _, a in observations])
    conditional, independent, anchor, detached = [], [], [], []
    with torch.no_grad():
        for i in range(len(observations)):
            features = amp[i]*basis  # Already-encoded patch features, independent of target RNA.
            q = model.match_queries(rna[i:i+1])
            qd = model.match_queries(donor[i:i+1])
            pooled = _pool_summary(features, _feature_scores(model, features, q))[1]
            conditional.append(pooled[-1])
            anchor.append(pooled[1])
            independent.append(_pool_summary(features, _feature_scores(model, features, qd))[1][-1])
            detached.append(_pool_summary(features, _feature_scores(model, features, q.detach()))[1][-1])
    conditional, independent, anchor, detached = map(torch.stack, (conditional, independent, anchor, detached))
    coef = 2*torch.sinh(torch.tensor(1/temperature, dtype=torch.float64))/(2*torch.cosh(torch.tensor(1/temperature, dtype=torch.float64))+2*(dim-1))
    return {'enumerated_samples': len(rna), 'independent_patient_sample': False,
            'dimension': dim, 'temperature': temperature,
            'construction': 'Cartesian product: pathology amplitude x target RNA basis x independent query basis',
            'raw_pathology_independent_of_rna_by_construction': True,
            'analytic_unregularized_diagonal': (amplitudes.mean()/amplitudes.square().mean().sqrt()).item(),
            'closed_form_pool_max_error': (conditional-coef*amp[:, None]*rna).abs().max().item(),
            'own_query': correlation_summary(conditional, rna),
            'independent_query': correlation_summary(independent, rna),
            'pathology_only_anchor': correlation_summary(anchor, rna),
            'detach_query_max_difference': (conditional-detached).abs().max().item()}


def unique_nonlinear_null():
    rna = torch.linspace(-1, 1, 1001, dtype=torch.float64)
    path = rna.square()-rna.square().mean()
    loss, info = pathway_correlation_loss(path[:, None, None], rna[:, None, None], 0)
    return {'zero_target_loss': loss.item(), 'linear_correlation': correlation(path[:, None], rna[:, None]).item(),
            'path_variance': path.var(unbiased=False).item(), 'low_variance_coordinates': info['low_variance_coordinates'],
            'exact_nonlinear_dependence': 'path = RNA**2 - mean(RNA**2)'}


def stability_shift(seed):
    generator = torch.Generator().manual_seed(seed)
    n, d = 128, 8
    # Independent synthetic training reference: fixed scales without validation fitting.
    calibration = torch.randn(2048, d, generator=generator, dtype=torch.float64)
    scale = calibration.std(0, unbiased=False)
    p, r = [torch.randn(n, d, generator=generator, dtype=torch.float64) for _ in range(2)]
    sigma = torch.cat((torch.full((n//2, 1), .03), torch.full((n//2, 1), .8)))
    p2 = p+sigma*torch.randn(n, d, generator=generator, dtype=torch.float64)
    r2 = r+sigma*torch.randn(n, d, generator=generator, dtype=torch.float64)
    shifted = [v+100. for v in (p, p2, r, r2)]
    _, legacy = stability_weighted_common(p, p2, r, r2)
    _, moved = stability_weighted_common(*shifted)
    candidate, candidate_info = scaled_difference_weights(p, p2, r, r2, scale, scale)
    candidate_shift, _ = scaled_difference_weights(*shifted, scale, scale)
    coordinate_scale = torch.linspace(.5, 2., d, dtype=torch.float64)
    rescaled, _ = scaled_difference_weights(p*coordinate_scale, p2*coordinate_scale,
                                            r*coordinate_scale, r2*coordinate_scale,
                                            scale*coordinate_scale, scale*coordinate_scale)
    def summary(w):
        return {'stable_half_mean': w[:n//2].mean().item(), 'noisy_half_mean': w[n//2:].mean().item(),
                'ess_fraction': (w.sum().square()/w.square().sum()/n).item()}
    return {'legacy_original': summary(legacy['weights']), 'legacy_shifted': summary(moved['weights']),
            'legacy_shift_max_weight_change': (legacy['weights']-moved['weights']).abs().max().item(),
            'candidate': summary(candidate), 'candidate_shift_max_weight_change': (candidate-candidate_shift).abs().max().item(),
            'candidate_joint_rescale_max_weight_change': (candidate-rescaled).abs().max().item(),
            'candidate_detached': not candidate.requires_grad,
            'scale_source': '2048 independent synthetic training-reference rows, frozen',
            'sensitivity_is_not_clinical_reliability': True}


def run(seeds):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return {'evidence': 'constructed CPU mechanism checks only; no patient data', 'stability_seeds': seeds,
                'matching_null': matching_null(seeds[0]),
                'unique_nonlinear_null': unique_nonlinear_null(),
                'stability_shift': [stability_shift(seed) for seed in seeds],
                'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ('research_framework_audit.py', 'cancer_stability_candidates.py',
                                               'cancer_pathway.py', 'cancer_pathway_stream.py', 'model.py')},
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
                'limits': ['Selection null uses controlled post-encoder patch/token inputs and known linear matching maps, not patient checkpoint inference.',
                           'The matching null enumerates a finite product population with repeated bags, not independent patient observations.',
                           'Constructed nulls establish allowed failure modes, not their prevalence or effect in trained models.',
                           'The candidate weight is research-only; fixed representation scales and hyperparameters require separate evaluation.',
                           'Repeated seeds here vary only the synthetic setup; they are not independent patient evidence.']}
    finally:
        torch.set_num_threads(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--out', required=True, help='New JSON path; never overwritten')
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    result = run(args.seeds)
    with Path(args.out).open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'out': args.out, 'seeds': args.seeds, 'evidence': result['evidence']}))


if __name__ == '__main__':
    main()
