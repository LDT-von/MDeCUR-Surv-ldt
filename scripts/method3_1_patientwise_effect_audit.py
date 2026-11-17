"""Post-hoc patient-axis interaction audit of the fixed synthetic XOR study.

This independently replays the preregistered training and selection.  It never
loads patient data: generated split IDs and patientwise outputs are synthetic.
It refuses to run without a complete, matching formal receipt, and refuses to
overwrite any existing audit.  The per-time-bin SD is taken over patients,
not flattened over patients and time bins.
"""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cancer_morphid_model import FrozenMorphologyBank  # noqa: E402
from method3_1_model import Method31  # noqa: E402
from method3_1_interaction_features import GlobalResidualInteractions  # noqa: E402
from method3_1_strict_pipeline import Method31StrictPipeline  # noqa: E402
from scripts.method3_1_stability_study import SEEDS, SIZES, case_data, evaluate, nll  # noqa: E402


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _survival(logits):
    """Per-patient survival at each discrete bin end."""
    return torch.cumprod(1.-torch.sigmoid(logits), dim=1)


def replay(seed, original):
    # Outcome-free bank, patient-level SSL, and OOF probes exactly follow the
    # formal runner. The audit truth returned by case_data is discarded.
    data, _, _ = case_data('xor', seed)
    tr, va, te = (data[k] for k in ('train', 'validation', 'test'))
    bank = FrozenMorphologyBank.fit(tr['bags'], ids=tr['ids'], clusters=2, latent_dim=3,
        max_fit_patches=2560, per_patient_cap=16, seed=seed)
    base = Method31(bank, rna_dim=2, projection_dim=4, bins=4, seed=seed)
    base.pretrain(tr['bags'], tr['rna'], tr['ids'], epochs=35)
    base.fit_probes(tr['bags'], tr['rna'], tr['ids'], folds=4)
    mapper = GlobalResidualInteractions(base).fit(tr['bags'], tr['rna'], tr['ids'], folds=4)
    map_train = mapper.training_features()
    map_val = mapper.validation_features(va['bags'], va['rna'], va['ids'])
    map_test = mapper.test_features(te['bags'], te['rna'], te['ids'])
    enc_train, rna_train, c_train = base.cached
    _, enc_val, rna_val = base._input(va['bags'], va['rna'], va['ids'])
    _, enc_test, rna_test = base._input(te['bags'], te['rna'], te['ids'])
    pipeline = Method31StrictPipeline(mapper, mode='tanh_outer', expand_clr=True)
    pipeline._fit_moments()
    blocks_train = pipeline._blocks(map_train, enc_train, rna_train, c_train)
    blocks_val = pipeline._blocks(map_val, enc_val, rna_val, None)
    blocks_test = pipeline._blocks(map_test, enc_test, rna_test, None)
    row, head = evaluate('new_bounded', tr, va, te, blocks_train, blocks_val, blocks_test)
    with torch.no_grad():
        logits = head.predict(blocks_test, te['ids'])
        interaction = (blocks_test['interaction']-head.mean['interaction'])/head.scale['interaction']
        contribution = interaction @ head.weights['interaction'].T
        logits_without = logits-contribution
        survival, survival_without = _survival(logits), _survival(logits_without)
        risk, risk_without = -survival.sum(1), -survival_without.sum(1)
        risk_delta, survival_delta = risk-risk_without, survival-survival_without
        actual = {'test_nll': nll(logits, te),
                  'zero_interaction_test_nll': nll(logits_without, te),
                  'interaction_weight_norm': float(head.weights['interaction'].norm())}
    # Audit gradients need autograd even if the caller disabled it.
    with torch.enable_grad():
        actual['survival_only_interaction_grad_norm'] = head.survival_interaction_gradient_norm(
            blocks_train, tr['labels'], tr['censorship'], tr['ids'])
    checks = {key: {'recorded': float(original[key]), 'replayed': value,
                    'absolute_difference': abs(float(original[key])-value),
                    'matches_atol_1e-10': abs(float(original[key])-value) <= 1e-10}
              for key, value in actual.items()}
    if not all(item['matches_atol_1e-10'] for item in checks.values()):
        raise RuntimeError(f'Formal receipt mismatch for seed {seed}: {checks}')
    return {'seed': seed, 'formal_receipt_checks': checks,
            'selection': {'epoch': row['selection']['epoch'],
                          'penalties': row['selection']['penalties']},
            'interaction_contribution_per_bin_patient_std': contribution.std(0, unbiased=False).tolist(),
            'interaction_contribution_per_bin_patient_mean': contribution.mean(0).tolist(),
            'patientwise_risk_delta_mean_absolute': float(risk_delta.abs().mean()),
            'patientwise_risk_delta_std': float(risk_delta.std(unbiased=False)),
            'patientwise_survival_delta_mean_absolute_per_bin': survival_delta.abs().mean(0).tolist(),
            'patientwise': {'synthetic_ids': list(te['ids']),
                            'risk_with_interaction': risk.tolist(),
                            'risk_without_interaction': risk_without.tolist(),
                            'risk_difference': risk_delta.tolist(),
                            'survival_with_interaction': survival.tolist(),
                            'survival_without_interaction': survival_without.tolist()}}


def main():
    torch.set_num_threads(1)
    source = ROOT/'results'/'method3_1_stability_20261117.json'
    destination = ROOT/'results'/'method3_1_patientwise_effect_audit_20261117.json'
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing audit')
    if not source.is_file():
        raise FileNotFoundError('The merged formal receipt is required before this post-hoc audit')
    formal_hash = _file_sha(source)
    formal = json.loads(source.read_text(encoding='utf8'))
    if not formal.get('synthetic_only') or formal.get('failures') != 0:
        raise ValueError('The formal receipt must be synthetic-only and have no failed groups')
    if formal.get('seeds') != list(SEEDS) or formal.get('sizes') != SIZES:
        raise ValueError('Formal fixed seeds or split sizes do not match the replay protocol')
    for path, expected in formal['source_sha256'].items():
        if _file_sha(ROOT/path) != expected:
            raise ValueError(f'Study source differs from its formal receipt: {path}')
    original_rows = [r for r in formal['runs'] if r['case'] == 'xor' and 'failure' not in r]
    by_seed = {r['seed']: r for r in original_rows}
    if len(original_rows) != len(SEEDS) or set(by_seed) != set(SEEDS):
        raise ValueError('The formal receipt must have exactly one XOR row per fixed seed')
    audit_sha = _file_sha(Path(__file__).resolve())
    rows = []
    for seed in SEEDS:
        chosen = [a for a in by_seed[seed]['outcomes'] if a['arm'] == 'new_bounded']
        if len(chosen) != 1:
            raise ValueError(f'Expected one new_bounded arm for seed {seed}')
        result = replay(seed, chosen[0])
        rows.append(result)
        print(f"seed {seed}: per-bin patient SD "
              f"{[round(x, 5) for x in result['interaction_contribution_per_bin_patient_std']]}; "
              f"mean abs risk delta {result['patientwise_risk_delta_mean_absolute']:.6f}", flush=True)
    if _file_sha(source) != formal_hash or _file_sha(Path(__file__).resolve()) != audit_sha:
        raise RuntimeError('Formal receipt or audit source changed during replay')
    for path, expected in formal['source_sha256'].items():
        if _file_sha(ROOT/path) != expected:
            raise RuntimeError(f'Study source changed during replay: {path}')
    result = {'synthetic_only': True, 'scenario': 'xor', 'seeds': list(SEEDS),
              'formal_receipt': source.name, 'formal_receipt_sha256': formal_hash,
              'audit_source_sha256': audit_sha,
              'checks_per_seed': ['NLL with and without interaction', 'interaction weight norm',
                                  'survival-only interaction gradient norm'],
              'risk_definition': 'negative sum of bin-end survival; larger means higher risk',
              'contribution_std_axis': 'patient axis independently for each time bin, not flattened',
              'causal_boundary': 'within-model feature removal is a synthetic post-hoc perturbation, not causal or PID proof',
              'runs': rows}
    with destination.open('x', encoding='utf8') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(f'Wrote {destination}; checked {len(rows)} fixed seeds.', flush=True)


if __name__ == '__main__':
    main()
