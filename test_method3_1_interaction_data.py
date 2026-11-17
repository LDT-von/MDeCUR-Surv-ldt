"""Validate synthetic generator semantics; no real data are touched."""
import numpy as np
import pytest
import torch

from method3_1_interaction_data import SCENARIOS, make_interaction_splits


def test_disjoint_reproducible_rotated_splits_and_censoring():
    left, truth, meta = make_interaction_splits(307, 'xor', n_train=40, n_val=20, n_test=25)
    right, truth2, _ = make_interaction_splits(307, 'xor', n_train=40, n_val=20, n_test=25)
    idsets = [set(left[x]['ids']) for x in ('train', 'validation', 'test')]
    assert not idsets[0]&idsets[1] and not idsets[0]&idsets[2] and not idsets[1]&idsets[2]
    for k in ('basis_path_audit_only', 'basis_rna_audit_only'):
        q = truth[k]
        assert np.allclose(q.T @ q, np.eye(q.shape[0]), atol=1e-12)
        assert np.array_equal(q, truth2[k])
    assert meta['audit_truth_never_supplied_to_model']
    for split, data in left.items():
        assert set(data) == {'bags', 'rna', 'ids', 'labels', 'censorship'}
        assert torch.equal(data['rna'], right[split]['rna'])
        assert all(torch.equal(a, b) for a, b in zip(data['bags'], right[split]['bags']))
        assert all(16 <= len(b) <= 32 and b.shape[1] == 4 for b in data['bags'])
        audit = truth[split]
        observed = audit['event_bin_audit_only'] <= audit['censor_bin_audit_only']
        assert np.array_equal(data['censorship'].numpy(), ~observed)
        assert np.array_equal(data['labels'].numpy(), np.where(observed, audit['event_bin_audit_only'],
                                                                audit['censor_bin_audit_only']))
        assert np.array_equal(audit['generating_risk_audit_only'],
                              1.4*np.sign(audit['latent_path_audit_only'][:, 0]*audit['latent_rna_audit_only'][:, 0]))


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_outcome_generation_covers_signal_and_nonlinearity(scenario):
    parts, audit, _ = make_interaction_splits(601, scenario, n_train=400, n_val=8, n_test=8)
    p = audit['train']['latent_path_audit_only'][:, 0]
    g = audit['train']['latent_rna_audit_only'][:, 0]
    risk = audit['train']['generating_risk_audit_only']
    if scenario == 'null':
        assert np.all(risk == 0.)
    if scenario == 'redundant_nonlinear':
        assert np.corrcoef(p, g)[0, 1] > .96
        assert np.allclose(risk, 1.4*(p*p-1.))
    if scenario == 'rna_only':
        assert np.allclose(risk, 1.2*g)
    if scenario == 'path_only':
        assert np.allclose(risk, 1.2*p)
    assert set(parts['train']['censorship'].numpy().tolist()) == {0., 1.}
    assert torch.isfinite(parts['train']['rna']).all()


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        make_interaction_splits(7, 'not-a-scenario')
    with pytest.raises(ValueError):
        make_interaction_splits(7, 'xor', min_patches=32, max_patches=16)
