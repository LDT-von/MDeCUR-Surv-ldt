"""Known-answer tests for the independent in-memory synthetic generator."""
import numpy as np
import pytest
import torch

from method3_1_independent_synthetic import SCENARIOS, make_independent_splits


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_scenarios_expose_only_observable_inputs_and_correct_censoring(scenario):
    splits, truth, meta = make_independent_splits(871, scenario, n_train=20, n_val=12, n_test=16)
    assert meta['synthetic_only'] and not meta['truth_in_model_inputs']
    ids = []
    for name, count in [('train', 20), ('validation', 12), ('test', 16)]:
        part, audit = splits[name], truth[name]
        assert set(part) == {'bags', 'rna', 'ids', 'labels', 'censorship'}
        assert len(part['bags']) == len(part['ids']) == count and part['rna'].shape == (count, 2)
        assert len({len(bag) for bag in part['bags']}) > 1
        assert all(bag.ndim == 2 and bag.shape[1] == 4 and torch.isfinite(bag).all() for bag in part['bags'])
        assert torch.isfinite(part['rna']).all()
        event, censor = audit['event_bin_audit_only'], audit['censor_bin_audit_only']
        observed = event <= censor
        assert np.array_equal(part['censorship'].numpy(), ~observed)
        assert np.array_equal(part['labels'].numpy(), np.where(observed, event, censor))
        assert np.all((part['labels'].numpy() >= 0) & (part['labels'].numpy() < 4))
        assert np.isfinite(audit['risk_by_bin_audit_only']).all()
        ids.append(set(part['ids']))
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    for key in ('pathology_basis_audit_only', 'rna_basis_audit_only'):
        basis = truth[key]
        assert np.allclose(basis.T @ basis, np.eye(len(basis)), atol=1e-12)


def test_reproducibility_and_test_only_rna_mismatch():
    matched, _, _ = make_independent_splits(813, 'xor', n_train=12, n_val=8, n_test=10)
    repeated, _, _ = make_independent_splits(813, 'xor', n_train=12, n_val=8, n_test=10)
    mismatched, truth, meta = make_independent_splits(813, 'xor', n_train=12, n_val=8, n_test=10, mismatch_test=True)
    assert meta['mismatch_test_rna_only']
    for name in matched:
        for key in ('rna', 'labels', 'censorship'):
            assert torch.equal(matched[name][key], repeated[name][key])
        assert all(torch.equal(a, b) for a, b in zip(matched[name]['bags'], repeated[name]['bags']))
        assert matched[name]['ids'] == mismatched[name]['ids']
        assert torch.equal(matched[name]['labels'], mismatched[name]['labels'])
        assert torch.equal(matched[name]['censorship'], mismatched[name]['censorship'])
        assert all(torch.equal(a, b) for a, b in zip(matched[name]['bags'], mismatched[name]['bags']))
        if name != 'test':
            assert torch.equal(matched[name]['rna'], mismatched[name]['rna'])
        else:
            assert torch.equal(mismatched[name]['rna'], matched[name]['rna'].roll(1, 0))
            assert np.all(truth[name]['rna_row_permutation_audit_only'] != np.arange(len(matched[name]['ids'])))


def test_xor_support_and_time_varying_sign_are_known():
    _, truth, _ = make_independent_splits(873, 'xor_time_varying', n_train=1200, n_val=2, n_test=3)
    audit = truth['train']
    p, g = audit['pathology_latent_audit_only'][:, 0], audit['rna_latent_audit_only'][:, 0]
    risk = audit['risk_by_bin_audit_only']
    assert abs(np.corrcoef(p, g)[0, 1]) < .10
    assert .40 < np.mean(risk[:, 0] > 0) < .60
    assert np.allclose(risk[:, 0], 1.4*np.sign(p*g))
    assert np.array_equal(risk[:, 0], -risk[:, 2])


def test_heteroscedastic_null_retains_dependence_despite_zero_mean_prediction():
    _, truth, _ = make_independent_splits(874, 'heteroscedastic_null', n_train=3000, n_val=2, n_test=3)
    audit = truth['train']
    p, g = audit['pathology_latent_audit_only'][:, 0], audit['rna_latent_audit_only'][:, 0]
    assert not np.any(audit['risk_by_bin_audit_only'])
    assert abs(np.corrcoef(p, g)[0, 1]) < .1
    small, large = p[np.abs(g) < .4], p[np.abs(g) > 1.5]
    assert len(small) > 500 and len(large) > 200
    assert np.var(large)/np.var(small) > 20
    assert np.corrcoef(p*p, g*g)[0, 1] > .4


def test_composition_plus_rna_has_independent_distinct_drivers():
    _, truth, _ = make_independent_splits(875, 'composition_plus_rna', n_train=2000, n_val=2, n_test=3)
    audit = truth['train']
    mixture, g = 2*audit['realized_group_fraction_audit_only']-1, audit['rna_latent_audit_only'][:, 0]
    assert abs(np.corrcoef(mixture, g)[0, 1]) < .1
    assert np.allclose(audit['risk_by_bin_audit_only'][:, 0], 1.8*mixture+g)
    assert np.std(mixture) > .3 and np.std(g) > .9


def test_redundant_nonlinear_is_only_a_pathology_main_effect():
    _, truth, _ = make_independent_splits(876, 'redundant_nonlinear', n_train=1500, n_val=2, n_test=3)
    audit = truth['train']
    p, g = audit['pathology_latent_audit_only'][:, 0], audit['rna_latent_audit_only'][:, 0]
    assert np.corrcoef(p, g)[0, 1] > .93
    assert np.allclose(audit['risk_by_bin_audit_only'][:, 0], 2.2*(p*p/(1+p*p)-.35))


def test_reject_invalid_dimensions_scenario_and_undetectable_mismatch():
    with pytest.raises(ValueError):
        make_independent_splits(1, 'typo')
    with pytest.raises(ValueError):
        make_independent_splits(1, 'xor', n_test=1, mismatch_test=True)
    with pytest.raises(ValueError):
        make_independent_splits(1, 'xor', min_patches=5, max_patches=4)
