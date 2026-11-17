"""End-to-end checks for the fixed independent gate sample sizes."""
import pytest
import torch

from scripts.method3_1_gate_size_study import run_one


def test_nested_gates_use_frozen_candidates_and_disjoint_splits():
    torch.set_num_threads(1)
    row = run_one(9993, 'xor')
    arms = {x['arm']: x for x in row['outcomes']}
    assert row['gate_outcomes_used_for_refit'] is False
    assert len(set(row['split_ids_sha256'].values())) == 4
    assert len(arms) == 8
    assert all(len(x['test_patientwise_nll']) == 360 for x in arms.values())
    assert arms['always_off120']['interaction_active'] is False
    for n in (120, 240, 480):
        decision = row['decisions'][str(n)]
        chosen = arms['always_on120'] if decision['enabled'] else arms['always_off120']
        assert arms[f'gate{n}']['test_nll'] == pytest.approx(chosen['test_nll'], abs=1e-12)
    assert arms['corrected600']['selection']['val_nll'] <= arms['legacy600']['selection']['val_nll'] + 1e-12
