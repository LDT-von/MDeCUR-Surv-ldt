"""Known-answer checks for the synthetic-only end-to-end comparison runner."""
import pytest
import torch

from scripts.method3_1_gate_study import CORRECTED_GRID, OLD_GRID, run_one


def test_candidate_grid_contains_strict_off_without_removing_main():
    assert set(OLD_GRID).issubset(set(CORRECTED_GRID))
    added = set(CORRECTED_GRID)-set(OLD_GRID)
    assert len(added) == 4
    assert all(a is not None and b is not None and c is None for a,b,c in added)


def test_synthetic_run_respects_frozen_policy_and_nested_choice():
    torch.set_num_threads(1)
    row = run_one(9981,'xor')
    outcomes = {x['arm']:x for x in row['outcomes']}
    assert row['gate_outcomes_used_for_refit'] is False
    assert len(set(row['split_ids_sha256'].values())) == 4
    assert len(outcomes) == 8 and all(len(x['test_patientwise_nll'])==360 for x in outcomes.values())
    assert outcomes['always_off120']['interaction_active'] is False
    expected = outcomes['always_on120'] if row['decision']['strict']['enabled'] else outcomes['always_off120']
    assert outcomes['strict_gate120']['test_nll'] == pytest.approx(expected['test_nll'],abs=1e-12)
    assert not row['decision']['strict']['enabled'] or row['decision']['loose']['enabled']
    assert not row['decision']['loose']['enabled'] or row['decision']['direct']['enabled']
    assert outcomes['corrected240']['selection']['val_nll'] <= outcomes['legacy240']['selection']['val_nll']+1e-12
