"""Fixed-budget CPU toys for the production centered token interaction module.

Two exhaustive event-time bins, no censoring, and categorical event-time NLL.
This is NOT the parity trainer's hazard parameterization or alpha-weighted loss,
not a PID estimator, and not patient/generalization evidence for the full model.
Only the output path is configurable: seeds, samples and steps are prespecified.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cancer_dense_model import CenteredTokenInteraction


SEEDS = (17, 29, 43)
SCENARIOS = ('xor', 'redundancy', 'path_only', 'null')
TRAIN_N, TEST_N, STEPS = 4096, 8192, 300
HIDDEN, LEARNING_RATE, WEIGHT_DECAY = 8, .03, .001


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def generate(scenario, n, seed):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n, 2), generator=generator)
    if scenario == 'redundancy':
        bits[:, 1] = bits[:, 0]
    if scenario == 'xor':
        risk_bit = bits[:, 0] ^ bits[:, 1]
    elif scenario in ('redundancy', 'path_only'):
        risk_bit = bits[:, 0]
    elif scenario == 'null':
        risk_bit = None
    else:
        raise ValueError(scenario)
    early_probability = (torch.full((n,), .5) if risk_bit is None
                         else .1+.8*risk_bit.float())
    # Event bins are exhaustive: every observation is an uncensored event.
    event_bin = (torch.rand(n, generator=generator) >= early_probability).long()
    nuisance = torch.randn(n, 2, generator=generator)
    path = torch.stack((2*bits[:, 0].float()-1, nuisance[:, 0]), 1).unsqueeze(1)
    rna = torch.stack((2*bits[:, 1].float()-1, nuisance[:, 1]), 1).unsqueeze(1)
    oracle_nll = -torch.where(event_bin == 0, early_probability,
                             1-early_probability).double().log().mean().item()
    return path, rna, event_bin, oracle_nll


class AdditiveHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.path = nn.Sequential(nn.Linear(2, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 2))
        self.rna = nn.Sequential(nn.Linear(2, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 2))

    def forward(self, path, rna):
        return self.path(path.mean(1))+self.rna(rna.mean(1))


class ToyModel(nn.Module):
    def __init__(self, additive, interaction):
        super().__init__()
        self.additive = deepcopy(additive)
        self.interaction = CenteredTokenInteraction(2, HIDDEN, 2) if interaction else None

    def forward(self, path, rna):
        logits = self.additive(path, rna)
        return logits if self.interaction is None else logits+self.interaction(path, rna)


def categorical_nll(logits, targets):
    return -F.log_softmax(logits.double(), 1).gather(1, targets[:, None]).mean().item()


def full_rna_derangement(n, seed):
    # A random single-cycle derangement, not a uniform sample of all derangements.
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    permutation = torch.empty_like(order)
    permutation[order] = order.roll(1)
    if n < 2 or torch.any(permutation == torch.arange(n)):
        raise AssertionError('Expected a patient-level derangement')
    return permutation


def fit_and_score(model, train, test, permutation):
    path, rna, targets, _ = train
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    first_loss, last_step_loss = None, None
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(path, rna), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite training gradient')
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        last_step_loss = float(loss.detach())
    # No test access occurred in training or checkpoint selection: there is no selection.
    model.eval()
    before = state_digest(model)
    with torch.no_grad():
        test_path, test_rna, test_targets, oracle_nll = test
        training_nll = categorical_nll(model(path, rna), targets)
        test_nll = categorical_nll(model(test_path, test_rna), test_targets)
        shuffled_nll = categorical_nll(model(test_path, test_rna[permutation]), test_targets)
    after = state_digest(model)
    if before != after:
        raise AssertionError('Evaluation changed parameters or training reference buffers')
    updates = 0 if model.interaction is None else int(model.interaction.training_reference_updates)
    if model.interaction is not None and updates != STEPS:
        raise AssertionError('Unexpected number of training-reference updates')
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'initial_training_nll': first_loss, 'last_pre_update_training_nll': last_step_loss,
            'final_training_nll': training_nll, 'test_nll': test_nll,
            'test_full_rna_deranged_nll': shuffled_nll, 'test_oracle_nll': oracle_nll,
            'training_reference_updates': updates,
            'evaluation_state_unchanged': True, 'final_state_sha256': after}


def entropy(values):
    n = len(values)
    return -sum((v/n)*math.log2(v/n) for v in Counter(values).values())


def exact_latent_oracles():
    pairs = list(itertools.product((0, 1), repeat=2))
    examples = {
        'xor': [(p, r, p ^ r) for p, r in pairs],
        'redundancy': [(p, p, p) for p, r in pairs],
        'path_only': [(p, r, p) for p, r in pairs],
        'redundancy_plus_xor': [((d, p), (d, r), (d, p ^ r))
                               for d, p, r in itertools.product((0, 1), repeat=3)],
    }
    def mi(a, b):
        return entropy(a)+entropy(b)-entropy(list(zip(a, b)))
    result = {}
    for name, rows in examples.items():
        p, r, target = zip(*rows)
        left, right, joint = mi(p, target), mi(r, target), mi(list(zip(p, r)), target)
        result[name] = {'I_target_path_bits': left, 'I_target_rna_bits': right,
                        'I_target_joint_bits': joint, 'joint_minus_sum_bits': joint-left-right}
    if result['redundancy_plus_xor']['joint_minus_sum_bits'] != 0.:
        raise AssertionError('Known-answer information identity failed')
    return {'target': 'discrete latent oracle, NOT the sampled stochastic event bin',
            'values': result,
            'interpretation': 'Joint minus marginal MI mixes redundancy and synergy; it is not a pure synergy estimator.'}


def numeric_summary(values):
    return {'mean': statistics.mean(values), 'min': min(values), 'max': max(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else None}


def run():
    start = time.perf_counter()
    source_paths = (Path(__file__).resolve(), ROOT/'cancer_dense_model.py')
    source_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    records = []
    for scenario_number, scenario in enumerate(SCENARIOS):
        for seed in SEEDS:
            base_seed = 20261114+100000*scenario_number+1000*seed
            train = generate(scenario, TRAIN_N, base_seed+1)
            test = generate(scenario, TEST_N, base_seed+2)
            permutation = full_rna_derangement(TEST_N, base_seed+3)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(base_seed+4)
                additive = AdditiveHeads()
                models = {'additive': ToyModel(additive, False),
                          'additive_plus_token_product': ToyModel(additive, True)}
            if state_digest(models['additive'].additive) != state_digest(models['additive_plus_token_product'].additive):
                raise AssertionError('The two arms must share identical additive initialization')
            arms = {name: fit_and_score(model, train, test, permutation)
                    for name, model in models.items()}
            delta = arms['additive_plus_token_product']['test_nll']-arms['additive']['test_nll']
            record = {'scenario': scenario, 'seed': seed,
                      'data_seeds': {'train': base_seed+1, 'test': base_seed+2,
                                     'rna_permutation': base_seed+3, 'model': base_seed+4},
                      'paired_additive_initialization': True,
                      'full_rna_derangement_fixed_points': int((permutation == torch.arange(TEST_N)).sum()),
                      'derangement_sha256': hashlib.sha256(permutation.numpy().tobytes()).hexdigest(),
                      'arms': arms, 'joint_minus_additive_test_nll': delta,
                      'joint_deranged_minus_original_test_nll':
                      arms['additive_plus_token_product']['test_full_rna_deranged_nll']-arms['additive_plus_token_product']['test_nll']}
            records.append(record)
            print(json.dumps({'scenario': scenario, 'seed': seed, 'delta_test_nll': delta,
                              'additive_test_nll': arms['additive']['test_nll'],
                              'joint_test_nll': arms['additive_plus_token_product']['test_nll'],
                              'joint_deranged_test_nll': arms['additive_plus_token_product']['test_full_rna_deranged_nll']}), flush=True)
    after_hashes = {str(p.relative_to(ROOT)).replace('\\', '/'):
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if source_hashes != after_hashes:
        raise RuntimeError('A source file changed during the fixed experiment')
    summary = {}
    for scenario in SCENARIOS:
        rows = [r for r in records if r['scenario'] == scenario]
        summary[scenario] = {'joint_minus_additive_test_nll': numeric_summary([r['joint_minus_additive_test_nll'] for r in rows]),
                             'arms': {arm: {metric: numeric_summary([r['arms'][arm][metric] for r in rows])
                                            for metric in ('test_nll', 'test_full_rna_deranged_nll', 'test_oracle_nll')}
                                      for arm in rows[0]['arms']}}
    xor_rows = [r for r in records if r['scenario'] == 'xor']
    return {'evidence': 'CPU uncensored synthetic learnability and failure-boundary study only',
            'production_component': 'cancer_dense_model.CenteredTokenInteraction',
            'protocol': {'scenarios': list(SCENARIOS), 'seeds': list(SEEDS),
                         'n_train_per_scenario_seed': TRAIN_N, 'n_independent_test': TEST_N,
                         'full_batch_training_steps': STEPS, 'optimizer': 'Adam',
                         'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
                         'input_shape': '[patients, 1 token, 2 features: signed bit and independent Gaussian nuisance]',
                         'head_hidden': HIDDEN, 'joint_extra_parameters': 48,
                         'event_time_bins': 2, 'censoring': 'none; all observations have an event in bin 0 or 1',
                         'event_probability': 'P(event_bin=0)=0.1+0.8*risk_bit; null case probability=0.5',
                         'loss': 'categorical NLL over two exhaustive event-time bins; NOT parity hazard loss/alpha weighting',
                         'checkpoint_selection': 'none; final step only, no test or validation selection',
                         'rna_intervention': 'one prespecified random single-cycle patient derangement of the entire RNA token',
                         'test_used_during_optimization': False},
            'records': records, 'summary': summary,
            'checks': {'source_unchanged': True, 'test_evaluation_preserved_all_model_states': True,
                       'xor_expected_pattern_observed_all_seeds':
                       all(r['joint_minus_additive_test_nll'] < -.1 and
                           r['joint_deranged_minus_original_test_nll'] > .1 for r in xor_rows),
                       'xor_pattern_threshold_prespecified_nll': .1,
                       'other_scenarios_require_joint_win': False},
            'exact_latent_information_oracles': exact_latent_oracles(),
            'source_sha256': source_hashes,
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                        'elapsed_seconds': time.perf_counter()-start},
            'limits': ['This isolates one production interaction module, not the complete pathology/RNA framework or its auxiliaries.',
                       'The additive arm has 84 parameters; the joint arm has 132. This is not a parameter-count-matched effect estimate.',
                       'Three independently generated synthetic datasets are not clinical cohorts or independent patient evidence.',
                       'Input pathology tokens here are independent of RNA by construction; the full model can use RNA-conditioned pathology tokens.',
                       'The explicit no-censoring categorical toy differs from the parity discrete-hazard parameterization and alpha weighting.',
                       'Test-time RNA permutation is a distribution intervention, not a retrained-control or a causal effect.',
                       'No strict PID synergy, spatial correspondence, disentanglement, or clinical effectiveness is established.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New JSON result path; refuses overwrite')
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = run()
    finally:
        torch.set_num_threads(previous)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'out': str(destination), 'records': len(result['records']),
                      'checks': result['checks'], 'seconds': result['runtime']['elapsed_seconds']}))


if __name__ == '__main__':
    main()
