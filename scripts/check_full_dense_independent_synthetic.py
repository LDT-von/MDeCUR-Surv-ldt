"""Matched full-Dense integration training on fixed CPU-generated data only.

Three known-answer scenarios and two seeds; no patient input or GPU. This is a
short execution/gradient/checkpoint comparison, not evidence of an advantage,
disentanglement, an information-theoretic PID quantity, or clinical benefit.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cancer_dense_independent import IndependentRegionDenseSurvival
from cancer_dense_model import DensePathwayConfig, DensePathwaySurvival, DENSE_LOSS_NAMES, dense_pathway_objective
from cancer_parity import DISCRETE_NLL_VERSION, discrete_nll, discrete_survival
from cancer_pathway_stream import LOSS_NAMES, pathway_stream_forward
from scripts.check_interaction_candidate_synthetic import generate

SCENARIOS = ('xor', 'continuous_redundancy', 'null')
SEEDS = (17, 29)
SIZES = {'train': 20, 'validation': 32, 'fresh_test': 64}
STEPS, CHUNK = 5, 16
PATH_DIM, RNA_DIM, PATCHES, BINS = 6, 8, 13, 3
PATHWAYS = [[0, 1], [2, 3], [4, 5], [6, 7]]
LEARNING_RATE, WEIGHT_DECAY = .001, .001


class GeneratedBag:
    view_source = 'single_view'

    def __init__(self, identity, values):
        self.patient_id = identity
        self.values = values
        self.n_patches, self.feature_dim = values.shape
        self.padding = 0
        self.read_calls = 0
        self.rows_read = 0

    def blocks(self, size):
        for start in range(0, self.n_patches, size):
            yield (self.patient_id, start, min(start+size, self.n_patches))

    def read_block(self, token, names):
        if tuple(names) != ('features_reference',):
            raise ValueError('Generated fixture has reference features only')
        self.read_calls += 1
        self.rows_read += token[2]-token[1]
        return (self.values[token[1]:token[2]],)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_dataset(scenario, n, seed, split):
    # Shared event-generation protocol with the separate interaction-only test.
    base = generate(scenario, n, seed)
    generator = torch.Generator().manual_seed(seed+100000)
    p = base['path'][:, :, :].mean(1)
    g = base['rna'][:, :, :].mean(1)
    path = torch.randn(n, PATCHES, PATH_DIM, generator=generator)
    path[:, :, :2] = p[:, None, :] + .2*path[:, :, :2]
    # Patient-dependent shifts create nondegenerate correlations without
    # replacing patient samples by the patch/token axis.
    path += .7*torch.randn(n, 1, PATH_DIM, generator=generator)
    rna = torch.randn(n, RNA_DIM, generator=generator)
    rna[:, :2] = g
    if scenario == 'continuous_redundancy':
        # Preserve the exact known-answer: either observed input alone contains
        # Z. Adding a patient shift here would silently turn the intended
        # redundant-data control into a noisy partial-observation scenario.
        path[:, :, 0] = p[:, None, 0]
        if not torch.equal(path[:, 0, 0], rna[:, 0]):
            raise AssertionError('Continuous redundancy must expose the same Z in both inputs')
    bags = [GeneratedBag(f'{split}:{scenario}:{seed}:{i}', path[i].clone()) for i in range(n)]
    return {'bags': bags, 'rna': rna, 'rna_mask': torch.ones(RNA_DIM, dtype=torch.bool),
            'label': base['label'], 'censorship': base['censorship'],
            'oracle_logits': base['oracle_logits'], 'raw_path': path}


def dataset_digest(data):
    digest = hashlib.sha256()
    for name in ('raw_path', 'rna', 'label', 'censorship', 'oracle_logits'):
        digest.update(name.encode())
        digest.update(data[name].contiguous().numpy().tobytes())
    for bag in data['bags']:
        digest.update(bag.patient_id.encode())
    return digest.hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_io_counts(data):
    for bag in data['bags']:
        bag.read_calls = bag.rows_read = 0


def io_counts(data):
    return {'bag_read_calls': sum(bag.read_calls for bag in data['bags']),
            'patch_rows_read': sum(bag.rows_read for bag in data['bags'])}


def grad_norm(grads):
    values = [g for g in grads if g is not None]
    if not values:
        return None
    if any(not torch.isfinite(g).all() for g in values):
        raise FloatingPointError('Nonfinite gradient')
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in values))


def estimability(stats):
    """Report actual diagnostic support, not finite loss as proof of variance."""
    info = {name: {'status': 'not_reported_by_existing_objective'} for name in LOSS_NAMES}
    info['global_common'] = {'status': ('not_estimable_skipped' if stats['common_skipped'] else 'estimable_by_common_guard'),
                             'reason': stats['common_skip_reason'], 'effective_sample_size': stats.get('global_common_ess')}
    patch = stats.get('patch_diagnostics')
    if patch is not None:
        info['patch_within'] = {'status': 'diagnostic_available', **patch}
    for name, diag in stats['pathway_diagnostics'].items():
        info[name] = {'status': 'diagnostic_available', 'diagnostic': diag}
    dense = stats['dense_diagnostics']
    extract = {
        'dense_path_within': lambda d: d['within']['path']['valid'],
        'dense_rna_within': lambda d: d['within']['rna']['valid'],
        'dense_common': lambda d: all(c['common']['valid'] for c in d['cross']),
        'dense_unique': lambda d: all(c['unique']['valid'] for c in d['cross']),
    }
    for name, check in extract.items():
        valid = sum(bool(check(row)) for row in dense)
        info[name] = {'status': 'diagnostic_available', 'patients_valid': valid,
                      'patients_invalid': len(dense)-valid, 'patients_total': len(dense)}
    return info


def survival(logits, data):
    return discrete_nll(logits, data['label'], data['censorship'], alpha=.5)


def train_arm(model, data, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    set_io_counts(data)
    history = []
    for epoch in range(1, STEPS+1):
        torch.manual_seed(seed+epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, stats, components = dense_pathway_objective(
            model, data, DensePathwayConfig(), CHUNK, seed+1000+epoch, epoch,
            survival_loss=survival, return_components=True)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite full objective')
        expected = set(LOSS_NAMES) | set(DENSE_LOSS_NAMES)
        if set(stats['auxiliary_enabled']) != expected or not all(stats['auxiliary_enabled'].values()):
            raise AssertionError('The fixed full auxiliary configuration was changed')
        # Any skipped term is recorded faithfully, never silently promoted to
        # estimable just because the total objective was finite.
        survival_grad = grad_norm(torch.autograd.grad(
            components['survival'], tuple(model.token_cooperation.parameters()),
            retain_graph=True, allow_unused=True))
        aux_grad = grad_norm(torch.autograd.grad(
            total-components['survival'], tuple(model.parameters()),
            retain_graph=True, allow_unused=True))
        total.backward()
        all_grad = grad_norm([p.grad for p in model.parameters()])
        if all_grad is None:
            raise AssertionError('No total-objective gradients')
        optimizer.step()
        history.append({'step': epoch, 'total_loss': float(total.detach()),
                        'survival_loss': float(components['survival'].detach()),
                        'auxiliary_terms': stats['terms'],
                        'auxiliary_enabled': stats['auxiliary_enabled'],
                        'dense_effective_weights': stats['dense_effective_weights'],
                        'estimability': estimability(stats),
                        'gradient_norms': {'survival_to_interaction_parameters': survival_grad,
                                           'weighted_auxiliary_all_parameters': aux_grad,
                                           'total_all_parameters': all_grad},
                        'ema_training_reference_updates': int(model.token_cooperation.training_reference_updates)})
    return {'history': history, 'optimizer_steps': STEPS,
            'training_io_including_checkpoint_replay': io_counts(data)}


def evaluate(model, data):
    model.eval()
    set_io_counts(data)
    with torch.no_grad():
        logits = pathway_stream_forward(model, data['bags'], data['rna'], CHUNK,
                                        checkpoint_chunks=False)['log_rates']
        predicted = discrete_survival(logits)
        truth = discrete_survival(data['oracle_logits'])
        values = {'weighted_objective_alpha_0p5': float(discrete_nll(logits, data['label'], data['censorship'], alpha=.5)),
                  'observed_nll_alpha_0': float(discrete_nll(logits, data['label'], data['censorship'], alpha=0.)),
                  'generator_oracle_observed_nll_alpha_0': float(discrete_nll(data['oracle_logits'], data['label'], data['censorship'], alpha=0.)),
                  'generated_truth_survival_mse': float((predicted-truth).square().mean()),
                  'censored_fraction': float(data['censorship'].mean()), 'io': io_counts(data)}
    if any(not math.isfinite(v) for k, v in values.items() if k != 'io'):
        raise FloatingPointError('Nonfinite held-out evaluation')
    return values, logits.detach().clone()


def independent_checkpoint_replay(model, cls, data, logits):
    state = copy.deepcopy(model.state_dict())
    buffer = io.BytesIO()
    torch.save({'architecture': cls.__name__, 'model_state': state}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location='cpu', weights_only=True)
    # Instantiate separately; do not deep-copy the trained model.
    replay = cls(PATH_DIM, RNA_DIM, PATHWAYS, bins=BINS)
    replay.load_state_dict(payload['model_state'], strict=True)
    replay.eval()
    _, restored = evaluate(replay, data)
    if not torch.equal(logits, restored):
        raise AssertionError('Independent checkpoint replay differs')
    return {'independent_instance': True, 'weights_only_loading': True,
            'exact_logits': True, 'max_logit_difference': float((logits-restored).abs().max()),
            'state_tensor_sha256': tensor_digest(payload['model_state'])}


def run():
    sources = (Path(__file__).resolve(), ROOT/'scripts/check_interaction_candidate_synthetic.py',
               ROOT/'cancer_dense_independent.py', ROOT/'cancer_dense_model.py',
               ROOT/'cancer_pathway.py', ROOT/'cancer_pathway_stream.py',
               ROOT/'cancer_dense_tokens.py', ROOT/'cancer_stream.py',
               ROOT/'cancer_independent_interaction.py', ROOT/'cancer_parity.py')
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    start = time.perf_counter()
    records = []
    for scenario_no, scenario in enumerate(SCENARIOS):
        for seed in SEEDS:
            base = 20261116 + scenario_no*100000 + seed*1000
            datasets = {split: create_dataset(scenario, count, base+offset, split)
                        for offset, (split, count) in enumerate(SIZES.items(), start=1)}
            idsets = [{bag.patient_id for bag in data['bags']} for data in datasets.values()]
            if any(idsets[i] & idsets[j] for i in range(3) for j in range(i+1, 3)):
                raise AssertionError('Synthetic split IDs overlap')
            torch.manual_seed(base+4)
            original = DensePathwaySurvival(PATH_DIM, RNA_DIM, PATHWAYS, bins=BINS)
            candidate = IndependentRegionDenseSurvival(PATH_DIM, RNA_DIM, PATHWAYS, bins=BINS)
            candidate.initialize_from_dense(original)
            orig_params = dict(original.named_parameters())
            candidate_params = dict(candidate.named_parameters())
            if set(orig_params) != set(candidate_params) or any(
                not torch.equal(orig_params[k], candidate_params[k]) for k in orig_params
            ):
                raise AssertionError('Expected identical initial trainable parameter values')
            param_count = sum(p.numel() for p in original.parameters())
            record = {'scenario': scenario, 'seed': seed, 'model_initialization_seed': base+4,
                      'dataset_seeds': {split: base+offset for offset, split in enumerate(SIZES, start=1)},
                      'data_sha256': {split: dataset_digest(data) for split, data in datasets.items()},
                      'no_id_overlap': True, 'same_initial_trainable_parameters': True,
                      'initial_tensor_sha256': tensor_digest(original.state_dict()),
                      'parameters_per_arm': param_count, 'arms': {}}
            for name, model, cls in (('original', original, DensePathwaySurvival),
                                     ('independent_regions', candidate, IndependentRegionDenseSurvival)):
                begin = time.perf_counter()
                train = train_arm(model, datasets['train'], base+5)
                validation, _ = evaluate(model, datasets['validation'])
                fresh, logits = evaluate(model, datasets['fresh_test'])
                checkpoint = independent_checkpoint_replay(model, cls, datasets['fresh_test'], logits)
                record['arms'][name] = {'train': train, 'validation': validation,
                                        'fresh_test': fresh, 'checkpoint_replay': checkpoint,
                                        'elapsed_seconds': time.perf_counter()-begin}
            records.append(record)
            print(json.dumps({'scenario': scenario, 'seed': seed,
                              'old_fresh_nll': record['arms']['original']['fresh_test']['observed_nll_alpha_0'],
                              'candidate_fresh_nll': record['arms']['independent_regions']['fresh_test']['observed_nll_alpha_0'],
                              'old_dense_valid_last': record['arms']['original']['train']['history'][-1]['estimability']['dense_common'],
                              'candidate_dense_valid_last': record['arms']['independent_regions']['train']['history'][-1]['estimability']['dense_common']},
                             allow_nan=False), flush=True)
    after = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    if after != before:
        raise RuntimeError('At least one source changed during execution; reject receipt')
    return {'version': 'full_dense_independent_regions_generated_train_v1',
            'scope': 'CPU generated-data full objective short training and state replay only; no patient/server/GPU.',
            'protocol': {'scenarios': SCENARIOS, 'seeds': SEEDS, 'split_sizes': SIZES,
                         'distinct_generated_patient_ids': True, 'patches_per_patient': PATCHES,
                         'padding': 0, 'path_dim': PATH_DIM, 'rna_dim': RNA_DIM,
                         'pathway_indices': PATHWAYS, 'bins': BINS,
                         'optimizer': 'AdamW', 'learning_rate': LEARNING_RATE,
                         'weight_decay': WEIGHT_DECAY, 'full_batch': True,
                         'optimizer_steps': STEPS, 'no_early_stopping': True,
                         'selection': 'final fixed optimizer step only; no validation or test selection',
                         'input_generation': 'fixed XOR / continuous P=G=Z, risk from Z² / null; original 2 signals augmented with independent nuisance path patches and RNA',
                         'censor_generation': 'independent end-of-bin censoring with administrative final-bin censoring',
                         'loss_version': DISCRETE_NLL_VERSION, 'training_alpha': .5,
                         'train_rng': 'same seed and epoch-specific torch seed and view_seed in both arms',
                         'auxiliaries': 'all original 11 plus all 4 dense terms enabled and raw terms/diagnostics recorded per step',
                         'candidate': 'pathology-only learned query regions supply ONLY token-cooperation pathology input; old RNA-conditioned predictions and direct head remain',
                         'candidate_extra_io': 'candidate uses a separate pathology-only streaming pass; measured reads include checkpoint backward replays',
                         'calibration_caveat': 'oracle survival error uses synthetic true hazard; training alpha=.5 weights censored rows less than ordinary likelihood'},
            'records': records, 'source_sha256': before,
            'sources_unchanged_during_run': True,
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu',
                        'torch_threads': torch.get_num_threads(), 'elapsed_seconds': time.perf_counter()-start},
            'limits': ['Five optimizer steps and small generated data check actual wiring only; validation and fresh NLL are not converged model performance.',
                       'Not a PID estimator or proof of identified synergy, disentanglement, or biological meaning.',
                       'The prior RNA-conditioned prediction branch remains in BOTH models.',
                       'No padding in these generated cases; existing unit tests address padding separately.',
                       'The two models use same initial parameter values and 15-term configuration, but candidate has extra pathology forward pass and distinct gradients.',
                       'Auxiliary enabled does not establish estimability; per-term measured/unknown diagnostics are recorded.',
                       'No patient data, clinical generalization or GPU evidence.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f'Output exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    receipt = run()
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Wrote full-Dense generated receipt: {destination}', flush=True)


if __name__ == '__main__':
    main()
