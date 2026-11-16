"""Fixed generated-data comparison of the two token interaction modules.

CPU only; all data are generated in this file. On identical modality-independent
inputs and matched frozen references, the two modules should be equivalent; the
intentional RNA-conditioned pathology arm is an input-contract sensitivity
test, not a reproduction of the project's actual RNA-conditioned matcher.

Generate discrete event time from known hazards and independent end-of-bin
censoring. The repository's alpha=.5 objective downweights censored rows, so
report it separately from the ordinary alpha=0 observed likelihood. No PID or
patient-benefit claim is made. Run once with an unoccupied --out path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cancer_dense_model import CenteredTokenInteraction
from cancer_independent_interaction import IndependentCenteredInteraction
from cancer_parity import DISCRETE_NLL_VERSION, discrete_nll, discrete_survival

SEEDS = (17, 29, 43)
SCENARIOS = ('xor', 'redundancy', 'path_only', 'rna_only', 'null',
             'shared_plus_xor', 'correlated_xor', 'continuous_redundancy')
DIMENSION, HIDDEN, BINS, TOKENS = 2, 8, 3, 3
TRAIN_N, VALID_N, TEST_N = 1024, 512, 2048
STEPS, LEARNING_RATE, WEIGHT_DECAY = 120, .025, .001
CONDITION_COEFFICIENT = .7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_digest(data):
    digest = hashlib.sha256()
    for name in ('path', 'rna', 'label', 'censorship', 'oracle_logits'):
        value = data[name].contiguous()
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def generate(scenario, n, seed):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 2, (n, 3), generator=generator)
    p, g, shared = (2 * raw[:, j].float() - 1 for j in range(3))
    if scenario == 'redundancy':
        g = p.clone()
    elif scenario == 'correlated_xor':
        g = torch.where(torch.rand(n, generator=generator) < .85, p, -p)
    elif scenario == 'continuous_redundancy':
        p = torch.randn(n, generator=generator)
        g = p.clone()
    if scenario in ('xor', 'correlated_xor'):
        effect = p * g
    elif scenario == 'redundancy':
        effect = p
    elif scenario == 'path_only':
        effect = p
    elif scenario == 'rna_only':
        effect = g
    elif scenario == 'shared_plus_xor':
        effect = .7 * shared + 1.0 * p * g
    elif scenario == 'continuous_redundancy':
        # Each modality alone has exactly the whole generator signal Z² - 1.
        # The product pathway can respond to it despite no need for joint data.
        effect = (.8 * (p.square()-1)).clamp(-1., 3.)
    elif scenario == 'null':
        effect = torch.zeros(n)
    else:
        raise ValueError(scenario)
    if scenario == 'shared_plus_xor':
        pn, gn = shared, shared
    else:
        pn = torch.randn(n, generator=generator)
        gn = torch.randn(n, generator=generator)
    path = torch.stack((p, pn), dim=1)[:, None, :].expand(-1, TOKENS, -1).clone()
    rna = torch.stack((g, gn), dim=1)[:, None, :].expand(-1, TOKENS, -1).clone()
    path += .05 * torch.randn(path.shape, generator=generator)
    rna += .05 * torch.randn(rna.shape, generator=generator)
    if scenario == 'continuous_redundancy':
        path[:, :, 0] = p[:, None]
        rna[:, :, 0] = p[:, None]

    oracle_logits = torch.tensor([-1.25, -1.10, -.95])[None, :] + effect[:, None]
    hazards = oracle_logits.sigmoid()
    hits = torch.rand((n, BINS), generator=generator) < hazards
    event_time = torch.full((n,), BINS, dtype=torch.long)
    for t in range(BINS):
        event_time = torch.where((event_time == BINS) & hits[:, t], t, event_time)
    # Censoring happens after a full bin; 2 also gives administrative censoring
    # at the study end when no event was observed in any bin.
    cut = torch.rand(n, generator=generator)
    censor_time = torch.where(cut < .15, 0, torch.where(cut < .30, 1, 2))
    event_observed = event_time <= censor_time
    label = torch.where(event_observed, event_time, censor_time).long()
    censorship = (~event_observed).float()
    return {'path': path, 'rna': rna, 'label': label, 'censorship': censorship,
            'oracle_logits': oracle_logits, 'effect': effect}


def fixed_reference(path, rna):
    # Exactly the same token-then-patient mean formula used by candidate.
    return path.mean(dim=1).mean(dim=0), rna.mean(dim=1).mean(dim=0)


class AdditiveHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.path = nn.Sequential(nn.Linear(DIMENSION, HIDDEN), nn.Tanh(),
                                  nn.Linear(HIDDEN, BINS))
        self.rna = nn.Sequential(nn.Linear(DIMENSION, HIDDEN), nn.Tanh(),
                                 nn.Linear(HIDDEN, BINS))

    def forward(self, path, rna):
        return self.path(path.mean(1)) + self.rna(rna.mean(1))


class ToyModel(nn.Module):
    def __init__(self, base, interaction, *, conditioned=False, ema=False, enabled=True):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.interaction = interaction
        self.conditioned = conditioned
        self.ema = ema
        self.enabled = enabled

    def forward(self, path, rna):
        source = path + CONDITION_COEFFICIENT * rna if self.conditioned else path
        interaction = self.interaction(source, rna)
        return self.base(path, rna) + (interaction if self.enabled else interaction * 0.)


def matched_models(train, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = AdditiveHeads()
        old_ind = CenteredTokenInteraction(DIMENSION, HIDDEN, BINS)
        old_ema = copy.deepcopy(old_ind)
        old_cond = copy.deepcopy(old_ind)
        candidate = IndependentCenteredInteraction(DIMENSION, HIDDEN, BINS)
    # Copy matching trainable tensors exactly, without assuming attribute names.
    source = list(old_ind.named_parameters())
    dest = list(candidate.named_parameters())
    if len(source) != len(dest):
        raise AssertionError('Interaction module trainable parameter count mismatch')
    with torch.no_grad():
        for (sn, sv), (dn, dv) in zip(source, dest):
            if sv.shape != dv.shape:
                raise AssertionError(f'Parameter shapes differ: {sn} vs {dn}')
            dv.copy_(sv)
        pmean, gmean = fixed_reference(train['path'], train['rna'])
        old_ind.path_reference.copy_(pmean)
        old_ind.rna_reference.copy_(gmean)
        old_ema.path_reference.copy_(pmean)
        old_ema.rna_reference.copy_(gmean)
        cpmean, cgmean = fixed_reference(train['path'] + CONDITION_COEFFICIENT * train['rna'], train['rna'])
        old_cond.path_reference.copy_(cpmean)
        old_cond.rna_reference.copy_(cgmean)
    candidate.fit_reference(train['path'], train['rna'])
    models = {'old_independent': ToyModel(base, old_ind),
              'candidate_independent': ToyModel(base, candidate),
              'candidate_interaction_off': ToyModel(base, copy.deepcopy(candidate), enabled=False),
              'old_ema_independent': ToyModel(base, old_ema, ema=True),
              'old_rna_conditioned_path': ToyModel(base, old_cond, conditioned=True)}
    parameter_counts = {name: sum(x.numel() for x in model.parameters())
                        for name, model in models.items()}
    if len(set(parameter_counts.values())) != 1:
        raise AssertionError(f'Parameter counts differ: {parameter_counts}')
    for model in models.values():
        model.eval()
    with torch.no_grad():
        old = models['old_independent'](train['path'], train['rna'])
        new = models['candidate_independent'](train['path'], train['rna'])
        max_difference = (old-new).abs().max().item()
    if max_difference > 2e-6:
        raise AssertionError(f'Independent reference/initial operator mismatch: {max_difference}')
    return models, parameter_counts, max_difference


def metrics(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data['path'], data['rna'])
        predicted = discrete_survival(logits)
        oracle = discrete_survival(data['oracle_logits'])
        result = {
            'weighted_objective_alpha_0p5': float(discrete_nll(logits, data['label'], data['censorship'], alpha=.5)),
            'observed_nll_alpha_0': float(discrete_nll(logits, data['label'], data['censorship'], alpha=0.)),
            'oracle_observed_nll_alpha_0': float(discrete_nll(data['oracle_logits'], data['label'], data['censorship'], alpha=0.)),
            'mean_squared_survival_error_to_generator': float((predicted-oracle).square().mean()),
            'mean_prediction_minus_generator_survival': float((predicted-oracle).mean()),
            'censor_fraction': float(data['censorship'].mean()),
        }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError('Nonfinite evaluation')
    return result, logits.detach().clone()


def fit(model, train):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    first_loss, first_grad, final_grad = None, None, None
    for step in range(STEPS):
        model.train()
        # Freeze old model's moving references to match candidate. eval() does
        # not turn gradients off; only module EMA behavior changes.
        if not model.ema:
            model.interaction.eval()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train['path'], train['rna'])
        loss = discrete_nll(logits, train['label'], train['censorship'], alpha=.5)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training objective')
        loss.backward()
        grads = [p.grad for p in model.interaction.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('Missing or nonfinite interaction gradient')
        norm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
        if step == 0:
            first_loss, first_grad = float(loss.detach()), norm
        final_grad = norm
        optimizer.step()
    return {'first_weighted_train_objective': first_loss,
            'first_interaction_gradient_norm': first_grad,
            'last_interaction_gradient_norm_before_step': final_grad,
            'legacy_reference_updates': (int(model.interaction.training_reference_updates)
                                         if hasattr(model.interaction, 'training_reference_updates') else None),
            'optimizer_steps': STEPS}


def summarize(records):
    answer = {}
    for scenario in SCENARIOS:
        rows = [r for r in records if r['scenario'] == scenario]
        answer[scenario] = {}
        for name in rows[0]['arms']:
            nll = [r['arms'][name]['test']['observed_nll_alpha_0'] for r in rows]
            mse = [r['arms'][name]['test']['mean_squared_survival_error_to_generator'] for r in rows]
            answer[scenario][name] = {'mean_observed_nll_alpha_0': statistics.mean(nll),
                                      'min_observed_nll_alpha_0': min(nll),
                                      'max_observed_nll_alpha_0': max(nll),
                                      'mean_survival_mse_to_generator': statistics.mean(mse)}
        answer[scenario]['independent_old_candidate_max_test_logit_difference'] = max(
            r['independent_old_candidate_max_test_logit_difference'] for r in rows)
        deltas = [r['arms']['candidate_independent']['test']['observed_nll_alpha_0']
                  - r['arms']['candidate_interaction_off']['test']['observed_nll_alpha_0'] for r in rows]
        answer[scenario]['candidate_minus_off_observed_nll_alpha_0'] = {
            'mean': statistics.mean(deltas), 'min': min(deltas), 'max': max(deltas)}
    return answer


def run():
    start = time.perf_counter()
    source_paths = [Path(__file__).resolve(), ROOT/'cancer_dense_model.py',
                    ROOT/'cancer_independent_interaction.py', ROOT/'cancer_parity.py']
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    records = []
    for scenario_no, scenario in enumerate(SCENARIOS):
        for seed in SEEDS:
            base_seed = 20261116 + scenario_no * 100000 + seed * 1000
            train, valid, test = (generate(scenario, n, base_seed+offset)
                                  for n, offset in ((TRAIN_N, 1), (VALID_N, 2), (TEST_N, 3)))
            models, counts, initial_difference = matched_models(train, base_seed+4)
            arms, test_logits = {}, {}
            for name, model in models.items():
                training = fit(model, train)
                validation, _ = metrics(model, valid)
                test_metrics, predicted_logits = metrics(model, test)
                state = copy.deepcopy(model.state_dict())
                clone = copy.deepcopy(model)
                clone.load_state_dict(state, strict=True)
                _, replay_logits = metrics(clone, test)
                if not torch.equal(predicted_logits, replay_logits):
                    raise AssertionError('Exact model state replay failed')
                arms[name] = {'parameters': counts[name], 'training': training,
                              'validation': validation, 'test': test_metrics,
                              'state_dict_reload_exact_logits': True}
                test_logits[name] = predicted_logits
            difference = float((test_logits['old_independent'] - test_logits['candidate_independent']).abs().max())
            if difference > 5e-5:
                raise AssertionError(f'Equivalent modules diverged unexpectedly: {difference}')
            # The old conditioned arm receives P + .7G. Holding P fixed and
            # altering G demonstrably alters its pathology input; this is a
            # deliberately constructed sensitivity test, not causal evidence.
            input_change = float((CONDITION_COEFFICIENT * (test['rna'].roll(1, 0)-test['rna'])).abs().mean())
            records.append({'scenario': scenario, 'seed': seed,
                            'generated_seeds': {'train': base_seed+1, 'validation': base_seed+2,
                                                'test': base_seed+3, 'initialization': base_seed+4},
                            'data_sha256': {'train': data_digest(train), 'validation': data_digest(valid),
                                            'test': data_digest(test)},
                            'initial_independent_max_logit_difference': initial_difference,
                            'independent_old_candidate_max_test_logit_difference': difference,
                            'conditioned_path_mean_absolute_change_when_only_rna_rolled': input_change,
                            'arms': arms})
            print(json.dumps({'scenario': scenario, 'seed': seed, 'independent_max_diff': difference,
                              'old_nll': arms['old_independent']['test']['observed_nll_alpha_0'],
                              'conditioned_nll': arms['old_rna_conditioned_path']['test']['observed_nll_alpha_0']},
                             allow_nan=False), flush=True)
    after = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    if before != after:
        raise RuntimeError('Sources changed during experiment; reject result')
    return {'version': 'generated_censored_interaction_comparison_v1',
            'scope': 'CPU generated data only; no patient/CUDA/server input.',
            'protocol': {'scenarios': SCENARIOS, 'seeds': SEEDS, 'counts': {'train': TRAIN_N,
                         'validation': VALID_N, 'fresh_test': TEST_N}, 'tokens': TOKENS,
                         'feature_dim': DIMENSION, 'bins': BINS, 'training_steps': STEPS,
                         'optimizer': 'Adam', 'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
                         'loss_version': DISCRETE_NLL_VERSION, 'training_alpha': .5,
                         'checkpoint': 'fixed final step; validation and test not used for selection',
                         'event_generation': 'conditional bin hazards; censor time independent and occurs after bin; no event by end is administratively censored',
                         'matched_reference': 'old EMA held frozen in eval mode after same training-only patient/token mean fit; candidate uses explicit frozen fit; gradients enabled',
                         'legacy_ema_arm': 'old model also trained in train mode with EMA; full-batch repeated fixed inputs make each reference update use the same mean',
                         'baseline_pairing': 'same optimization, full generated data, parameter counts and copied initial weights',
                         'interaction_off': 'candidate module retained with identical parameter count and copied initial weights but its logits multiplied by zero; non-interaction heads identically initialized and fully trained',
                         'continuous_redundancy': 'P=G=Z (standard normal) exactly at first token coordinate; hazard effect = clip(.8*(Z²−1),−1,3); either modality alone determines effect; additivity head is non-linear',
                         'conditioned_arm': 'synthetic approximation P_conditioned=P+.7*G ONLY for interaction input; additive heads still use P and G separately'},
            'summary': summarize(records), 'records': records,
            'source_sha256': before, 'sources_unchanged_during_run': True,
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                        'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                        'elapsed_seconds': time.perf_counter()-start},
            'limits': ['No patient data or evidence of clinical benefit.',
                       'Old and candidate are mathematically equivalent under paired independent inputs, matched frozen references and copied weights; tiny numerical differences are expected.',
                       'The conditioned pathology arm is a deliberately simplified RNA-leakage sensitivity demonstration, not the actual trainable matcher or full DensePathwaySurvival.',
                       'The weighted alpha=.5 training objective is not ordinary observed likelihood or necessarily calibrated to the generator.',
                       'Ground-truth survival MSE is available only because the generated event hazards are known.',
                       'Continuous redundancy deliberately permits a product branch to predict a purely redundant target; a nonzero product or gradient is not evidence of synergy.',
                       'The same-sized off arm contains noncontributing interaction parameters; it controls parameter count but does not represent an equally expressive non-interaction model.',
                       'Interaction predictions do not identify PID synergy or imply statistical independence.',
                       'Full batch fixed-last-step generated results do not test minibatch, real inputs or CUDA behavior.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='new JSON output path; existing path is rejected')
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    result = run()
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Wrote fixed generated-data receipt: {destination}', flush=True)


if __name__ == '__main__':
    main()
