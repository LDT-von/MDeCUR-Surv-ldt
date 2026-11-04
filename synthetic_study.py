"""Synthetic-only, fixed-configuration stress test. No external data are loaded."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import torch
from model import survival_nll
from pipeline import Config, fit_outer_train, select_and_fit


def generate(seed, scenario, n=240, intervals=3):
    if scenario not in ('null', 'additive', 'interaction', 'correlated_additive'):
        raise ValueError('Unknown scenario')
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 4)).astype('float32')
    g = rng.normal(size=(n, 4)).astype('float32')
    if scenario == 'correlated_additive':
        g[:, 0] = 0.8*p[:, 0] + 0.6*g[:, 0]
    x = p[:, None, :] + rng.normal(0, 0.25, (n, 6, 4)).astype('float32')
    if scenario == 'null':
        risk = np.zeros(n)
    elif scenario == 'interaction':
        risk = 1.3*p[:, 0]*g[:, 0]
    else:
        risk = 0.8*p[:, 0] - 0.8*g[:, 0]
    # Discrete event process; censor occurs at the end of a bin.
    hazard = 1/(1+np.exp(-(-1.3 + risk)))
    event_bin = np.full(n, intervals, dtype=int)
    for k in range(intervals):
        happened = (rng.random(n) < hazard) & (event_bin == intervals)
        event_bin[happened] = k
    censor_bin = rng.integers(0, intervals+1, n)
    observed_event = (event_bin < intervals) & (event_bin <= censor_bin)
    observed_bin = np.where(observed_event, event_bin, np.minimum(censor_bin, intervals-1))
    return (torch.tensor(x), torch.tensor(g), torch.tensor(observed_bin, dtype=torch.long),
            torch.tensor(observed_event, dtype=torch.float32), torch.ones(n,6,dtype=torch.bool))


def run(seed, scenario, cfg, selection=False):
    x,g,t,e,mask = generate(seed, scenario)
    rng = np.random.default_rng(seed+10000)
    order = rng.permutation(len(x))
    train, test = order[:160], order[160:]
    ids = [f'synthetic-{i}' for i in train]
    variants = {
        'additive': (False, replace(cfg, alpha=0, lambda_within=0, lambda_common=0, lambda_unique=0)),
        'interaction': (True, replace(cfg, alpha=0, lambda_within=0, lambda_common=0, lambda_unique=0)),
        'decur_interaction': (True, replace(cfg, alpha=0)),
        'full': (True, cfg),
    }
    result = {}
    for name, (interaction, settings) in variants.items():
        args = (x[train],g[train],t[train],e[train],mask[train],ids,3,settings)
        if selection:
            model, mu, sd, teacher, folds, audit, selection_audit = select_and_fit(*args,interaction=interaction)
        else:
            model, mu, sd, teacher, folds, audit = fit_outer_train(*args,interaction=interaction)
            selection_audit = None
        model.eval()
        with torch.no_grad():
            logits = model(x[test], (g[test]-mu)/sd, mask[test])['logits']
            nll = survival_nll(logits,t[test],e[test]).item()
            shuffle_rng = np.random.default_rng(seed+20000)
            shuffled = []
            for _ in range(10):
                perm = shuffle_rng.permutation(len(test))
                broken = model(x[test], (g[test][perm]-mu)/sd, mask[test])['logits']
                shuffled.append(survival_nll(broken,t[test],e[test]).item())
        result[name] = {'heldout_nll':nll, 'train_n':len(train), 'test_n':len(test),
                        'teacher_oof_n':0 if teacher is None else len(teacher),
                        'shuffled_omics_nll':shuffled,
                        'selection_audit':selection_audit}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, help='New JSON output path; existing files are not overwritten')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--teacher-epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11,23,37])
    parser.add_argument('--select-epochs', action='store_true')
    parser.add_argument('--interaction-l2', type=float, default=0.0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    cfg = Config(epochs=args.epochs, teacher_epochs=args.teacher_epochs, interaction_l2=args.interaction_l2, alpha=1.0, lambda_within=.0001,
                 lambda_common=.0001, lambda_unique=.0001)
    data = {'purpose':'synthetic stress test; no patient data', 'config':asdict(cfg), 'runs':{}}
    for scenario in ('null','additive','interaction','correlated_additive'):
        data['runs'][scenario] = {}
        for seed in args.seeds:
            data['runs'][scenario][str(seed)] = run(seed,scenario,replace(cfg,seed=seed),args.select_epochs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:
        json.dump(data,handle,indent=2,ensure_ascii=False)
    print(f'Wrote synthetic results to {output}')


if __name__ == '__main__':
    main()
