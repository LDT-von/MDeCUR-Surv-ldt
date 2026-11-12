"""Local synthetic gradient geometry; NOT patient checkpoint findings."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import LOSS_NAMES, PathwayConfig
from pathway_gradient_audit import audit_batch


class SyntheticBag:
    view_source = 'single_view'
    def __init__(self, patient_id, features):
        self.patient_id, self.features = patient_id, features
        self.n_patches, self.feature_dim = features.shape
    def blocks(self, size):
        for start in range(0, self.n_patches, size):
            yield (f'{self.patient_id}.pt', start, min(start+size, self.n_patches))
    def read_block(self, block, names):
        if names != ('features_reference',):
            raise ValueError('Only synthetic reference features exist')
        return (self.features[block[1]:block[2]].clone(),)


def setup(seed, n=20, small=False):
    cfg = (PathwayConfig(hidden=4, common=2, local_dim=4, local_common=2, common_mode='stability')
           if small else PathwayConfig(common_mode='stability'))
    generator = torch.Generator().manual_seed(seed+5300)
    latent = torch.randn(n, 2, generator=generator)
    rna = torch.randn(n, 8, generator=generator)*.3
    rna[:, 0] += latent[:, 0]  # Shared nuisance is independent of event rate.
    rna[:, 1] += latent[:, 1]  # A shared prognostic variable.
    bags = []
    for i in range(n):
        features = torch.randn(6+i%3, 6, generator=generator)*.3
        features[:, 0] += latent[i, 0]
        features[:, 1] += latent[i, 1]
        bags.append(SyntheticBag(f'synthetic_{i}', features))
    rate = .08*torch.exp(.6*latent[:, 1])
    event_time = -torch.rand(n, generator=generator).clamp_min(1e-6).log()/rate
    censor_time = -torch.rand(n, generator=generator).clamp_min(1e-6).log()*18
    batch = {'bags': bags, 'rna': rna, 'time': torch.minimum(event_time, censor_time),
             'event': (event_time <= censor_time).float(), 'rna_mask': torch.ones(8, dtype=torch.bool)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PathwayLocalSurvival(6, 8, [[0, 1, 2], [2, 3, 4], [4, 5, 6, 7]],
                                    hidden=cfg.hidden, common=cfg.common, local_dim=cfg.local_dim,
                                    local_common=cfg.local_common).train()
    return model, batch, cfg


def run():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        records = []
        for seed in (17, 29, 43):
            model, batch, cfg = setup(seed)
            for noise_seed in (11, 22):
                record = audit_batch(model, batch, torch.tensor([0.]), cfg, 4, 'feature_noise',
                                     seed*1000+noise_seed, noise_seed)
                records.append({'model_seed': seed, **record})
                print(json.dumps({'model_seed': seed, 'noise_seed': noise_seed,
                                  'auxiliary_cosine': record['sum_weighted_auxiliary']['survival_graph_space']['cosine']}), flush=True)
        frequencies = {}
        for name in LOSS_NAMES:
            counts = {}
            for record in records:
                term = record['terms'][name]
                label = term['status'] if term['status'] != 'computed' else (
                    term['survival_graph_space']['direction'] or term['survival_graph_space']['status'])
                counts[label] = counts.get(label, 0)+1
            frequencies[name] = counts
        files = ('research_gradient_study.py', 'pathway_gradient_audit.py', 'cancer_pathway_stream.py',
                 'cancer_pathway.py', 'cancer_model.py', 'cancer_auxiliary.py', 'cancer_stream.py', 'model.py', 'cancer_rna.py')
        return {'evidence': 'CPU synthetic geometry only; untrained random model initializations, no patient data',
                'design': '20 synthetic subjects; one shared nuisance and one shared event-rate variable; independent censoring',
                'model_seeds': [17, 29, 43], 'noise_seeds_per_model': [11, 22],
                'records': records, 'direction_counts': frequencies,
                'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files},
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
                'limits': ['Six batches reuse three synthetic cohorts/models with two noise draws each; they are not six independent patient studies.',
                           'Directions are local training-batch Euclidean geometry, not validation benefit or an AdamW update prediction.',
                           'No term is recommended for deletion solely from these constructed results.',
                           'No training, parameter update, server access, five-fold evaluation, or patient checkpoint audit was performed.']}
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    data = run()
    with Path(args.out).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Synthetic gradient audit saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
