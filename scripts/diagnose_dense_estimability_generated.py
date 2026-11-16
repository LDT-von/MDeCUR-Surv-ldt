"""Read-only model diagnostics on fixed generated data; no optimization."""
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cancer_dense_model import DensePathwaySurvival
from cancer_pathway_stream import feature_noise_pair
from cancer_dense_tokens import dense_token_objective
from cancer_rna import rna_noise_view
from scripts.check_full_dense_independent_synthetic import create_dataset, PATH_DIM, RNA_DIM, PATHWAYS, BINS


def run():
    torch.set_num_threads(1)
    sourcefiles = ['scripts/diagnose_dense_estimability_generated.py',
                   'scripts/check_full_dense_independent_synthetic.py',
                   'scripts/check_interaction_candidate_synthetic.py',
                   'cancer_dense_model.py', 'cancer_dense_tokens.py',
                   'cancer_pathway_stream.py', 'cancer_pathway.py', 'cancer_rna.py']
    sha = lambda n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
    before = {n: sha(n) for n in sourcefiles}
    path16 = PATHWAYS + [[i % 8, (i + 1) % 8] for i in range(12)]
    records = []
    for scenario_no, scenario in enumerate(('xor', 'continuous_redundancy', 'null')):
        for seed in (17, 29):
            base = 20261116 + scenario_no * 100000 + seed * 1000
            data = create_dataset(scenario, 20, base + 1, 'train')
            torch.manual_seed(base + 4)
            m4 = DensePathwaySurvival(PATH_DIM, RNA_DIM, PATHWAYS, bins=BINS)
            torch.manual_seed(base + 4)
            m16 = DensePathwaySurvival(PATH_DIM, RNA_DIM, path16, bins=BINS)
            state4, state16 = m4.state_dict(), m16.state_dict()
            for key, value in state4.items():
                if key in state16 and value.shape == state16[key].shape:
                    state16[key] = value.clone()
            state16['pathway_queries'][:4] = state4['pathway_queries']
            m16.load_state_dict(state16, strict=True)
            torch.manual_seed(base + 6)
            # Same epoch-1 RNG prefix: two global RNA views precede dense views.
            _ = [rna_noise_view(data['rna'], .05, data['rna_mask']) for _ in range(2)]
            rv = tuple(rna_noise_view(data['rna'], .05, data['rna_mask']) for _ in range(2))
            with torch.no_grad():
                rf16 = tuple(m16.pathways(g) for g in rv)
                rf4 = tuple(m4.pathways(g) for g in rv)
                if any(not torch.equal(x[:, :4], y) for x, y in zip(rf16, rf4)):
                    raise AssertionError('first four pathway embeddings differ')
                rp = tuple(m16.local_rna_projector(x) for x in rf16)
                rp4 = tuple(m4.local_rna_projector(x) for x in rf4)
                if any(not torch.equal(a[:, :4], b) for a, b in zip(rp, rp4)):
                    raise AssertionError('first four projected tokens differ')
                for i, bag in enumerate(data['bags']):
                    h = m4.base.patch_encoder(bag.values)
                    pv = feature_noise_pair(h, bag.patient_id,
                                            ('<dense-valid-tokens>', 0, len(h)), base + 1006, .05)
                    pp = tuple(m4.local_path_projector(x) for x in pv)
                    results = {}
                    for count in (4, 16):
                        _, diag = dense_token_objective(
                            path_features=pv, rna_features=tuple(v[i, :count] for v in rf16),
                            path_projected=pp, rna_projected=tuple(v[i, :count] for v in rp),
                            common_dim=4, eps=1e-4, match_chunk_size=128)
                        pairs = []
                        for x in diag['cross']:
                            pairs.append({'pview': x['path_view'], 'gview': x['rna_view'],
                                          'unique_targets': x['unique_matched_target_tokens'],
                                          'common_valid': x['common']['valid'],
                                          'common_low_left': x['common']['low_variance_left'],
                                          'common_low_right': x['common']['low_variance_right'],
                                          'unique_valid': x['unique']['valid'],
                                          'unique_low_left': x['unique']['low_variance_left'],
                                          'unique_low_right': x['unique']['low_variance_right']})
                        results[str(count)] = {
                            'cross': pairs,
                            'all_four_common_valid': all(x['common_valid'] for x in pairs),
                            'all_four_unique_valid': all(x['unique_valid'] for x in pairs)}
                    records.append({'scenario': scenario, 'seed': seed, 'patient': i,
                                    'conditions': results})

    def summary(count):
        vals = [r['conditions'][str(count)] for r in records]
        pairs = [p for v in vals for p in v['cross']]
        return {'patients': len(vals), 'cross_view_combinations': len(pairs),
                'patients_all4_common_valid': sum(v['all_four_common_valid'] for v in vals),
                'pairs_common_valid': sum(p['common_valid'] for p in pairs),
                'pairs_unique_valid': sum(p['unique_valid'] for p in pairs),
                'unique_targets_histogram': dict(sorted(
                    Counter(str(p['unique_targets']) for p in pairs).items(), key=lambda z: int(z[0]))),
                'mean_unique_targets': statistics.mean(p['unique_targets'] for p in pairs),
                'common_low_right_pair_count': sum(p['common_low_right'] > 0 for p in pairs),
                'common_low_left_pair_count': sum(p['common_low_left'] > 0 for p in pairs),
                'common_low_right_mean_coordinates': statistics.mean(p['common_low_right'] for p in pairs)}
    if before != {n: sha(n) for n in sourcefiles}:
        raise RuntimeError('source changed during diagnostic')
    return {'version': 'dense_estimability_generated_token_count_v1',
            'scope': 'CPU generated diagnosis only; no optimization, no patient data',
            'control': 'Same 13 pathology tokens, RNA vector and projected first 4 pathways for both conditions; 16 adds 12 independently initialized pathway encoders over repeated two-gene groups. Not realistic 16 independent pathways, not evidence of patient improvement.',
            'scenarios': ['xor', 'continuous_redundancy', 'null'], 'seeds': [17, 29],
            'n_patients_per_case': 20, 'rna_views': 2, 'path_views': 2, 'path_tokens': 13,
            'rna_token_counts': [4, 16], 'projected_dim': 8, 'common_dim': 4, 'eps': 1e-4,
            'matching': 'shared cosine pathology to RNA, four view pairs; many-to-one allowed',
            'data_and_init': 'Exact full-Dense fixture synthetic generator, copied shared weights and first four pathway embeddings exactly; evaluate same initial parameters without training',
            'source_sha256': before, 'summary': {'4': summary(4), '16': summary(16)},
            'records': records}


if __name__ == '__main__':
    destination = ROOT / 'results/dense_estimability_token_count_diagnostic_20261116.json'
    if destination.exists():
        raise FileExistsError(destination)
    result = run()
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'path': str(destination), 'summary': result['summary']}, ensure_ascii=True))
