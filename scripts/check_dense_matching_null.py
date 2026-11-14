"""Known-answer warning: NN can align independently generated token clouds."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cancer_dense_tokens import dense_token_objective
from comparison_fold0 import write_new


def check():
    generator = torch.Generator().manual_seed(17)
    # Arrays are independently sampled; no token shares a latent draw across modalities.
    a, b = [2*torch.pi*torch.rand(512, generator=generator) for _ in range(2)]
    p, r = [torch.stack((x.cos(), x.sin()), 1) for x in (a, b)]
    zp, zr = [torch.cat((x, torch.randn(512, 2, generator=generator)), 1) for x in (p, r)]
    _, info = dense_token_objective((p, p), (r, r), (zp, zp), (zr, zr), 2, match_chunk_size=128)
    pc, rc = p-p.mean(0), r-r.mean(0)
    unpaired = ((pc*rc).mean(0)/(pc.square().mean(0)*rc.square().mean(0)).sqrt()).mean().item()
    paired = info['cross'][0]['common']['mean_diagonal']
    assert abs(unpaired) < .2 and paired > .99
    return {'scope': 'Known independent token populations, no training or patient data',
            'seed': 17, 'tokens_per_modality': 512,
            'raw_index_paired_mean_diagonal': unpaired,
            'nearest_neighbor_mean_diagonal': paired,
            'shared_latent_tokens_by_construction': False,
            'conclusion': 'High post-matching common correlation is not evidence of common biological information.',
            'limits': 'Within-patient token geometry demonstration, not a clinical null test or proof every match is spurious.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    torch.set_num_threads(1)
    result = check()
    write_new(parser.parse_args().out, result)
    print(json.dumps(result, indent=2))
