"""Structural diagnostic: does the interaction branch actually reach the prediction?

Read-only diagnostic on a frozen checkpoint.  It does not train, does not
update parameters, does not evaluate the outer test set and does not re-select
any epoch.

For every validation patient it recomputes, from the frozen checkpoint:

  base_head_logit      -- the plain pathology+RNA head contribution
  interaction_logit    -- the raw interaction branch output (pre-scaling)
  interaction_scaled   -- interaction_strength * interaction_logit
  total_logit          -- base + scaled interaction
  S(h) from base only  -- survival if the interaction branch were removed
  S(h) from total      -- the actual model survival

It then reports how much of the logit range, the risk ranking and the
12-month survival the interaction branch accounts for, and the concordance of
the base-only ranking against the full-model ranking.

If the base-only ranking is already as good as the full ranking, then the
interaction branch contributes no discriminative information on this cohort.

Usage:
    python diagnose_interaction_contribution.py \
        --checkpoint RUN_DIR/checkpoint.pt \
        --manifest MANIFEST.csv \
        --out interaction_contribution_diagnostic.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cancer_metrics import harrell_c
from cancer_model import survival_at
from cancer_pathway import PathwayLocalSurvival
from cancer_pathway_stream import PathwayConfig, pathway_stream_forward
from cancer_rna import transform_rna
from cancer_stream_data import load_stream_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True,
                        help='checkpoint.pt of a completed pathway run')
    parser.add_argument('--manifest', required=True,
                        help='train/val CSV used for that run')
    parser.add_argument('--out', required=True, help='Output JSON path')
    parser.add_argument('--horizon', type=float, default=12.0)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    report_path = run_dir / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    config = report.get('config') or checkpoint.get('config')
    if config is None:
        raise ValueError('The checkpoint does not carry a configuration')
    cfg = PathwayConfig(**{name: config[name] for name in PathwayConfig.__dataclass_fields__})
    model = PathwayLocalSurvival(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    cohort = load_stream_manifest(args.manifest, path_views='single_view', include_test=False)
    validation = cohort['val']

    fitted = {'model': model, 'rna_mean': checkpoint['rna_mean'],
              'rna_scale': checkpoint['rna_scale'], 'rna_mask': checkpoint['rna_mask'],
              'rna_train_std': checkpoint['rna_train_std'],
              'rna_preprocessing': checkpoint['rna_preprocessing'],
              'starts': checkpoint['starts'], 'model_kwargs': checkpoint['model_kwargs'],
              'pathway_mapping': checkpoint['pathway_mapping'],
              'stream_config': checkpoint['stream_config']}

    starts = torch.tensor(checkpoint['starts'], dtype=torch.float32)
    with torch.no_grad():
        rna = transform_rna(validation['rna'], fitted)
        outputs = []
        for i, bag in enumerate(validation['bags']):
            outputs.append(pathway_stream_forward(
                model, [bag], rna[i:i+1],
                checkpoint['stream_config']['patch_chunk_size'],
                checkpoint['stream_config']['view_source'],
                checkpoint_chunks=False))
        path_hidden = torch.cat([o['path_hidden'] for o in outputs])
        global_rna = torch.cat([o['rna_hidden'] for o in outputs])
        tokens = torch.cat([o['rna_pathways'] for o in outputs])
        matched = torch.cat([o['matched_path'] for o in outputs])
        log_rates = torch.cat([o['log_rates'] for o in outputs])

        # Recompute the two logit components exactly as PathwayLocalSurvival does.
        base = model.base.head(torch.cat((path_hidden, global_rna), dim=-1))
        interaction = (torch.nn.functional.gelu(model.interaction_path(matched))
                       * torch.nn.functional.gelu(model.interaction_rna(tokens))).mean(dim=1)
        extra = model.interaction_head(interaction)
        scaled = cfg.interaction_strength * extra

        query = torch.tensor([args.horizon])
        base_survival = survival_at(base, query, starts)[:, 0]
        full_survival = survival_at(log_rates, query, starts)[:, 0]

    time = validation['time'].numpy()
    event = validation['event'].numpy()
    base_risk = 1 - base_survival.numpy()
    full_risk = 1 - full_survival.numpy()

    full_c = harrell_c(time, event, full_risk)
    base_c = harrell_c(time, event, base_risk)

    # Concordance between the two rankings on comparable pairs.
    concordant = discordant = 0
    for i in range(len(time)):
        valid = (time > time[i]) & (event[i] == 1)
        for j in np.flatnonzero(valid):
            if full_risk[i] > full_risk[j]:
                concordant += int(base_risk[i] > base_risk[j])
                discordant += int(base_risk[i] < base_risk[j])
            elif full_risk[i] < full_risk[j]:
                concordant += int(base_risk[i] < base_risk[j])
                discordant += int(base_risk[i] > base_risk[j])

    base_np = base.numpy()
    scaled_np = scaled.numpy()
    result = {
        'checkpoint_run': str(run_dir),
        'patients': len(time),
        'events': int(event.sum()),
        'interaction_strength': cfg.interaction_strength,
        'interval_starts': [float(x) for x in starts],
        'prediction_horizon': float(args.horizon),
        'full_model': {
            'c_index': full_c['value'],
            'comparable_pairs': full_c['comparable_pairs'],
        },
        'base_head_only': {
            'c_index': base_c['value'],
            'comparable_pairs': base_c['comparable_pairs'],
            'logit_range': float(base_np.max() - base_np.min()),
        },
        'interaction_branch': {
            'raw_extra_abs_mean': float(extra.abs().mean()),
            'scaled_abs_mean': float(scaled.abs().mean()),
            'base_abs_mean': float(base.abs().mean()),
            'scaled_to_base_abs_ratio': float(scaled.abs().mean() / base.abs().mean()),
            'logit_range_fraction': float((scaled_np.max() - scaled_np.min())
                                         / (base_np.max() - base_np.min())),
            'base_plus_scaled': 'identical by construction',
        },
        'risk_ranking': {
            'base_vs_full_comparable_pairs': concordant + discordant,
            'concordant_pairs': concordant,
            'discordant_pairs': discordant,
            'ranking_agreement': concordant / (concordant + discordant)
                                 if (concordant + discordant) else None,
            'max_abs_survival_difference_base_vs_full': float(
                (base_survival - full_survival).abs().max()),
            'mean_abs_survival_difference': float(
                (base_survival - full_survival).abs().mean()),
        },
        'limits': [
            'Single fixed checkpoint on one validation cohort; no training, no epoch '
            'selection, no outer-test access.',
            'base_head_only C is computed by removing the interaction contribution '
            'from the same frozen parameters, not by retraining without it.',
            'This is a mechanism diagnostic, not an effect estimate for a retrained '
            'interaction-off arm.',
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
