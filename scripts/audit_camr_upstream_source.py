"""Read-only audit of the official CAMR source against the delivered adapter.

The upstream repository is https://github.com/wxq-ustc/CAMR at commit
8fbae0a0bc4e9da31547b5eab5d773e6b2049e93. This script does not train, does not
touch patient data and does not modify the adapter. It only checks that the
survival objective and the output layer of the official code are what the
delivered ``comparison_camr.py`` adaptation documents, so that the adaptation
label can never be read as an official reproduction.

Usage:
    python scripts/audit_camr_upstream_source.py \
        --source /path/to/CAMR/checkout --out results/camr_upstream_audit.json

The checkout may be the upstream working tree or a bare export of the pinned
commit. Line endings are normalised before hashing because the recorded
2026-11-13 hashes were taken on a CRLF checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path

UPSTREAM_URL = 'https://github.com/wxq-ustc/CAMR'
UPSTREAM_COMMIT = '8fbae0a0bc4e9da31547b5eab5d773e6b2049e93'
SOURCE_FILES = ('networks.py', 'fusion.py', 'utils.py', 'train_test.py',
                'data_loaders.py', 'README.md')
# Recorded on 2026-11-13 from a CRLF checkout of the pinned commit.
RECORDED_CRLF_SHA256 = {
    'networks.py': '41c3881397500abc4e84f33a48c1a7b69a7ff8237efbaa9d9fc2d9a2a161da90',
    'fusion.py': '13370f73670b2e75f71177d0f679da8549cc2dbc4013b2362a63273f73446cf9',
    'utils.py': '8bb0bd341d208e3286ba8f3263d75f7eea172f4ff2960df6f761febd3461e720',
    'train_test.py': 'aef6c3a68c84701baa10652e71546a8c6b0cb2e387ed9f381507b814a04c40e2',
    'data_loaders.py': '8dd128b15e097411bdeb298aed64b73d069a147ec0f65d00d426db596e488874',
    'README.md': '25856e05b726bcd2141597f77d9a0319ba035bcd433c3632908962aede9cb464',
}


def normalise(raw):
    """Return the LF variant and the CRLF variant of a source file."""
    lf = raw.replace(b'\r\n', b'\n')
    crlf = lf.replace(b'\n', b'\r\n')
    return lf, crlf


def hash_variants(raw):
    lf, crlf = normalise(raw)
    return {
        'raw_sha256': hashlib.sha256(raw).hexdigest(),
        'lf_sha256': hashlib.sha256(lf).hexdigest(),
        'crlf_sha256': hashlib.sha256(crlf).hexdigest(),
    }


def classify(variants, recorded):
    if variants['raw_sha256'] == recorded:
        return 'raw'
    if variants['lf_sha256'] == recorded:
        return 'lf'
    if variants['crlf_sha256'] == recorded:
        return 'crlf'
    return 'no_match'


def find_markers(text, needles):
    return {needle: (needle in text) for needle in needles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Path to the CAMR checkout')
    parser.add_argument('--out', required=True, help='Where to write the audit JSON')
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f'CAMR checkout not found: {source}')

    file_records = {}
    for name in SOURCE_FILES:
        path = source / name
        if not path.is_file():
            file_records[name] = {'present': False}
            continue
        raw = path.read_bytes()
        variants = hash_variants(raw)
        recorded = RECORDED_CRLF_SHA256[name]
        file_records[name] = {
            'present': True,
            'recorded_crlf_sha256': recorded,
            **variants,
            'recorded_matches': classify(variants, recorded),
        }

    networks = (source / 'networks.py')
    utils = (source / 'utils.py')
    train_test = (source / 'train_test.py')
    options = (source / 'options.py')

    network_text = networks.read_text(errors='replace') if networks.is_file() else ''
    utils_text = utils.read_text(errors='replace') if utils.is_file() else ''
    train_text = train_test.read_text(errors='replace') if train_test.is_file() else ''
    options_text = options.read_text(errors='replace') if options.is_file() else ''

    upstream_markers = {
        'scalar_sigmoid_classifier': find_markers(
            network_text, ['self.classifier = nn.Sequential(nn.Linear(15, output_dim), nn.Sigmoid())']),
        'bounded_output_affine': find_markers(
            network_text, ['self.output_range', 'self.output_shift', 'out = out * self.output_range + self.output_shift']),
        'cox_loss1_continuous_risk_set': find_markers(
            utils_text, ['def CoxLoss1', 'R_mat[i,j] = survtime[j] >= survtime[i]',
                         'loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)']),
        'cox_loss2_index_order_risk_set': find_markers(
            utils_text, ['def CoxLoss2', 'def R_set', 'return indicator_matrix', 'torch.tril(matrix_ones)']),
        'training_uses_cox_loss2': find_markers(train_text, ['loss_cox =CoxLoss2', 'loss_cox = CoxLoss2']),
        'training_never_calls_cox_loss1': find_markers(train_text, ['CoxLoss1(']),
        'label_dim_default_one': find_markers(options_text, ["'--label_dim'", 'default=1']),
        'three_modality_loader': find_markers(
            (source / 'data_loaders.py').read_text(errors='replace')
            if (source / 'data_loaders.py').is_file() else '',
            ['x_gene', 'x_path', 'x_cna']),
    }

    # What the delivered adapter actually does, read back from the source file.
    adapter_path = Path(__file__).resolve().parent.parent / 'comparison_camr.py'
    adapter_text = adapter_path.read_text() if adapter_path.is_file() else ''
    # The adapter replaced the upstream bounded scalar Cox head with a
    # discrete-hazard head.  It does this by replacing the upstream
    # ``classifier`` with a plain ``nn.Linear(15, bins)`` and by neutralising
    # the upstream ``output_range`` / ``output_shift`` rescaling, so the
    # upstream computation itself is untouched.  The markers below follow the
    # current adapter source; an older revision of this audit looked for a
    # standalone ``self.head``, which the adapter no longer has.
    adapter_markers = {
        'discrete_hazard_head': find_markers(
            adapter_text, ['self.upstream_model.classifier = nn.Linear(15, bins)']),
        'no_sigmoid_classifier': not bool(find_markers(adapter_text, ['nn.Sigmoid())'])['nn.Sigmoid())']),
        'bounded_output_neutralised': find_markers(
            adapter_text, ['self.upstream_model.output_range.fill_(1.)',
                           'self.upstream_model.output_shift.fill_(0.)']),
        'documents_replaced_bounded_output': find_markers(
            adapter_text, ['discrete-hazard logits instead of the bounded Cox',
                           'neutralising the rescaling']),
        'declares_not_a_reproduction': find_markers(
            adapter_text, ["'official_model_reproduction': False"]),
        'no_continuous_cox_risk_set': 'survtime' not in adapter_text and 'R_mat' not in adapter_text,
        'attention_mil_added': find_markers(adapter_text, ['self.attention']),
        'two_way_fusion': find_markers(adapter_text, ['class TwoWayLowRankFusion']),
        'no_l1_weight_regularisation': 'regularize_weights' not in adapter_text,
    }

    checks = {
        'upstream_scalar_sigmoid_output_confirmed': all(upstream_markers['scalar_sigmoid_classifier'].values()),
        'upstream_bounded_output_confirmed': all(upstream_markers['bounded_output_affine'].values()),
        'upstream_continuous_cox_partial_likelihood_confirmed': all(
            upstream_markers['cox_loss1_continuous_risk_set'].values()),
        'upstream_cox_loss2_index_order_risk_set_confirmed': all(
            upstream_markers['cox_loss2_index_order_risk_set'].values()),
        'upstream_training_calls_cox_loss2': any(upstream_markers['training_uses_cox_loss2'].values()),
        'upstream_training_never_calls_cox_loss1': not any(upstream_markers['training_never_calls_cox_loss1'].values()),
        'adapter_uses_discrete_hazard_head_not_bounded_scalar': (
            adapter_markers['discrete_hazard_head'][
                'self.upstream_model.classifier = nn.Linear(15, bins)']
            and adapter_markers['no_sigmoid_classifier']
            and all(adapter_markers['bounded_output_neutralised'].values())),
        'adapter_documents_the_replacement': all(
            adapter_markers['documents_replaced_bounded_output'].values()),
        'adapter_declares_non_reproduction': adapter_markers['declares_not_a_reproduction'][
            "'official_model_reproduction': False"],
        'adapter_survival_objective_differs_from_upstream': (
            adapter_markers['no_continuous_cox_risk_set']),
        'recorded_hashes_match_this_checkout': all(
            record.get('recorded_matches', 'no_match') != 'no_match'
            for record in file_records.values() if record.get('present')),
        'all_recorded_files_present': all(record.get('present') for record in file_records.values()),
    }

    report = {
        'protocol': 'camr_upstream_source_audit',
        'date': '2026-11-16',
        'upstream_url': UPSTREAM_URL,
        'upstream_commit': UPSTREAM_COMMIT,
        'source_path': str(source),
        'line_ending_note': ('Recorded 2026-11-13 hashes were taken on a CRLF checkout; '
                             'the LF and CRLF variants are both reported so the record '
                             'is reproducible on either platform.'),
        'files': file_records,
        'upstream_markers': upstream_markers,
        'adapter_markers': adapter_markers,
        'checks': checks,
        'status': 'passed' if all(checks.values()) else 'failed',
        'conclusion': ('The official CAMR code uses a bounded scalar Cox output and a '
                       'continuous-time partial likelihood with no discretization, and '
                       'its training loop calls CoxLoss2 whose risk set ignores survival '
                       'time. The delivered adapter replaces the output layer and the '
                       'survival objective, so camr_2m is an explicit adaptation and '
                       'not an official reproduction.'),
        'limits': ('Source-only audit. It does not train, does not evaluate patients, '
                   'does not verify the upstream bundled LGG pickle, and does not make '
                   'the adapter comparable to published CAMR numbers.'),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'out': str(out), 'status': report['status'],
                      'checks': checks}, indent=1))


if __name__ == '__main__':
    main()
