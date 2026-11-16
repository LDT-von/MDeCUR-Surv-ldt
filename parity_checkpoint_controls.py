"""Replay explicit parity input-zero controls without loading any data files.

These are constant-input interventions in a retained multimodal architecture,
not pure single-modality architectures. Historical checkpoints without a
control contract default to no intervention and are marked unverified.
"""
import copy

import torch


INPUT_CONTROL_VERSION = 'parity_input_zero_v1'


class ZeroPatientBag:
    """Zero content over an existing budget, retaining source provenance."""
    def __init__(self, source, num_patches):
        if type(num_patches) is not int or num_patches < 1:
            raise ValueError('A positive integer patch budget is required')
        # The underlying bag is kept so that the report can still state how
        # many patches the patient actually has; only the features are frozen.
        self._source = getattr(source, '_source', source)
        self._budget = source
        self.patient_id = source.patient_id
        self.feature_dim = source.feature_dim
        self.view_source = getattr(source, 'view_source', 'single_view')
        self.n_patches = self.num_patches = num_patches

    @property
    def padding(self):
        """The zero padding of the wrapped budget bag, not the whole budget."""
        return getattr(self._budget, 'padding', 0)

    def resample(self, seed=0):
        return self

    def blocks(self, chunk_size):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('Positive integer chunk size required')
        for start in range(0, self.n_patches, chunk_size):
            yield ('<parity-budget>', start, min(start+chunk_size, self.n_patches))

    def read_block(self, token, names):
        if tuple(names) != ('features_reference',) or len(token) != 3:
            raise ValueError('Zero bag supports reference rows only')
        path, start, stop = token
        if path != '<parity-budget>' or not 0 <= start < stop <= self.n_patches:
            raise ValueError('Invalid zero-bag block')
        return (torch.zeros((stop-start, self.feature_dim), dtype=torch.float32),)

    def assert_unchanged(self):
        checker = getattr(self._source, 'assert_unchanged', None)
        if checker:
            checker()

    def close(self):
        closer = getattr(self._source, 'close', None)
        if closer:
            closer()

    @property
    def provenance(self):
        return copy.deepcopy(getattr(self._source, 'provenance', []))


def apply_checkpoint_controls(checkpoint, bags, rna):
    """Apply saved controls to already-budgeted bags and normalized RNA.

    Returns (bags, rna, metadata). No file is opened and the supplied RNA is
    never modified in place. A saved budget must agree with every bag.
    """
    controls = checkpoint.get('controls')
    explicit = controls is not None
    if explicit and not isinstance(controls, dict):
        raise ValueError('Checkpoint controls must be a mapping')
    controls = controls or {}
    frozen = controls.get('frozen_modality', 'none')
    if frozen not in ('none', 'pathology', 'rna'):
        raise ValueError('Unknown frozen modality in checkpoint')
    budget = checkpoint.get('num_patches')
    chunk = checkpoint.get('patch_chunk_size')
    if budget is not None:
        if type(budget) is not int or budget < 1:
            raise ValueError('Invalid checkpoint patch budget')
        if any(getattr(bag, 'num_patches', getattr(bag, 'n_patches', None)) != budget for bag in bags):
            raise ValueError('Checkpoint patch budget differs from supplied bags')
    if frozen == 'pathology' and budget is None:
        raise ValueError('Frozen pathology replay requires an explicit saved patch budget')
    if not isinstance(rna, torch.Tensor) or rna.ndim != 2 or len(rna) != len(bags):
        raise ValueError('One normalized RNA row per supplied bag is required')
    verified = (explicit and controls.get('input_control_version') == INPUT_CONTROL_VERSION
                and type(budget) is int and budget > 0
                and type(chunk) is int and chunk > 0
                and isinstance(checkpoint.get('loss_version'), str))
    result_bags = [ZeroPatientBag(b, budget) for b in bags] if frozen == 'pathology' else list(bags)
    result_rna = torch.zeros_like(rna) if frozen == 'rna' else rna
    return result_bags, result_rna, {
        'frozen_modality': frozen, 'has_explicit_contract': explicit,
        'input_contract_verified': bool(verified), 'num_patches': budget,
        'patch_chunk_size': chunk, 'input_control_version': controls.get('input_control_version'),
        'interpretation': 'constant-input control in retained multimodal architecture; not a pure single-modality model'}
