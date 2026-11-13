"""Patch-budget wrapper reproducing the reference patch sampling convention.

The reference pipeline concatenates all slides of a patient, draws a fixed
budget of patch rows (random without replacement while training, evenly spaced
while evaluating) and zero-pads the bag when the patient has fewer rows than
the budget.  This wrapper applies the same rule to any bag that already
satisfies the streaming contract, without changing the underlying files.

The padding is materialised as explicit zero rows at the end of the bag, so
the pooling sees the same number of rows as the reference implementation,
including the padded patients.
"""
import copy

import torch

from cancer_parity import select_patch_rows


class BudgetedPatientBag:
    """A patient bag restricted to a fixed number of patch rows.

    ``mode='train'`` draws the rows without replacement and keeps them sorted;
    ``mode='eval'`` takes evenly spaced rows.  A patient with fewer available
    rows than the budget is completed with zero rows, exactly like the
    reference implementation.
    """

    def __init__(self, source, num_patches, mode='train', seed=0):
        if mode not in ('train', 'eval'):
            raise ValueError('Patch budget mode must be train or eval')
        if type(num_patches) is not int or num_patches < 1:
            raise ValueError('The patch budget must be a positive integer')
        self._source = source
        self.patient_id = source.patient_id
        self.feature_dim = source.feature_dim
        self.view_source = getattr(source, 'view_source', 'single_view')
        self.num_patches = num_patches
        self.mode = mode
        self._offsets = self._slide_offsets()
        self.n_patches = num_patches
        self.padding = 0
        self._rows = []
        self.resample(seed)

    def _slide_offsets(self):
        """Map every slide of the source bag to its global row range."""
        offsets = []
        cursor = 0
        for path, start, stop in self._source.blocks(self._source.n_patches):
            offsets.append((path, cursor, cursor + (stop - start)))
            cursor += stop - start
        if cursor != self._source.n_patches:
            raise ValueError('The source bag rows do not match its declared patch count')
        return offsets

    def resample(self, seed=0):
        """Redraw the selected rows; used once per training epoch."""
        rows, self.padding = select_patch_rows(self._source.n_patches, self.num_patches,
                                                self.mode, seed, self.patient_id)
        self._rows = [int(row) for row in rows]
        return self

    @property
    def provenance(self):
        return copy.deepcopy(getattr(self._source, 'provenance', []))

    def assert_unchanged(self):
        checker = getattr(self._source, 'assert_unchanged', None)
        if checker is not None:
            checker()

    def close(self):
        closer = getattr(self._source, 'close', None)
        if closer is not None:
            closer()

    def blocks(self, chunk_size):
        """Cover exactly the budget, including the padded rows."""
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        for start in range(0, self.num_patches, chunk_size):
            yield ('<parity-budget>', start, min(start + chunk_size, self.num_patches))

    def read_block(self, token, names):
        if not isinstance(names, (tuple, list)) or tuple(names) != ('features_reference',):
            raise ValueError('Budgeted single-view features have reference only')
        if not isinstance(token, tuple) or len(token) != 3:
            raise ValueError('A block token must be a (path,start,stop) tuple')
        path, start, stop = token
        if path != '<parity-budget>':
            raise ValueError('Unexpected block token for a budgeted bag')
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= self.num_patches:
            raise ValueError('Invalid block row range')
        real_stop = min(stop, len(self._rows))
        pieces = []
        if start < real_stop:
            reader = getattr(self._source, 'read_rows', None)
            if reader is None:
                raise ValueError('The source bag does not support batched row reads')
            pieces.append(reader(self._rows[start:real_stop]))
        if real_stop < stop:
            # The reference pads a short patient bag with zero feature rows.
            pieces.append(torch.zeros((stop-real_stop, self.feature_dim), dtype=torch.float32))
        return (torch.cat(pieces, dim=0),)
