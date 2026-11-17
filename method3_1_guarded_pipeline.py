"""Method 3.1 with a genuinely independent, patient-level holdout gate.

Model/epoch selection and the gate use disjoint patients. Once the gate is
observed no component is refit. This module does not load patient files.
"""
import copy
import itertools

import torch

from method3_1_active_head import ActiveHierarchicalSurvivalHead
from method3_1_hierarchical_head import HierarchicalSurvivalHead
from method3_1_holdout_gate import evaluate_holdout_gate
from method3_1_interaction_features import GlobalResidualInteractions
from method3_1_model import _ids, _matrix
from method3_1_strict_pipeline import Method31StrictPipeline


# Both arms see the same six interaction input columns. Off truly masks those
# weights, while retaining the main-effects-only cases missing in the old grid.
OFF_CANDIDATES = tuple((c, m, None) for c, m in itertools.product((0., .1), (.1, 1.))) + (
    (0., None, None), (.1, None, None))
ON_CANDIDATES = tuple(itertools.product((0., .1), (.1, 1.), (.1, 1.)))


def _split_arguments(values, name, rna_dim):
    if not isinstance(values, (tuple, list)) or len(values) != 5:
        raise ValueError(f'{name} must be (bags, RNA, patient IDs, labels, censorship)')
    bags, rna, ids, labels, censorship = values
    bags = list(bags)
    if not bags:
        raise ValueError(f'{name} requires at least one patient')
    identities = _ids(ids, len(bags))
    rna = _matrix(rna, len(bags), rna_dim, f'{name} RNA')
    for i, bag in zip(identities, bags):
        if hasattr(bag, 'patient_id') and str(bag.patient_id) != i:
            raise ValueError(f'{name} patient ID differs from bag identity')
    return bags, rna, identities, labels, censorship


def _split_blocks(blocks, first):
    return ({name: value[:first] for name, value in blocks.items()},
            {name: value[first:] for name, value in blocks.items()})


class GuardedMethod31Pipeline:
    """One selected 3.1 predictor, fitted without reusing the gate outcomes.

    Selection patients choose each head and epoch; distinct gate patients
    compare the two already-frozen heads. The gate cannot update either head,
    the upstream representation, normalization, or probe. A separate test ID
    list is optional; when present predictions are limited to that set.
    """
    def __init__(self, mapper, *, mode='tanh_outer', interaction_mode='per_bin'):
        if not isinstance(mapper, GlobalResidualInteractions) or mapper.probe is None:
            raise ValueError('A training-fitted interaction mapper is required')
        if mode not in ('diagonal', 'outer', 'tanh_outer', 'raw_outer'):
            raise ValueError('An explicit, supported interaction mode is required')
        self.mapper = mapper
        self.mode, self.interaction_mode = mode, interaction_mode
        self.builder = Method31StrictPipeline(mapper, mode=mode, interaction_mode=interaction_mode)
        self.train_ids = tuple(mapper.train_ids)
        self.selection_ids = self.gate_ids = ()
        self.test_ids = None
        self.head = None
        self._diagnostic_heads = None
        self.result = None
        self._attempted = False

    def fit(self, train_labels, train_censorship, train_ids, *, selection, gate,
            test_ids=None, alpha=.05, min_gain=.005, min_n=30, min_events=5,
            std_floor=1e-12, epochs=160, lr=.03, eval_every=5, retain_candidates=False):
        if self._attempted:
            raise RuntimeError('Use a fresh pipeline for each fit')
        if not isinstance(retain_candidates, bool):
            raise ValueError('retain_candidates must be boolean')
        if _ids(train_ids, len(self.train_ids)) != self.train_ids:
            raise ValueError('Training IDs must match the fitted mapper in order')
        sel = _split_arguments(selection, 'Selection split', self.mapper.backbone.rna_dim)
        cal = _split_arguments(gate, 'Gate split', self.mapper.backbone.rna_dim)
        reserved = None if test_ids is None else _ids(test_ids, len(test_ids))
        groups = [set(self.train_ids), set(sel[2]), set(cal[2])]
        if reserved is not None:
            groups.append(set(reserved))
        if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
            raise ValueError('Training, selection, gate and reserved test IDs must be disjoint')
        bank_ids = set(map(str, self.mapper.backbone.bank.metadata.get('fit_patient_ids', [])))
        if bank_ids and not bank_ids.issubset(set(self.train_ids)):
            raise ValueError('Morphology bank contains patients outside the training split')
        if self.mapper.val_ids and tuple(self.mapper.val_ids) != sel[2]+cal[2]:
            raise ValueError('Mapper validation IDs do not match selection plus gate IDs')
        self._attempted = True

        mapped_train = self.builder._fit_moments()
        train_blocks = self.builder._train_blocks(mapped_train)
        combined_bags, combined_rna = sel[0]+cal[0], torch.cat((sel[1], cal[1]), dim=0)
        combined_ids = sel[2]+cal[2]
        mapped = self.mapper.validation_features(combined_bags, combined_rna, combined_ids)
        _, encoded, rna = self.mapper.backbone._input(combined_bags, combined_rna, combined_ids)
        # Strict main/interaction blocks do not consume cluster predictions;
        # only the KRR maps computed inside the frozen mapper are used.
        selection_blocks, gate_blocks = _split_blocks(self.builder._blocks(mapped, encoded, rna, None), len(sel[2]))
        bins = self.mapper.backbone.bins
        off = ActiveHierarchicalSurvivalHead(bins=bins, interaction_mode=self.interaction_mode,
                                            min_select_epoch=0, require_interaction=False)
        on = ActiveHierarchicalSurvivalHead(bins=bins, interaction_mode=self.interaction_mode,
                                           min_select_epoch=5, require_interaction=True)
        common = dict(epochs=epochs, lr=lr, eval_every=eval_every)
        off_selection = off.fit(train_blocks, train_labels, train_censorship, self.train_ids,
                                selection_blocks, sel[3], sel[4], sel[2], candidates=OFF_CANDIDATES, **common)
        on_selection = on.fit(train_blocks, train_labels, train_censorship, self.train_ids,
                              selection_blocks, sel[3], sel[4], sel[2], candidates=ON_CANDIDATES, **common)
        if not all(torch.equal(off.mean[x], on.mean[x]) and torch.equal(off.scale[x], on.scale[x])
                   for x in ('clr', 'other', 'interaction')):
            raise RuntimeError('Candidate heads did not use identical training-only feature moments')
        if bool(torch.count_nonzero(off.weights['interaction'])):
            raise RuntimeError('The no-interaction candidate was not hard masked')
        off_logits = off.predict(gate_blocks, cal[2])
        on_logits = on.predict(gate_blocks, cal[2])
        decision = evaluate_holdout_gate(off_logits, on_logits, cal[3], cal[4], cal[2],
            train_ids=self.train_ids, validation_ids=sel[2], test_ids=reserved,
            alpha=alpha, min_gain=min_gain, min_n=min_n, min_events=min_events, std_floor=std_floor)
        self.head = on if decision['enabled'] else off
        if retain_candidates:
            # Optional in-memory audit only: checkpoints intentionally contain
            # the selected head, never both candidate heads.
            self._diagnostic_heads = (off, on)
        self.selection_ids, self.gate_ids, self.test_ids = sel[2], cal[2], reserved
        self.result = {'selected': 'on' if decision['enabled'] else 'off', 'gate': decision,
                       'off_selection': off_selection, 'on_selection': on_selection,
                       'selection_rule': 'each head tuned only on selection split; gate compared fixed heads once',
                       'gate_outcomes_used_for_refit': False, 'off_candidates': OFF_CANDIDATES,
                       'on_candidates': ON_CANDIDATES}
        return copy.deepcopy(self.result)

    @torch.no_grad()
    def predict(self, bags, rna, ids):
        if self.head is None:
            raise RuntimeError('Fit the guarded pipeline first')
        if isinstance(ids, (str, bytes)):
            raise ValueError('Provide one patient ID per bag')
        bags, ids = list(bags), tuple(ids)
        identities = _ids(ids, len(bags))
        if set(identities) & (set(self.train_ids) | set(self.selection_ids) | set(self.gate_ids)):
            raise ValueError('Prediction IDs overlap training, selection or gate patients')
        if self.test_ids is not None and not set(identities).issubset(self.test_ids):
            raise ValueError('Prediction IDs are outside the reserved test set')
        blocks = self.builder._fresh_blocks(bags, rna, identities, validation=False)
        return self.head.predict(blocks, identities)

    def checkpoint(self):
        """Return a single-selected-head snapshot; KRR state may contain support inputs."""
        if self.head is None:
            raise RuntimeError('Fit the guarded pipeline before checkpointing')
        return {'version': 'method3_1_guarded_pipeline_v1', 'mode': self.mode,
                'interaction_mode': self.interaction_mode, 'mapper': self.mapper.checkpoint(),
                'head': self.head.checkpoint(), 'selection_ids': self.selection_ids,
                'gate_ids': self.gate_ids, 'test_ids': self.test_ids, 'result': copy.deepcopy(self.result),
                'pooled_mean': self.builder.pooled_mean.clone(), 'pooled_scale': self.builder.pooled_scale.clone(),
                'clr_mean': self.builder.clr_mean.clone(), 'clr_scale': self.builder.clr_scale.clone()}

    @classmethod
    def from_checkpoint(cls, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get('version') != 'method3_1_guarded_pipeline_v1':
            raise ValueError('Unsupported guarded pipeline checkpoint')
        mapper = GlobalResidualInteractions.from_checkpoint(snapshot['mapper'])
        model = cls(mapper, mode=snapshot['mode'], interaction_mode=snapshot['interaction_mode'])
        model.head = HierarchicalSurvivalHead.from_checkpoint(snapshot['head'])
        model.selection_ids, model.gate_ids = tuple(snapshot['selection_ids']), tuple(snapshot['gate_ids'])
        model.test_ids = None if snapshot['test_ids'] is None else tuple(snapshot['test_ids'])
        if (model.head.train_ids != model.train_ids or model.head.val_ids != model.selection_ids
                or mapper.val_ids != model.selection_ids+model.gate_ids):
            raise ValueError('Checkpoint patient split identities do not match')
        groups = [set(model.train_ids), set(model.selection_ids), set(model.gate_ids)]
        if model.test_ids is not None:
            groups.append(set(model.test_ids))
        if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
            raise ValueError('Checkpoint patient split overlap')
        model.builder.pooled_mean, model.builder.pooled_scale = snapshot['pooled_mean'].clone(), snapshot['pooled_scale'].clone()
        model.builder.clr_mean, model.builder.clr_scale = snapshot['clr_mean'].clone(), snapshot['clr_scale'].clone()
        model.result, model._attempted = copy.deepcopy(snapshot['result']), True
        if model.result['selected'] not in ('on', 'off') or bool(model.result['gate']['enabled']) != (model.result['selected'] == 'on'):
            raise ValueError('Checkpoint gate decision is inconsistent')
        return model
