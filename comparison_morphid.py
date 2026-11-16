"""Fold-0 comparison adapter for the frozen MorphID candidate.

This module bridges the frozen MorphID design and the locked one-fold
comparison protocol.  It changes no protocol constant, no metric and no
existing arm.  The adapter exists because MorphID is a frozen-representation
model: its morphology basis and its patient-axis cross-fitted probes are
fitted once on the training fold without outcomes, and only the discrete
survival head is trained.  The other arms instead forward the whole model in
every epoch and backpropagate through their encoders.

Two properties are load-bearing and are checked by the tests:

* The frozen fit accepts training-fold patients only.  Validation and outer
  test patients never reach the morphology bank, the standardization, the
  probe folds or the head.  A patient identifier that was used for fitting is
  refused at inference time.
* The head is the only object that receives an outcome gradient.  The frozen
  digest is compared before and after head training.

The cluster count and the latent dimension are fixed constants supplied by the
caller and recorded in the run metadata.  They are not selected with
validation outcomes, and no R-squared threshold removes a cluster: a low
held-out R-squared is a statement about one predictor family, not evidence of
independence.
"""
import torch
from torch import nn

from cancer_morphid_model import (FrozenMorphologyBank, MorphIDSurvival,
                                  MORPHID_VERSION)
from cancer_parity import DISCRETE_NLL_VERSION


MORPHID_ADAPTER_VERSION = 'morphid_fold0_adapter_v1'

# Fixed, pre-declared shape constants.  They are never tuned on validation
# outcomes and they are recorded in the run metadata for the plan lock.
MORPHID_CLUSTERS = 4
MORPHID_LATENT_DIM = 8
MORPHID_PROBE_FOLDS = 4
MORPHID_INTERACTION_RANK = 8
MORPHID_MIN_PROBE_PATIENTS = 8


class MorphIDArm(nn.Module):
    """Frozen MorphID presented as a comparable arm.

    The arm owns the frozen fit and exposes the same surface the other arms
    use: ``head_parameters`` for the optimizer, ``adaptation_metadata`` for
    the report and ``predict`` for evaluation.  The head consumes cached
    patient-level out-of-fold features, so the per-epoch patch resampling that
    the other arms perform cannot change what this arm learns; that is a
    property of the frozen design and is reported rather than hidden.
    """

    def __init__(self, path_dim, rna_dim, bins=4, *, clusters=MORPHID_CLUSTERS,
                 latent_dim=MORPHID_LATENT_DIM, probe_folds=MORPHID_PROBE_FOLDS,
                 interaction=True, interaction_rank=MORPHID_INTERACTION_RANK,
                 min_probe_patients=MORPHID_MIN_PROBE_PATIENTS, seed=3,
                 probe_config=None):
        super().__init__()
        self.path_dim = int(path_dim)
        self.rna_dim = int(rna_dim)
        self.bins = int(bins)
        self.clusters = int(clusters)
        self.latent_dim = int(latent_dim)
        self.probe_folds = int(probe_folds)
        self.interaction = bool(interaction)
        self.interaction_rank = int(interaction_rank)
        self.min_probe_patients = int(min_probe_patients)
        self.seed = int(seed)
        self.probe_config = dict(probe_config or {'alpha': 1e-3})
        self.model = None
        self.head_report = None
        self.frozen_reference = None
        self.bank_metadata = None
        self.bank_latent_dim = None
        self.fitted = False

    # ---------------------------------------------------------------- fitting

    def fit_frozen(self, train_bags, train_rna, train_ids, *, chunk_size=4096):
        """Fit the morphology bank and the cross-fitted probes on the training fold.

        No survival outcome is accepted by this call, and no validation or
        test patient may be present in ``train_bags``.
        """
        bags = list(train_bags)
        identities = tuple(str(value) for value in train_ids)
        if len(identities) != len(bags):
            raise ValueError('MorphID fitting needs one identifier per training bag')
        if self.fitted:
            raise RuntimeError('A fitted MorphID arm cannot be refitted; create a new arm')
        bank = FrozenMorphologyBank.fit(
            bags, ids=identities, clusters=self.clusters, latent_dim=self.latent_dim,
            seed=self.seed)
        self.bank_metadata = dict(bank.metadata)
        # The bank may reduce the latent dimension when the sampled patches do
        # not support it.  The realised value is recorded, not the request.
        self.bank_latent_dim = int(bank.latent_dim)
        model = MorphIDSurvival(
            bank, self.rna_dim, bins=self.bins, interaction_rank=self.interaction_rank,
            interaction=self.interaction, probe_config=self.probe_config,
            min_probe_patients=self.min_probe_patients, seed=self.seed)
        model.fit_frozen(bags, train_rna, ids=identities, folds=self.probe_folds,
                         seed=self.seed, chunk_size=chunk_size)
        self.model = model
        self.frozen_reference = model.frozen_digest()
        self.fitted = True
        return self

    def check_frozen_unchanged(self):
        """Reject a run whose frozen components changed during head training."""
        if self.model is None:
            raise RuntimeError('MorphID has not been fitted')
        if self.model.frozen_digest() != self.frozen_reference:
            raise RuntimeError('Frozen morphology or probe state changed during head training')
        # The shared comparison loop trains the head itself, so the arm records
        # the verified frozen state rather than a separate head-training report.
        self.record_head_report({
            'optimizer_scope': 'survival_head_only',
            'frozen_state_unchanged': True,
            'head_trained_by': 'shared comparison training loop',
            'head_parameters': self.named_head_parameters(),
        })
        return True

    # -------------------------------------------------------------- training

    def head_parameters(self):
        if self.model is None:
            raise RuntimeError('MorphID has not been fitted')
        return list(self.model.head.parameters())

    def named_head_parameters(self):
        """Qualified parameter names, so the frozen scope can be asserted."""
        if self.model is None:
            raise RuntimeError('MorphID has not been fitted')
        return [f'head.{name}' for name, _ in self.model.head.named_parameters()]

    def training_features(self):
        """The cached patient-level out-of-fold features used to train the head."""
        if self.model is None:
            raise RuntimeError('MorphID has not been fitted')
        features = self.model._training_features
        if features is None:
            raise RuntimeError('MorphID has no cached training features')
        return features

    def head_logits(self, indices):
        """Forward the head on the cached features of the selected patients."""
        features = self.training_features()
        selected = {key: value[indices] for key, value in features.items()}
        logits, interaction = self.model.head(selected)
        return logits, interaction

    def record_head_report(self, report):
        self.head_report = report
        return report

    # -------------------------------------------------------------- inference

    def predict(self, bags, rna, ids, *, chunk_size=4096):
        """Score patients that were never used to fit the frozen components."""
        if self.model is None:
            raise RuntimeError('MorphID has not been fitted')
        with torch.no_grad():
            output = self.model(list(bags), rna, ids=[str(v) for v in ids],
                                chunk_size=chunk_size)
        return output

    # -------------------------------------------------------------- reporting

    def adaptation_metadata(self):
        if self.model is None:
            raise RuntimeError('MorphID has not been fitted')
        audit = self.model.audit
        return {
            'label': 'MorphID-frozen-conditional-residual',
            'architecture': MORPHID_VERSION,
            'adapter': MORPHID_ADAPTER_VERSION,
            'loss_version': DISCRETE_NLL_VERSION,
            'trainable_scope': 'discrete survival head only',
            'frozen_components': ['morphology bank', 'patch standardization', 'cluster prototypes',
                                  'per-cluster cross-fitted RNA probe', 'feature normalization'],
            'probe_reconstruction_target': 'frozen pathology representation, never a survival label',
            'cluster_count': self.clusters,
            'latent_dim': self.latent_dim,
            'realised_latent_dim': self.bank_latent_dim,
            'probe_folds': self.probe_folds,
            'interaction_enabled': self.interaction,
            'interaction_rank': self.interaction_rank,
            'cluster_count_selected_with_outcomes': False,
            'r2_used_as_gate': False,
            'independence_established': False,
            'prediction_residual_is_independent_signal': False,
            'composition_is_closed_with_k_minus_1_freedom': True,
            'morphology_basis': audit.get('morphology_basis'),
            'decomposition_fit_uses_survival_labels': audit.get('decomposition_fit_uses_survival_labels'),
            'frozen_state_unchanged': bool((self.head_report or {}).get('frozen_state_unchanged', False)),
            'per_epoch_patch_resampling_affects_this_arm': False,
            'limits': [
                'Frozen morphology bank and probes are fitted on the training fold only; the '
                'morphology basis is not a fully nested cross-validation of the whole pipeline.',
                'A low held-out probe R2 is a statement about this predictor family. It is not '
                'evidence of independence and not evidence of biologically unique information.',
                'The prediction residual is an RNA-conditioned estimate, not an identified private '
                'component, and the product term is not a PID synergy estimate.',
                'The CLR composition is closed and has K-1 degrees of freedom, so its coefficients '
                'are not independent absolute tissue effects.',
                'The head trains on cached out-of-fold features, so the per-epoch patch resampling '
                'used by the other arms does not change what this arm learns.',
                'This is an adapted candidate on generated-data-verified code. It is not a '
                'reproduction of any published method and carries no clinical claim.',
            ],
        }

    def probe_diagnostics(self):
        """Per-cluster probe audit for the run report."""
        if self.model is None:
            return {}
        rows = []
        for entry in self.model.audit.get('probe_clusters', []):
            rows.append({key: entry.get(key) for key in
                         ('cluster', 'fitted', 'reason', 'valid_patients', 'oof_r2')})
        return {'probe_clusters': rows,
                'frozen_state_unchanged': bool((self.head_report or {}).get('frozen_state_unchanged', False)),
                'fit_patient_ids': list((self.bank_metadata or {}).get('fit_patient_ids', [])),
                'head_training': self.head_report}
