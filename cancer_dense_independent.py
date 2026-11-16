"""Research-only DensePathway variant with a pathology-only interaction input.

Only the *new* token-cooperation branch changes its pathology input: it uses
the pathology-query regions already computed by pathway_stream_forward.
The legacy RNA-conditioned path, direct head and all fifteen training terms
are retained. This does not establish statistical independence or PID synergy.
"""
from cancer_dense_model import DensePathwaySurvival
from cancer_pathway import PathwayLocalSurvival
import torch


INDEPENDENT_DENSE_VERSION = 'independent_region_dense_v1'


class IndependentRegionDenseSurvival(DensePathwaySurvival):
    """Same parameters and EMA-centered interaction, different pathology input.

    Pathology region forward passes do not read the patient's RNA.  Shared
    encoder and learned queries can still be shaped by multimodal/survival
    losses, and the other prediction branch remains RNA-conditioned.
    """

    def get_extra_state(self):
        return {'architecture': INDEPENDENT_DENSE_VERSION}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Candidate checkpoint architecture version does not match')

    def initialize_from_dense(self, original):
        """Explicit weight transfer from original Dense, not checkpoint replay."""
        if type(original) is not DensePathwaySurvival:
            raise TypeError('Expected exactly the original DensePathwaySurvival')
        state = original.state_dict().copy()
        state['_extra_state'] = self.get_extra_state()
        self.load_state_dict(state, strict=True)
        return self

    def prediction(self, pooled_path, global_rna, matched_pathways, rna_tokens):
        raise RuntimeError('IndependentRegionDenseSurvival requires the pathology-region streaming hook')

    def prediction_with_pathology_regions(self, pooled_path, global_rna,
                                          matched_pathways, rna_tokens,
                                          pathology_regions):
        if pathology_regions.shape != rna_tokens.shape:
            raise ValueError('Pathology-only regions and RNA tokens must match in shape')
        legacy, legacy_interaction = PathwayLocalSurvival.prediction(
            self, pooled_path, global_rna, matched_pathways, rna_tokens)
        conditional_p, projected_r = self.local_projection(matched_pathways, rna_tokens)
        independent_p = self.local_path_projector(pathology_regions)
        c = self.local_common
        common = (conditional_p[..., :c] + projected_r[..., :c]) / 2
        direct = self.decomposition_head(torch.cat(
            (common, conditional_p[..., c:], projected_r[..., c:]), dim=-1).mean(1))
        cooperative = self.token_cooperation(independent_p[..., c:], projected_r[..., c:])
        return (legacy + direct + self.interaction_strength * cooperative,
                legacy_interaction + cooperative)

    def adaptation_metadata(self):
        result = super().adaptation_metadata()
        result.update({
            'label': 'MDeCUR-dense-independent-region-candidate-v1',
            'cooperation_pathology_input': 'pathology-query regions; no patient RNA in forward',
            'cooperation_reference': 'same training EMA as existing cooperation; not exact current mean',
            'synergy_estimator': False,
            'limits': [*result['limits'],
                       'Pathology queries and encoders may be shaped by joint outcome training.',
                       'Legacy prediction and direct head still include RNA-conditioned pathology.',
                       'Candidate regions exclude padding; legacy conditional pooling may include it.',
                       'Cooperation inputs can remain statistically dependent.',
                       'Requires generated-data and patient-level matched controls before claims.']})
        return result
