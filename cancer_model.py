"""Cancer WSI/RNA survival research model; no patient data are bundled."""
import torch
from torch import nn
from torch.nn import functional as F


class CancerDeCUR(nn.Module):
    """Masked attention MIL, RNA encoder, DeCUR branches and survival head.

    Attention scores are learned pooling weights, not biological explanations.
    The final head is additive in the learned representation for capacity control.
    """
    def __init__(self, path_dim, rna_dim, hidden=32, common=16, bins=4,
                 prediction_space='projected', local_dim=None):
        super().__init__()
        if not 0 < common < hidden or bins < 1:
            raise ValueError('Require 0 < common < hidden and bins >= 1')
        if prediction_space not in ('projected', 'encoder'):
            raise ValueError('prediction_space must be projected or encoder')
        if local_dim is not None and (not isinstance(local_dim, int) or local_dim < 1):
            raise ValueError('local_dim must be a positive integer or None')
        self.common = common
        self.prediction_space = prediction_space
        self.patch_encoder = nn.Sequential(nn.Linear(path_dim,hidden), nn.LayerNorm(hidden), nn.GELU())
        self.attention = nn.Sequential(nn.Linear(hidden,hidden//2), nn.Tanh(), nn.Linear(hidden//2,1))
        self.rna_encoder = nn.Sequential(nn.Linear(rna_dim,hidden), nn.LayerNorm(hidden), nn.GELU())
        self.path_projector = nn.Linear(hidden,hidden)
        self.rna_projector = nn.Linear(hidden,hidden)
        self.head = nn.Linear(hidden*2,bins)
        # Construct after the legacy modules and preserve the augmentation RNG.
        # Enabling a local head must not change shared parameter initialization.
        self.local_projector = None
        if local_dim is not None:
            with torch.random.fork_rng(devices=[]):
                self.local_projector = nn.Linear(hidden, local_dim, bias=False)

    def _encode_backbones(self, patches, rna, mask):
        if patches.ndim != 3 or rna.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError('Expected patches [B,N,D], RNA [B,G], bool mask [B,N]')
        if mask.shape != patches.shape[:2] or len(rna) != len(patches) or not mask.any(1).all():
            raise ValueError('Invalid batch or patch mask')
        safe = patches.masked_fill(~mask.unsqueeze(-1),0)
        features = self.patch_encoder(safe)
        scores = self.attention(features).squeeze(-1).masked_fill(~mask, -torch.inf)
        attention = scores.softmax(dim=1)
        pooled = (features*attention.unsqueeze(-1)).sum(1)
        return pooled, self.rna_encoder(rna), attention, features

    def encode_auxiliary(self, patches, rna, mask, include_local=False):
        """Training embeddings; all auxiliary gradients reach the backbones.

        Local rows retain the complete, aligned bag order. Padding is excluded
        by the local loss, and a cache loader must establish row identity first.
        """
        p, r, attention, features = self._encode_backbones(patches, rna, mask)
        out = {'path': self.path_projector(p), 'rna': self.rna_projector(r),
               'attention': attention}
        if include_local:
            if self.local_projector is None:
                raise ValueError('Construct a local projector before requesting local embeddings')
            out['local'] = self.local_projector(features)
        return out

    def encode(self, patches, rna, mask):
        """Compatibility API: return the patient-level auxiliary projections."""
        out = self.encode_auxiliary(patches, rna, mask)
        return out['path'], out['rna'], out['attention']

    def forward(self, patches, rna, mask):
        hp,hr,attn,_ = self._encode_backbones(patches,rna,mask)
        if self.prediction_space == 'projected':
            p,g = self.path_projector(hp),self.rna_projector(hr)
        else:
            # Prediction in encoder mode never executes auxiliary projectors.
            p,g = hp,hr
        return {'log_rates':self.head(torch.cat((p,g),dim=1)),
                'path':p, 'rna':g, 'attention':attn,
                'path_hidden':hp, 'rna_hidden':hr}


def piecewise_exposure(time, starts):
    """Exposure per interval with exact right-censoring time, last bin infinite.

    `starts` includes zero; times and starts must share the same units. Events at
    a cut belong to the interval beginning at that cut (right-continuous rates).
    """
    if time.ndim != 1 or starts.ndim != 1 or len(starts) < 1:
        raise ValueError('Expected time [B] and interval starts [K]')
    if not torch.isfinite(time).all() or not torch.isfinite(starts).all():
        raise ValueError('Time and interval starts must be finite')
    if (time < 0).any() or starts[0] != 0 or (starts[1:] <= starts[:-1]).any():
        raise ValueError('Times nonnegative; starts strictly increasing from zero')
    exposure = (time[:,None] - starts[None,:]).clamp_min(0)
    widths = starts[1:] - starts[:-1]
    if len(widths):
        exposure = torch.cat((torch.minimum(exposure[:,:-1],widths),exposure[:,-1:]),dim=1)
    return exposure


def piecewise_nll(log_rates, time, event, starts, reduction='mean'):
    """Continuous-time piecewise-exponential event or right-censor likelihood."""
    if log_rates.ndim != 2 or log_rates.shape != (len(time),len(starts)) or event.shape != time.shape:
        raise ValueError('Incompatible shapes')
    if not torch.isfinite(log_rates).all() or not torch.all((event == 0) | (event == 1)):
        raise ValueError('Nonfinite logits or invalid event')
    exposure = piecewise_exposure(time,starts)
    rates = F.softplus(log_rates) + 1e-8
    event_bin = torch.bucketize(time.contiguous(),starts[1:].contiguous(),right=True)
    log_event_rate = rates.gather(1,event_bin[:,None]).squeeze(1).log()
    per_patient = (rates*exposure).sum(1) - event*log_event_rate
    if reduction == 'none':
        return per_patient
    if reduction == 'mean':
        return per_patient.mean()
    raise ValueError('Invalid reduction')


def survival_at(log_rates, query_times, starts):
    """Return [patient,time] survival from the learned continuous-time model."""
    if query_times.ndim != 1:
        raise ValueError('Expected one-dimensional times')
    exposure = piecewise_exposure(query_times,starts)
    return torch.exp(-(F.softplus(log_rates)+1e-8) @ exposure.T)
