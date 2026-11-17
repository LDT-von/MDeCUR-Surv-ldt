"""Composition-only control inspired by the 2025 Nature HPL study.

The input bank is supplied externally and must be fitted on training patients
only.  Its soft assignments and available k-means option are NOT HPL Leiden;
its frozen UNI features are NOT the paper's Barlow Twins images; and this
control fits a discrete-hazard head, NOT the paper's L2 Cox model.  This common
likelihood enables same-protocol synthetic NLL comparison with Method 3.1.
"""
import copy
import math

import torch
from torch import nn

from cancer_morphid_model import FrozenMorphologyBank, discrete_nll


def _validate_patients(bags, ids):
    bags = list(bags)
    if isinstance(ids, (str, bytes)):
        raise ValueError('Provide one patient ID per bag')
    ids = tuple(ids)
    if any(x is None for x in ids):
        raise ValueError('Patient IDs cannot be missing')
    ids = tuple(str(x) for x in ids)
    if not bags or len(ids) != len(bags) or len(set(ids)) != len(ids) or any(not x.strip() for x in ids):
        raise ValueError('Provide one distinct nonempty patient ID per bag')
    for identity, bag in zip(ids, bags):
        if hasattr(bag, 'patient_id') and str(bag.patient_id) != identity:
            raise ValueError('Bag patient ID does not match its declared ID')
    return bags, ids


class NatureCompositionControl(nn.Module):
    """Train-standardized fixed CLR -> L2-penalized discrete-hazard head.

    NLL uses ``alpha=0`` in ``discrete_nll``: events contribute
    ``-log(S_previous*h_bin)``; censoring (``1``) contributes
    ``-log(S_bin)``. The convention regards a censored patient as surviving
    through the end of its label bin. L2 is applied to coefficients but not
    intercepts. Mean/std are fitted on training CLR only, with std < 1e-3
    replaced by 1 for numeric stability. This extra standardization matches
    the 3.1 comparison and is not the original HPL Cox contract. The CLR is
    rank-deficient (sum=0); the penalty makes the
    coefficient fit well-defined when l2 > 0, without calling components
    independent. With l2=0 the coefficient vector is not identifiable. The
    fit-time ``l2_grid`` defines the candidates; ``self.l2`` records the
    selected candidate (the constructor value is only its initial placeholder).
    """
    def __init__(self, bank, *, bins=4, l2=0.01):
        super().__init__()
        if not isinstance(bank, FrozenMorphologyBank):
            raise TypeError('A fixed training-only morphology bank is required')
        if type(bins) is not int or bins < 1 or not math.isfinite(l2) or l2 < 0:
            raise ValueError('bins must be positive and l2 must be finite and nonnegative')
        self.bank = bank
        self.bins, self.l2 = bins, float(l2)
        # Do not disturb another arm's global random-number stream.
        with torch.random.fork_rng(devices=[]):
            self.head = nn.Linear(bank.clusters, bins).double()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer('clr_mean', torch.zeros(bank.clusters, dtype=torch.float64))
        self.register_buffer('clr_scale', torch.ones(bank.clusters, dtype=torch.float64))
        self.train_ids = ()
        self.validation_ids = ()
        self.selection = None
        self.fitted = False

    def _encode(self, bags, ids):
        bags, ids = _validate_patients(bags, ids)
        return self.bank.encode(bags)['clr'], ids

    def fit(self, bags, labels, censorship, ids, *, validation,
            l2_grid=(0., .01, .1, 1., 10.), epochs=160, lr=.03, eval_every=5):
        """Train each L2 candidate and select using validation NLL alone.

        ``validation`` is ``(bags, labels, censorship, ids)``. The morphology
        bank must be fit solely on training patients; the validation split is
        used only to select the regularization and epoch, never to fit the bank
        or take optimizer steps. Test patients must remain unseen until predict.
        """
        if self.fitted:
            raise RuntimeError('Already fitted; create a new control')
        if (type(epochs) is not int or epochs < 1 or type(eval_every) is not int
                or eval_every < 1 or eval_every > epochs or not math.isfinite(lr) or lr <= 0):
            raise ValueError('Invalid training configuration')
        grid = tuple(l2_grid)
        if not grid or any(not math.isfinite(v) or v < 0 for v in grid):
            raise ValueError('The regularization grid must be finite and nonnegative')
        x, ids = self._encode(bags, ids)
        if not isinstance(validation, (tuple, list)) or len(validation) != 4:
            raise ValueError('validation requires bags, labels, censorship and IDs')
        val_bags, val_labels, val_censorship, val_ids = validation
        vx, val_ids = self._encode(val_bags, val_ids)
        bank_ids = tuple(map(str, self.bank.metadata.get('fit_patient_ids', [])))
        if bank_ids and not set(bank_ids).issubset(ids):
            raise ValueError('Morphology bank includes a patient outside this training split')
        if set(ids) & set(val_ids):
            raise ValueError('Training and validation patient IDs overlap')
        labels = torch.as_tensor(labels).detach().cpu()
        censorship = torch.as_tensor(censorship).detach().cpu()
        val_labels = torch.as_tensor(val_labels).detach().cpu()
        val_censorship = torch.as_tensor(val_censorship).detach().cpu()
        # Validate all labels and censoring before making any model update.
        discrete_nll(self.head(x), labels, censorship, alpha=0.)
        discrete_nll(self.head(vx), val_labels, val_censorship, alpha=0.)
        # Fit once on training CLR; small/constant columns retain scale 1.
        # Validation and final test never enter these moments.
        train_mean = x.mean(0)
        train_std = x.std(0, unbiased=False)
        train_scale = torch.where(train_std < 1e-3, torch.ones_like(train_std), train_std)
        self.clr_mean.copy_(train_mean)
        self.clr_scale.copy_(train_scale)
        x, vx = (x-self.clr_mean)/self.clr_scale, (vx-self.clr_mean)/self.clr_scale
        y, c = labels.reshape(-1).long(), censorship.reshape(-1)
        # Censoring in bin t survives through its end, so those with y >= t
        # enter the risk set. Jeffreys-style .5 pseudocount avoids 0/1 hazards.
        risk = torch.stack([(y >= t).sum() for t in range(self.bins)]).double()
        events = torch.stack([((y == t) & (c == 0)).sum() for t in range(self.bins)]).double()
        hazard = (events + .5)/(risk + 1.)
        initial_bias = torch.logit(hazard)
        history = []
        best, best_state, best_meta = float('inf'), None, None
        for coefficient in grid:
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.bias.copy_(initial_bias)
            optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
            for epoch in range(1, epochs + 1):
                optimizer.zero_grad(set_to_none=True)
                nll = discrete_nll(self.head(x), labels, censorship, alpha=0.)
                penalty = 0.5*float(coefficient)*self.head.weight.square().sum()
                loss = nll + penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite penalized survival objective')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.head.parameters()):
                    raise FloatingPointError('Missing or nonfinite survival head gradient')
                optimizer.step()
                if epoch % eval_every == 0 or epoch == epochs:
                    with torch.no_grad():
                        train_nll = float(discrete_nll(self.head(x), labels, censorship, alpha=0.))
                        val_nll = float(discrete_nll(self.head(vx), val_labels, val_censorship, alpha=0.))
                    item = {'l2': float(coefficient), 'epoch': epoch, 'train_nll': train_nll,
                            'validation_nll': val_nll}
                    history.append(item)
                    if val_nll < best:
                        best, best_state, best_meta = val_nll, copy.deepcopy(self.head.state_dict()), item.copy()
        self.head.load_state_dict(best_state, strict=True)
        self.l2 = best_meta['l2']
        self.train_ids, self.validation_ids, self.fitted = ids, val_ids, True
        self.selection = {'best': best_meta, 'grid': list(map(float, grid)), 'epochs': epochs,
                          'learning_rate': float(lr), 'eval_every': eval_every,
                          'initial_bias': initial_bias.tolist(),
                          'selection_rule': 'lowest validation NLL; grid order then earlier epoch break ties'}
        return history

    @torch.no_grad()
    def predict(self, bags, ids):
        if not self.fitted:
            raise RuntimeError('Fit the control before inference')
        x, ids = self._encode(bags, ids)
        if set(ids) & (set(self.train_ids) | set(self.validation_ids)):
            raise ValueError('Training or validation and inference patient IDs overlap')
        bank_ids = set(map(str, self.bank.metadata.get('fit_patient_ids', [])))
        if set(ids) & bank_ids:
            raise ValueError('Inference patient was used to fit morphology bank')
        return self.head((x-self.clr_mean)/self.clr_scale)

    def checkpoint(self):
        if not self.fitted:
            raise RuntimeError('Only a fitted control can be saved')
        return {'version': 'nature_composition_discrete_control_v1',
                'config': {'bins': self.bins, 'l2': self.l2},
                'bank': {'centers': self.bank.centers.clone(), 'mean': self.bank.patch_mean.clone(),
                         'scale': self.bank.patch_scale.clone(), 'components': self.bank.components.clone(),
                         'temperature': self.bank.temperature, 'min_mass': self.bank.min_mass,
                         'zero_delta': self.bank.zero_delta, 'metadata': copy.deepcopy(self.bank.metadata)},
                'state': copy.deepcopy(self.state_dict()), 'train_ids': self.train_ids,
                'validation_ids': self.validation_ids, 'selection': copy.deepcopy(self.selection)}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if checkpoint.get('version') != 'nature_composition_discrete_control_v1':
            raise ValueError('Unsupported composition control checkpoint')
        obj = cls(FrozenMorphologyBank(**checkpoint['bank']), **checkpoint['config'])
        obj.load_state_dict(checkpoint['state'], strict=True)
        obj.train_ids = tuple(checkpoint['train_ids'])
        obj.validation_ids = tuple(checkpoint['validation_ids'])
        obj.selection = copy.deepcopy(checkpoint['selection'])
        obj.fitted = True
        obj.eval()
        return obj
