"""Generated-data parity: the same CLR-only linear model in two implementations.

This is an implementation check, not evidence that either model generalizes.
The fixed synthetic bank and all patient bags below are generated in memory.
"""
import torch

from method3_1_nature_control import NatureCompositionControl
from method3_1_regularized import RegularizedMethod31
from test_method3_1 import generated, make_model


def test_clr_only_mask_matches_nature_control_end_to_end():
    torch.set_num_threads(1)
    train_bags, train_rna, train_ids = generated(32, 1401, 'parity-train')
    val_bags, val_rna, val_ids = generated(16, 1402, 'parity-val')
    test_bags, test_rna, test_ids = generated(12, 1403, 'parity-test')
    # Force variable two-cluster compositions, keeping at least two patches
    # per cluster so the check exercises CLR coefficients rather than only
    # matching two intercept-only models on constant composition.
    def varied(bags):
        return [torch.cat((bag[:2+i % 3], bag[6:6+4-i % 3]))
                for i, bag in enumerate(bags)]
    train_bags, val_bags, test_bags = map(varied, (train_bags, val_bags, test_bags))
    # Deliberately mix bins and censoring; all outcomes are synthetic.
    y = torch.arange(len(train_ids)) % 3
    c = (torch.arange(len(train_ids)) % 3 == 0).double()
    vy = (torch.arange(len(val_ids))+1) % 3
    vc = (torch.arange(len(val_ids)) % 4 == 0).double()

    backbone = make_model()
    backbone.pretrain(train_bags, train_rna, train_ids, epochs=3)
    backbone.fit_probes(train_bags, train_rna, train_ids, folds=4)
    assert float(backbone.bank.encode(train_bags)['clr'].std(0).min()) > .1

    # Features are laid out [RNA, c, u, CLR, cluster mask, SSL, products].
    clr_start = backbone.rna_dim + 2*backbone.k*backbone.d
    masked = RegularizedMethod31(backbone, arm='full')
    masked.mask.zero_()
    masked.mask[clr_start:clr_start+backbone.k] = 1
    selected = masked.fit(y, c, train_ids, val_bags, val_rna, val_ids, vy, vc,
                          penalties=(.01, .1, 1.), epochs=50, lr=.03, eval_every=5)

    nature = NatureCompositionControl(backbone.bank, bins=backbone.bins)
    nature.fit(train_bags, y, c, train_ids, validation=(val_bags, vy, vc, val_ids),
               l2_grid=(.01, .1, 1.), epochs=50, lr=.03, eval_every=5)
    selected_nature = nature.selection['best']
    assert selected['penalty'] == selected_nature['l2']
    assert selected['epoch'] == selected_nature['epoch']
    assert abs(selected['validation_nll']-selected_nature['validation_nll']) < 1e-12
    assert torch.allclose(masked.mean[clr_start:clr_start+backbone.k], nature.clr_mean, atol=1e-12, rtol=0)
    assert torch.allclose(masked.scale[clr_start:clr_start+backbone.k], nature.clr_scale, atol=1e-12, rtol=0)
    assert torch.allclose(masked.predict(test_bags, test_rna, test_ids),
                          nature.predict(test_bags, test_ids), atol=1e-12, rtol=0)
