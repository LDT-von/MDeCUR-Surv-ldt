import unittest
import torch
from model import ConditionalDeCURSurv, decur_auxiliary, survival_nll, utility_weighted_loss, stability_weighted_common, correlation_loss


class TestPrototype(unittest.TestCase):
    def test_nll_known_answer(self):
        z = torch.zeros(2, 3)
        # event in bin 0 and censored through bin 1 => -log(.5), -log(.25)
        got = survival_nll(z, torch.tensor([0, 1]), torch.tensor([1, 0]))
        self.assertAlmostEqual(got.item(), 1.5 * torch.log(torch.tensor(2.)).item(), places=6)

    def test_mask_and_gradient(self):
        torch.manual_seed(2)
        m = ConditionalDeCURSurv(5, 7, hidden=8, common=4, intervals=3, rank=3)
        p, g = torch.randn(4, 6, 5), torch.randn(4, 7)
        mask = torch.tensor([[1,1,1,0,0,0]] * 4, dtype=torch.bool)
        out = m(p, g, mask)
        p[:, 3:] = float('nan')
        self.assertTrue(torch.allclose(out['logits'], m(p, g, mask)['logits']))
        aux = decur_auxiliary(out['path'], out['path'] + .01, out['omic'], out['omic'] + .01, 4)
        loss = survival_nll(out['logits'], torch.tensor([0,1,2,1]), torch.tensor([1,0,1,0]))
        (loss + .001 * sum(aux.values())).backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None))

    def test_uniform_and_interaction(self):
        m = ConditionalDeCURSurv(3, 4, hidden=6, common=3, intervals=2, rank=2)
        o = m(torch.randn(3, 2, 3), torch.randn(3, 4))
        t, e = torch.tensor([0,1,0]), torch.tensor([1,0,0])
        loss = utility_weighted_loss(o['logits'], t, e, torch.ones(3), weight=0)
        self.assertTrue(torch.allclose(loss, survival_nll(o['logits'], t, e)))
        p, g = o['path'], o['omic']
        def f(x,y):
            return m.bias + m.path_head(x) + m.omic_head(y) + m.interaction_head(m.path_factor(x)*m.omic_factor(y))
        self.assertTrue(torch.allclose(f(p,g)-f(p,torch.zeros_like(g))-f(torch.zeros_like(p),g)+f(torch.zeros_like(p),torch.zeros_like(g)), o['interaction'], atol=1e-6))

    def test_stability_weighted_correlation(self):
        torch.manual_seed(31)
        p = torch.randn(12,3,requires_grad=True)
        g = torch.randn(12,3,requires_grad=True)
        val, stats = stability_weighted_common(p,p+.1,g,g+.1,strength=0)
        self.assertTrue(torch.allclose(val,correlation_loss(p,g,1),atol=1e-6))
        self.assertFalse(stats['weights'].requires_grad)
        self.assertAlmostEqual(stats['ess'].item(),12)
        val.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        with self.assertRaises(ValueError):
            stability_weighted_common(torch.ones(4,3),torch.ones(4,3),torch.ones(4,3),torch.ones(4,3))

    def test_stability_weighted_matches_manual_calculation(self):
        p=torch.tensor([[1.,2.],[2.,1.],[-1.,.5],[.5,-2.]],requires_grad=True)
        g=torch.tensor([[2.,1.],[-1.,2.],[1.,-.5],[-.5,-1.]],requires_grad=True)
        p2=p.detach().clone()
        p2[0]*=-1
        g2=g.detach().clone()
        loss,diag=stability_weighted_common(p,p2,g,g2,strength=2.,off_weight=.005)
        weights=diag['weights']
        self.assertLess(weights[0].item(),weights[1].item())
        w=weights[:,None]
        def norm(x):
            centered=x-(w*x).sum(0)/w.sum()
            return centered/torch.sqrt((w*centered.square()).sum(0)/w.sum()+1e-4)
        corr=(norm(p)*w).T@norm(g)/w.sum()
        expected=(corr.diag()-1).square().sum()+.005*(corr-torch.diag(corr.diag())).square().sum()
        self.assertTrue(torch.allclose(loss,expected,atol=1e-6))
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all() and torch.isfinite(g.grad).all())

if __name__ == '__main__':
    unittest.main()
