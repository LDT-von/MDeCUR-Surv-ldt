import math
import unittest
import torch
from cancer_model import CancerDeCUR, piecewise_exposure, piecewise_nll, survival_at
from cancer_train import CancerConfig, train_cancer_model, predict


def cohort(seed,n,prefix):
    torch.manual_seed(seed)
    return {'patches':torch.randn(n,4,5), 'rna':torch.randn(n,6),
            'mask':torch.ones(n,4,dtype=torch.bool), 'time':torch.rand(n)*3,
            'event':torch.randint(0,2,(n,)).float(), 'ids':[prefix+str(i) for i in range(n)]}


class CancerModelTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_continuous_likelihood_known_answers_and_boundaries(self):
        starts = torch.tensor([0.,1.,3.])
        t = torch.tensor([0.5,1.,2.,4.])
        self.assertTrue(torch.allclose(piecewise_exposure(t,starts),torch.tensor([[.5,0,0],[1,0,0],[1,1,0],[1,2,1]])))
        rates = torch.tensor([.2,.4,.8])
        logits = torch.log(torch.expm1(rates)).repeat(4,1)
        event = torch.tensor([0.,1.,1.,0.])
        expected = torch.tensor([.1,.2-math.log(.4),.6-math.log(.4),1.8])
        self.assertTrue(torch.allclose(piecewise_nll(logits,t,event,starts,'none'),expected,atol=1e-6))
        survival = survival_at(logits,torch.tensor([0.,.5,2.,4.]),starts)
        self.assertTrue(torch.allclose(survival[:,0],torch.ones(4)))
        self.assertTrue((survival[:,1:] <= survival[:,:-1]).all())

    def test_masked_attention_and_gradient(self):
        data = cohort(1,8,'a')
        data['mask'][:,2:] = False
        m=CancerDeCUR(5,6,hidden=8,common=4,bins=2)
        original=m(data['patches'],data['rna'],data['mask'])
        data['patches'][:,2:] = float('nan')
        out=m(data['patches'],data['rna'],data['mask'])
        self.assertTrue(torch.allclose(original['log_rates'],out['log_rates']))
        self.assertTrue(torch.all(out['attention'][:,2:] == 0))
        self.assertTrue(torch.allclose(out['attention'].sum(1),torch.ones(8)))
        piecewise_nll(out['log_rates'],data['time'],data['event'],torch.tensor([0.,1.])).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_train_controls_and_label_free_prediction(self):
        train,val=cohort(2,20,'tr'),cohort(3,10,'va')
        starts=torch.tensor([0.,1.])
        for mode in ('off','uniform','stability','shuffled'):
            cfg=CancerConfig(hidden=8,common=4,epochs=4,patience=4,common_mode=mode)
            fit=train_cancer_model(train,val,starts,cfg)
            self.assertEqual(fit['best_epoch'],min(fit['history'],key=lambda x:x['validation_nll'])['epoch'])
            self.assertTrue(torch.allclose(fit['rna_mean'],train['rna'].mean(0)))
            unlabelled={k:v for k,v in val.items() if k not in ('time','event')}
            self.assertTrue(torch.isfinite(predict(fit,unlabelled)['log_rates']).all())
            if mode != 'off':
                self.assertTrue(any(h['ess'] is not None for h in fit['history']))
        val['ids'][0]=train['ids'][0]
        with self.assertRaises(ValueError):
            train_cancer_model(train,val,starts,CancerConfig(epochs=1))

if __name__ == '__main__':
    unittest.main()
