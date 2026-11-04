import unittest
from dataclasses import replace
import numpy as np
import torch
from pipeline import Config, check_data, cross_fit_teacher, fit_outer_train, select_and_fit
from synthetic_study import generate


class TestPipeline(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x,self.g,self.t,self.e,self.mask = generate(4,'additive',n=15)
        self.ids = [str(i) for i in range(15)]
        self.cfg = Config(hidden=6,common=3,rank=2,epochs=3,teacher_epochs=3,folds=3,alpha=1)

    def test_oof_provenance_and_no_self_label_leak(self):
        loss, folds, audit = cross_fit_teacher(self.x,self.g,self.t,self.e,self.mask,self.ids,3,self.cfg)
        self.assertTrue(torch.isfinite(loss).all())
        for a in audit:
            self.assertFalse(set(a['train_ids']) & set(a['heldout_ids']))
            self.assertEqual(len(a['train_ids'])+len(a['heldout_ids']),15)
        # Change only held-out features: the teacher and training statistics for that fold do not change.
        g2 = self.g.clone()
        held = np.flatnonzero(folds == folds[0])
        g2[held] += 100
        _, _, audit2 = cross_fit_teacher(self.x,g2,self.t,self.e,self.mask,self.ids,3,self.cfg)
        self.assertEqual(audit[folds[0]]['omic_mean'], audit2[folds[0]]['omic_mean'])
        self.assertEqual(audit[folds[0]]['omic_scale'], audit2[folds[0]]['omic_scale'])
        # Changing another held-out patient's label cannot change this patient's
        # out-of-fold prediction, because the teacher did not see either label.
        changed = held[held != 0][0]
        e2 = self.e.clone()
        e2[changed] = 1 - e2[changed]
        loss2, _, _ = cross_fit_teacher(self.x,self.g,self.t,e2,self.mask,self.ids,3,self.cfg)
        self.assertAlmostEqual(loss[0].item(), loss2[0].item(), places=7)

    def test_reject_duplicate_ids_and_invalid_labels(self):
        with self.assertRaises(ValueError):
            check_data(self.x,self.g,self.t,self.e,self.mask,['same']*15,3)
        with self.assertRaises(ValueError):
            check_data(self.x,self.g,self.t.float()+.5,self.e,self.mask,self.ids,3)

    def test_full_and_additive_paths(self):
        for interaction in (True,False):
            m,mu,sd,teacher,folds,audit = fit_outer_train(self.x,self.g,self.t,self.e,self.mask,self.ids,3,self.cfg,interaction=interaction)
            self.assertEqual(len(teacher),15)
            out = m(self.x,(self.g-mu)/sd,self.mask)
            self.assertTrue(torch.isfinite(out['logits']).all())
            if not interaction:
                self.assertEqual(out['interaction'].abs().sum().item(),0)

    def test_selection_uses_only_inner_holdout(self):
        _,_,_,_,_,_,audit = select_and_fit(self.x,self.g,self.t,self.e,self.mask,self.ids,3,self.cfg)
        self.assertFalse(set(audit['inner_train_ids']) & set(audit['inner_validation_ids']))
        self.assertEqual(audit['selected_epoch'],min(audit['history'],key=lambda x:x['validation_nll'])['epoch'])
        self.assertEqual(len(audit['history']),self.cfg.epochs)

if __name__ == '__main__':
    unittest.main()
