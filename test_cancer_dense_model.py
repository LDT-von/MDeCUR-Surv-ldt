"""Known-answer task interactions and active full-objective gradients, synthetic only."""
from dataclasses import replace
import unittest
import tempfile
from pathlib import Path
import json
import os
import subprocess
import sys

import torch

from cancer_dense_model import (CenteredTokenInteraction, DensePathwaySurvival, DensePathwayConfig,
                                dense_weights, disable_all_auxiliary, require_full_auxiliary,
                                dense_pathway_objective, valid_dense_rows)
from cancer_parity import discrete_nll
from cancer_pathway_stream import LOSS_NAMES, effective_weights, minimum_batch_size, patient_batches


class DenseFullModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_full_defaults_and_rank_tail_guard(self):
        cfg = DensePathwayConfig()
        require_full_auxiliary(cfg)
        self.assertTrue(all(v > 0 for v in effective_weights(cfg).values()))
        self.assertTrue(all(v > 0 for v in dense_weights(cfg, 1).values()))
        self.assertEqual(minimum_batch_size(cfg), 17)
        self.assertEqual([len(b) for b in patient_batches(torch.arange(303), 20, 17)], [20]*14+[23])
        with self.assertRaises(ValueError):
            require_full_auxiliary(disable_all_auxiliary(cfg))
        with self.assertRaisesRegex(ValueError, 'explicit survival_loss'):
            dense_pathway_objective(None, None, cfg, 4096, 1)

    def test_training_entry_defaults_to_dense_and_reference_is_strict(self):
        from train_pathway_parity import make_parser, load_survival_reference
        parser = make_parser()
        self.assertFalse(parser.get_default('disable_auxiliary'))
        self.assertFalse(parser.get_default('legacy_architecture'))
        self.assertEqual(parser.get_default('patient_batch_size'), 20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'reference.csv'
            path.write_text('patient_id,time,event\np,1,1\np,2,1\n')
            with self.assertRaisesRegex(ValueError, 'unique valid'):
                load_survival_reference(path)
            path.write_text('patient_id,time,event\np,-4,2\n')
            with self.assertRaisesRegex(ValueError, 'unique valid'):
                load_survival_reference(path)

    def test_bilinear_represents_xor_but_additive_predictors_cannot(self):
        # Four exact states, no fit on a validation set and no claim of PID estimation.
        p = torch.tensor([-1., -1., 1., 1.]).reshape(4, 1, 1)
        r = torch.tensor([-1., 1., -1., 1.]).reshape(4, 1, 1)
        interaction = CenteredTokenInteraction(1, 1, 1).eval()
        with torch.no_grad():
            interaction.path.weight.fill_(1)
            interaction.rna.weight.fill_(1)
            interaction.head.weight.fill_(-1)
        target = torch.tensor([-1., 1., 1., -1.])
        torch.testing.assert_close(interaction(p, r).flatten(), target)
        design = torch.cat((torch.ones(4, 1), p.flatten()[:, None], r.flatten()[:, None]), 1)
        additive = design @ torch.linalg.lstsq(design, target[:, None]).solution
        self.assertGreater(float((additive.flatten()-target).square().mean()), .99)
        # Product is also nonzero on redundant modalities: it is not a synergy detector.
        self.assertTrue(torch.equal(interaction(p, p), -torch.ones(4, 1)))

    def test_validation_does_not_refit_interaction_reference(self):
        model = CenteredTokenInteraction(2, 3, 4)
        model(torch.randn(5, 8, 2), torch.randn(5, 8, 2))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.eval()
        model(torch.ones(1, 8, 2)*100, torch.ones(1, 8, 2)*200)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_survival_directly_reaches_both_token_projectors(self):
        torch.manual_seed(31)
        model = DensePathwaySurvival(6, 8, [[0, 1], [2, 3], [4, 5]], bins=4)
        b, k = 20, 3
        logits, _ = model.prediction(torch.randn(b, 16), torch.randn(b, 16),
                                      torch.randn(b, k, 16), torch.randn(b, k, 16))
        discrete_nll(logits, torch.arange(b) % 4, torch.zeros(b), alpha=.5).backward()
        for module in (model.local_path_projector, model.local_rna_projector,
                       model.decomposition_head, model.token_cooperation):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_parity_cli_full_default_saves_explicit_best_and_last(self):
        from test_comparison_fold0 import Fold0ComparisonTest
        import comparison_fold0 as compare
        case = Fold0ComparisonTest()
        case.setUp()
        try:
            genes, pathways = case.root/'genes.txt', case.root/'pathways.gmt'
            genes.write_text('\n'.join(f'G{i}' for i in range(8))+'\n')
            pathways.write_text(''.join(f'P{i}\tid{i}\t'+ '\t'.join(f'G{j}' for j in indices)+'\n'
                                        for i, indices in enumerate(case.original['model_kwargs']['pathway_indices'])))
            out = case.root/'parity_full'
            script = Path(compare.__file__).with_name('train_pathway_parity.py')
            process = subprocess.run([sys.executable, str(script), '--manifest', str(case.manifest),
                '--survival-reference', str(case.reference), '--genes', str(genes), '--pathways', str(pathways),
                '--out', str(out), '--num-patches', '4096', '--epochs', '1', '--endpoint', 'DSS',
                '--time-unit', 'months', '--split-name', 'generated_fold0'], capture_output=True, text=True,
                env=dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((out/'report.json').read_text())
            self.assertEqual(report['architecture'], 'DensePathwaySurvival')
            self.assertEqual(len(report['effective_loss_weights']), 15)
            self.assertTrue(all(v > 0 for v in report['effective_loss_weights'].values()))
            self.assertEqual(report['last_epoch'], 1)
            self.assertEqual(report['selected_checkpoint_validation'], report['last_validation'])
            for selection in ('best', 'last'):
                payload = torch.load(out/f'checkpoint_{selection}.pt', weights_only=True)
                self.assertEqual(payload['epoch'], 1)
                self.assertEqual(payload['selection'], selection)
            self.assertEqual((out/'val_predictions_best.csv').read_bytes(),
                             (out/'val_predictions_last.csv').read_bytes())
        finally:
            case.doCleanups()

    def test_all_fifteen_terms_have_finite_gradients_and_no_padding_tokens(self):
        # Reuse only the generated fixture; no external manifest is loaded.
        from test_comparison_fold0 import Fold0ComparisonTest
        import comparison_fold0 as compare
        case = Fold0ComparisonTest()
        case.setUp()
        try:
            cohort, reference, edges, original, _ = compare.load_inputs(case.manifest, case.reference, case.ref_run)
            try:
                cfg = replace(compare.full_config(original['model_kwargs']), dense_max_tokens=12)
                model = DensePathwaySurvival(**original['model_kwargs'])
                bags = [compare.BudgetedPatientBag(b, 4096, 'train', 3) for b in cohort['train']['bags']]
                class ShortPaddedBag:
                    n_patches, padding = 4096, 4084
                    def blocks(self, chunk):
                        yield ('synthetic', 0, 4096)
                    def read_block(self, block, names):
                        return (torch.cat((torch.ones(12, 6), torch.zeros(4084, 6))),)
                rows, info = valid_dense_rows(ShortPaddedBag(), 12)
                self.assertEqual(len(rows), 12)
                self.assertEqual(info['excluded_padding'], 4096-12)
                self.assertTrue(torch.equal(rows, torch.ones(12, 6)))
                batch = {'bags': bags, 'rna': cohort['train']['rna'], 'time': cohort['train']['time'],
                         'event': cohort['train']['event'], 'rna_mask': None,
                         'label': torch.arange(20) % 4, 'censorship': 1-cohort['train']['event']}
                def likelihood(logits, current):
                    return discrete_nll(logits, current['label'], current['censorship'], alpha=.5)
                total, stats, components = dense_pathway_objective(model, batch, cfg, 4096, 71, 1,
                                                                  survival_loss=likelihood, return_components=True)
                self.assertEqual(set(components), {'survival', *LOSS_NAMES, *compare.DENSE_LOSS_NAMES})
                parameters = tuple(p for p in model.parameters() if p.requires_grad)
                for name, term in components.items():
                    grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                    nonempty = [g for g in grads if g is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in nonempty), name)
                    self.assertGreater(sum(float(g.detach().abs().sum()) for g in nonempty), 0, name)
                total.backward()
                self.assertFalse(stats['common_skipped'])
                self.assertEqual(len(stats['dense_diagnostics']), 20)
            finally:
                compare.close(cohort)
        finally:
            case.doCleanups()


if __name__ == '__main__':
    unittest.main()
