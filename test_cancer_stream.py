"""Synthetic numerical and gradient checks for lazy, chunked survival training."""
from copy import deepcopy
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from cancer_auxiliary import patient_balanced_local_correlation
from cancer_model import CancerDeCUR, piecewise_nll
from cancer_train import CancerConfig
from cancer_stream import stream_forward, train_stream_model, predict_stream
from model import correlation_loss


class MemoryBag:
    """Fixed fake features with a read log; no patient files or UNI weights."""
    def __init__(self, reference, a=None, b=None):
        self.features = {'features_reference': reference.detach().clone()}
        if a is not None:
            self.features['features_a'] = a.detach().clone()
        if b is not None:
            self.features['features_b'] = b.detach().clone()
        self.n_patches, self.feature_dim = reference.shape
        self.reads = []

    def blocks(self, chunk_size):
        for start in range(0, self.n_patches, chunk_size):
            yield (0, start, min(start + chunk_size, self.n_patches))

    def read_block(self, token, names):
        _, start, end = token
        self.reads.append((token, tuple(names)))
        return tuple(self.features[name][start:end].clone() for name in names)


def make_bags(seed=31, sizes=(2, 5, 7), dim=5, dtype=torch.float64, views=True):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for n in sizes:
        reference = torch.randn(n, dim, generator=generator, dtype=dtype)
        a = reference + .15 * torch.randn(n, dim, generator=generator, dtype=dtype) if views else None
        b = reference + .25 * torch.randn(n, dim, generator=generator, dtype=dtype) if views else None
        result.append(MemoryBag(reference, a, b))
    return result


def padded(bags, name):
    tensor = pad_sequence([bag.features[name] for bag in bags], batch_first=True)
    mask = torch.arange(tensor.shape[1])[None] < torch.tensor([bag.n_patches for bag in bags])[:, None]
    return tensor, mask


def joint_loss(prediction, auxiliary, local, time, event, starts):
    return (piecewise_nll(prediction, time, event, starts)
            + .003 * local
            + .002 * correlation_loss(auxiliary['path1'], auxiliary['path2'], 1)
            + .001 * correlation_loss(auxiliary['rna1'], auxiliary['rna2'], 1)
            + .002 * correlation_loss(auxiliary['path1'][:, :4], auxiliary['rna1'][:, :4], 1)
            + .001 * correlation_loss(auxiliary['path1'][:, 4:], auxiliary['rna1'][:, 4:], 0))


def cohort(seed, prefix, sizes=(2, 5, 7), views=True):
    bags = make_bags(seed, sizes=sizes, dtype=torch.float32, views=views)
    generator = torch.Generator().manual_seed(seed + 9)
    n = len(bags)
    return {'ids': [prefix + str(i) for i in range(n)], 'bags': bags,
            'rna': torch.randn(n, 6, generator=generator),
            'time': .2 + torch.rand(n, generator=generator) * 2,
            'event': (torch.arange(n) % 2).float()}


class CancerStreamTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_dense_and_chunked_three_view_outputs_gradients_and_optimizer_step(self):
        bags = make_bags()
        torch.manual_seed(12)
        original = CancerDeCUR(5, 6, hidden=8, common=4, bins=2,
                               prediction_space='encoder', local_dim=3).double()
        rna = torch.randn(3, 6, dtype=torch.float64)
        rna1, rna2 = rna + .04, rna - .03
        time = torch.tensor([.4, 1.3, 2.1], dtype=torch.float64)
        event = torch.tensor([1., 0., 1.], dtype=torch.float64)
        starts = torch.tensor([0., 1.], dtype=torch.float64)
        reference, mask = padded(bags, 'features_reference')
        a, _ = padded(bags, 'features_a')
        b, _ = padded(bags, 'features_b')
        dense = deepcopy(original)
        expected = dense(reference, rna, mask)
        first = dense.encode_auxiliary(a, rna1, mask, include_local=True)
        second = dense.encode_auxiliary(b, rna2, mask, include_local=True)
        expected_aux = {'path1': first['path'], 'path2': second['path'],
                        'rna1': first['rna'], 'rna2': second['rna']}
        local, _ = patient_balanced_local_correlation(first['local'], second['local'], mask)
        expected_loss = joint_loss(expected['log_rates'], expected_aux, local, time, event, starts)
        expected_loss.backward()
        expected_gradients = {name: value.grad.clone() if value.grad is not None else None
                              for name, value in dense.named_parameters()}
        dense_optimizer = torch.optim.AdamW(dense.parameters(), lr=.001, weight_decay=.003)
        dense_optimizer.step()
        for chunk_size in (1, 3, 50):
            for checkpoint_chunks in (False, True):
                with self.subTest(chunk_size=chunk_size, checkpoint_chunks=checkpoint_chunks):
                    model = deepcopy(original)
                    out = stream_forward(model, bags, rna, chunk_size=chunk_size,
                                         include_views=True, include_local=True,
                                         checkpoint_chunks=checkpoint_chunks, rna_views=(rna1, rna2))
                    for key in ('log_rates', 'path', 'rna', 'path_hidden', 'rna_hidden'):
                        self.assertTrue(torch.allclose(out[key], expected[key], atol=1e-10, rtol=1e-9), key)
                    for key, value in expected_aux.items():
                        self.assertTrue(torch.allclose(out['auxiliary'][key], value, atol=1e-10, rtol=1e-9), key)
                    self.assertTrue(torch.allclose(out['local_loss'], local, atol=1e-10, rtol=1e-9))
                    self.assertEqual(out['local_diagnostics']['patients'], 3)
                    self.assertEqual(out['local_diagnostics']['matched_patches'], 14)
                    loss = joint_loss(out['log_rates'], out['auxiliary'], out['local_loss'], time, event, starts)
                    self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-10, rtol=1e-9))
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        expected_gradient = expected_gradients[name]
                        self.assertEqual(parameter.grad is None, expected_gradient is None, name)
                        if expected_gradient is not None:
                            self.assertTrue(torch.allclose(parameter.grad, expected_gradient,
                                                           atol=2e-9, rtol=2e-8), name)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.003)
                    optimizer.step()
                    for name, value in dense.state_dict().items():
                        self.assertTrue(torch.allclose(model.state_dict()[name], value,
                                                       atol=2e-9, rtol=2e-8), name)

    def test_reference_only_never_reads_augmented_features(self):
        bags = make_bags(views=False)
        model = CancerDeCUR(5, 6, hidden=8, common=4, bins=2, prediction_space='encoder').double()
        rna = torch.randn(3, 6, dtype=torch.float64)
        reference, mask = padded(bags, 'features_reference')
        with torch.no_grad():
            expected = model(reference, rna, mask)['log_rates']
            actual = stream_forward(model, bags, rna, chunk_size=2, include_views=False)
        self.assertTrue(torch.allclose(expected, actual['log_rates'], atol=1e-10, rtol=1e-9))
        for bag in bags:
            self.assertTrue(bag.reads)
            self.assertTrue(all(names == ('features_reference',) for _, names in bag.reads))
            self.assertTrue(all(end - start <= 2 for (_, start, end), _ in bag.reads))

    def test_checkpoint_backward_reloads_each_needed_chunk(self):
        bags = make_bags()
        model = CancerDeCUR(5, 6, hidden=8, common=4, bins=2,
                            prediction_space='encoder', local_dim=3).double()
        out = stream_forward(model, bags, torch.randn(3, 6, dtype=torch.float64), chunk_size=2,
                             include_views=True, include_local=True, checkpoint_chunks=True)
        forward_reads = [len(bag.reads) for bag in bags]
        (out['log_rates'].square().sum() + out['local_loss']).backward()
        for bag, before in zip(bags, forward_reads):
            self.assertGreater(len(bag.reads), before)
            counts = {}
            for token, names in bag.reads:
                counts[(token, names)] = counts.get((token, names), 0) + 1
            self.assertTrue(all(count >= 2 for count in counts.values()))

    def test_stable_attention_handles_large_scores_and_offsets(self):
        bags = make_bags()
        rna = torch.randn(3, 6, dtype=torch.float64)
        reference, mask = padded(bags, 'features_reference')
        for offset, multiplier in ((10000., 1.), (0., 1000.)):
            with self.subTest(offset=offset, multiplier=multiplier):
                model = CancerDeCUR(5, 6, hidden=8, common=4, bins=2, prediction_space='encoder').double()
                with torch.no_grad():
                    model.attention[-1].bias.fill_(offset)
                    model.attention[-1].weight.mul_(multiplier)
                    expected = model(reference, rna, mask)['log_rates']
                    actual = stream_forward(model, bags, rna, chunk_size=2)['log_rates']
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.allclose(expected, actual, atol=1e-9, rtol=1e-9))

    def test_large_offset_local_moments_match_independent_numpy(self):
        bags = make_bags(dim=8)
        for bag in bags:
            bag.features['features_a'] += 1e6
            bag.features['features_b'] -= 2e6
        model = CancerDeCUR(8, 6, hidden=8, common=4, bins=2,
                            prediction_space='encoder', local_dim=3).double()
        model.patch_encoder = nn.Identity()
        with torch.no_grad():
            model.local_projector.weight.zero_()
            model.local_projector.weight[:, :3] = torch.eye(3, dtype=torch.float64)
        aa = np.concatenate([bag.features['features_a'][:, :3].numpy() for bag in bags])
        bb = np.concatenate([bag.features['features_b'][:, :3].numpy() for bag in bags])
        q = np.concatenate([np.full(bag.n_patches, 1/(len(bags)*bag.n_patches)) for bag in bags])
        ac = aa - np.average(aa, axis=0, weights=q)
        bc = bb - np.average(bb, axis=0, weights=q)
        va = np.average(ac**2, axis=0, weights=q)
        vb = np.average(bc**2, axis=0, weights=q)
        corr = (ac / np.sqrt(va + 1e-4)).T @ (q[:, None] * bc / np.sqrt(vb + 1e-4))
        expected = np.square(np.diag(corr) - 1).sum() + .005 * (
            np.square(corr).sum() - np.square(np.diag(corr)).sum())
        for chunk in (1, 3, 50):
            with self.subTest(chunk=chunk), torch.no_grad():
                out = stream_forward(model, bags, torch.zeros(3, 6, dtype=torch.float64), chunk_size=chunk,
                                     include_views=True, include_local=True)
                self.assertAlmostEqual(out['local_loss'].item(), expected, places=7)

    def test_stream_training_keeps_all_patients_and_reference_only_validation(self):
        train = cohort(41, 'train', sizes=(2, 3, 4, 5, 6))
        validation = cohort(51, 'val', sizes=(3, 5, 2), views=False)
        cfg = CancerConfig(hidden=8, common=4, epochs=2, patience=2, path_views='cached_uni',
                           patch_keep=1, prediction_space='encoder', local_dim=3,
                           lambda_local=.002, lambda_path_within=.003, lambda_rna_within=0.,
                           common_mode='uniform', rna_noise=0.)
        fitted = train_stream_model(train, validation, torch.tensor([0., 1.]), cfg,
                                    patient_batch_size=2, patch_chunk_size=2, device='cpu')
        self.assertTrue(torch.allclose(fitted['rna_mean'], train['rna'].mean(0)))
        self.assertTrue(torch.allclose(fitted['rna_scale'], train['rna'].std(0, unbiased=False).clamp_min(1e-6)))
        self.assertEqual(fitted['best_epoch'], min(fitted['history'], key=lambda h: h['validation_nll'])['epoch'])
        self.assertTrue(fitted['stream_config'])
        for epoch in fitted['history']:
            self.assertEqual(sorted(epoch['batch_patient_counts']), [2, 3])
            self.assertEqual(sum(epoch['batch_patient_counts']), 5)
            self.assertEqual(len(epoch['batches']), 2)
        self.assertTrue(all(bag.reads for bag in train['bags']))
        self.assertTrue(all(names == ('features_reference',) for bag in validation['bags'] for _, names in bag.reads))
        test = cohort(61, 'test', sizes=(4, 1), views=False)
        label_free = {key: value for key, value in test.items() if key not in ('time', 'event')}
        prediction = predict_stream(fitted, label_free)
        self.assertEqual(prediction['log_rates'].shape, (2, 2))
        self.assertTrue(torch.isfinite(prediction['log_rates']).all())
        self.assertTrue(all(names == ('features_reference',) for bag in test['bags'] for _, names in bag.reads))

    def test_patient_batch_of_one_rejected_for_correlation_training(self):
        train = cohort(42, 'train', sizes=(2, 3, 4))
        validation = cohort(52, 'val', views=False)
        cfg = CancerConfig(hidden=8, common=4, epochs=1, path_views='cached_uni',
                           patch_keep=1, prediction_space='encoder')
        with self.assertRaises(ValueError):
            train_stream_model(train, validation, torch.tensor([0., 1.]), cfg,
                               patient_batch_size=1, patch_chunk_size=2)

    def test_all_auxiliary_terms_off_allows_single_patient_training(self):
        train = cohort(43, 'train', sizes=(4,), views=False)
        validation = cohort(53, 'val', sizes=(2,), views=False)
        cfg = CancerConfig(hidden=8, common=4, epochs=1, path_views='cached_uni',
                           patch_keep=1, prediction_space='encoder', common_mode='off',
                           lambda_within=0., lambda_unique=0., lambda_local=0.)
        fitted = train_stream_model(train, validation, torch.tensor([0., 1.]), cfg,
                                    patient_batch_size=1, patch_chunk_size=2)
        self.assertEqual(fitted['history'][0]['batch_patient_counts'], [1])
        self.assertTrue(torch.isfinite(predict_stream(fitted, validation)['log_rates']).all())
        self.assertTrue(all(names == ('features_reference',) for bag in train['bags'] for _, names in bag.reads))


if __name__ == '__main__':
    unittest.main()
