import pytest
import torch

from method3_1_cluster_adapter import bank_from_training_cluster_labels


def test_external_clusters_are_training_only_and_mapped_without_reclustering():
    bags = [torch.tensor([[-3., 0.], [-2., .1], [2., 0.], [3., .1]]) for _ in range(3)]
    labels = [torch.tensor([0, 0, 1, 1]) for _ in bags]
    bank = bank_from_training_cluster_labels(bags, labels, ['a', 'b', 'c'],
                                              latent_dim=2, label_provenance='synthetic-known-labels')
    assert bank.metadata['fit_patient_ids'] == ['a', 'b', 'c']
    assert bank.metadata['hpl_replication'] is False
    enc = bank.encode(bags)
    assert torch.allclose(enc['composition'], torch.full((3, 2), .5, dtype=torch.float64), atol=1e-6)


def test_external_cluster_label_validation():
    bags = [torch.tensor([[0., 1.], [1., 2.]])]
    with pytest.raises(ValueError, match='contiguous'):
        bank_from_training_cluster_labels(bags, [torch.tensor([0, 2])], ['a'], label_provenance='synthetic')
    with pytest.raises(ValueError, match='source'):
        bank_from_training_cluster_labels(bags, [torch.tensor([0, 1])], ['a'], label_provenance='')
