"""Lazy input checks using synthetic H5 features only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from cancer_data import load_manifest
from cancer_stream_data import H5PatientBag, load_stream_manifest, stream_cohort_fingerprint
from test_uni_views import create_cache, write_csv
from uni_views.cache import load_patient_cache
from uni_views.core import array_hash, verify_output, view_codes


def cohort_fixture(root):
    rows = []
    for i in range(6):
        path, _ = create_cache(root, f'slide{i}', f'fake{i}')
        np.save(root / f'rna{i}.npy', np.arange(5, dtype=np.float32) + i)
        rows.append({'patient_id': f'fake{i}', 'split': 'train' if i < 3 else 'val' if i < 5 else 'test',
                     'wsi_h5': path.name, 'rna_npy': f'rna{i}.npy', 'time': .5 + .2*i, 'event': i % 2})
    manifest = root / 'patients.csv'
    write_csv(manifest, rows)
    return manifest, rows


class CancerStreamDataTest(unittest.TestCase):
    def test_ordered_multislide_blocks_equal_eager_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _ = create_cache(root, 'slide_a', 'fake')
            second, _ = create_cache(root, 'slide_b', 'fake')
            cell = json.dumps([second.name, first.name])
            bag = H5PatientBag('fake', cell, root)
            base, views, _ = load_patient_cache(cell, root, 'fake', True)
            tokens = list(bag.blocks(3))
            self.assertEqual([t[2]-t[1] for t in tokens], [3, 1, 3, 1])
            self.assertEqual([p['slide_id'] for p in bag.provenance], ['slide_b', 'slide_a'])
            self.assertEqual((bag.n_patches, bag.feature_dim), (8, 1024))
            names = ('features_reference', 'features_a', 'features_b')
            blocks = [bag.read_block(token, names) for token in tokens]
            for i, expected in enumerate((base, *views)):
                actual = torch.cat([block[i] for block in blocks])
                np.testing.assert_array_equal(actual.numpy(), expected)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device.type, 'cpu')
            changed = bag.provenance
            changed[0]['rows'] = 999
            self.assertEqual(bag.provenance[0]['rows'], 4)

    def test_blocks_and_reads_reject_invalid_tokens_and_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = create_cache(root)
            bag = H5PatientBag('fake0', path.name, root)
            for size in (0, -1, 1.5, True):
                with self.assertRaises(ValueError):
                    list(bag.blocks(size))
            for token in ((str(root/'unknown.h5'), 0, 1), (str(path), -1, 2),
                          (str(path), 0, 5), (str(path), 2, 2), [str(path), 0, 1]):
                with self.assertRaises(ValueError):
                    bag.read_block(token, ('features_reference',))
            for names in ((), ('coords',), ('features_a', 'features_a')):
                with self.assertRaises(ValueError):
                    bag.read_block(next(bag.blocks(2)), names)

    def test_changes_after_verification_are_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = create_cache(root)
            bag = H5PatientBag('fake0', path.name, root)
            token = next(bag.blocks(2))
            stamp = path.stat()
            with h5py.File(path, 'r+') as handle:
                handle['features_a'][0, 0] += 1
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
            with self.assertRaisesRegex(ValueError, 'changed'):
                bag.read_block(token, ('features_a',))
            with self.assertRaisesRegex(ValueError, 'changed'):
                bag.assert_unchanged()

    def test_missing_reference_pilot_and_wrong_identity_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            no_reference, _ = create_cache(root, 'no_reference', include_reference=False)
            pilot, _ = create_cache(root, 'pilot', limit=2)
            full, _ = create_cache(root, 'full')
            for path, identity, error in ((no_reference, 'fake0', 'reference'),
                                          (pilot, 'fake0', 'Pilot'), (full, 'other', 'different patient')):
                with self.assertRaisesRegex(ValueError, error):
                    H5PatientBag(identity, path.name, root)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                H5PatientBag('fake0', json.dumps([full.name, full.name]), root)

    def test_manifest_preserves_patient_splits_rna_labels_without_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = cohort_fixture(root)
            stream = load_stream_manifest(manifest)
            eager = load_manifest(manifest, path_views='cached_uni')
            for split in ('train', 'val', 'test'):
                self.assertEqual(stream[split]['ids'], eager[split]['ids'])
                self.assertNotIn('patches', stream[split])
                self.assertNotIn('mask', stream[split])
                for key in ('rna', 'time', 'event'):
                    self.assertTrue(torch.equal(stream[split][key], eager[split][key]))
                self.assertEqual([bag.n_patches for bag in stream[split]['bags']], [4]*len(eager[split]['ids']))

    def test_manifest_rejects_duplicate_ids_bad_labels_and_gene_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = cohort_fixture(root)
            for key, value in (('patient_id', 'fake0'), ('event', 'nan'), ('time', '-1'), ('split', 'unknown')):
                modified = [dict(row) for row in rows]
                modified[-1][key] = value
                write_csv(manifest, modified)
                with self.assertRaises(ValueError):
                    load_stream_manifest(manifest)
            write_csv(manifest, rows)
            np.save(root/'rna5.npy', np.ones(6, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                load_stream_manifest(manifest)

    def test_cross_cohort_encoder_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = cohort_fixture(root)
            with h5py.File(root/'slide5.h5', 'r+') as handle:
                meta = json.loads(handle.attrs['metadata_json'])
                meta['encoder_sha256'] = 'another-encoder'
                handle.attrs['metadata_json'] = json.dumps(meta)
            with self.assertRaisesRegex(ValueError, 'different UNI'):
                load_stream_manifest(manifest)

    def test_fingerprint_is_content_sensitive_and_path_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            source.mkdir()
            manifest, _ = cohort_fixture(source)
            original = load_stream_manifest(manifest)
            expected = stream_cohort_fingerprint(original)
            destination = root/'moved'
            shutil.copytree(source, destination)
            moved = load_stream_manifest(destination/'patients.csv')
            self.assertEqual(stream_cohort_fingerprint(moved), expected)
            moved['train']['rna'][0, 0] += 1
            self.assertNotEqual(stream_cohort_fingerprint(moved), expected)
            with h5py.File(destination/'slide0.h5', 'r+') as handle:
                data = handle['features_a'][:]
                data[0, 0] += .5
                handle['features_a'][:] = data
                handle['features_a'].attrs['sha256'] = hashlib.sha256(data.tobytes()).hexdigest()
            changed = load_stream_manifest(destination/'patients.csv')
            self.assertNotEqual(stream_cohort_fingerprint(changed), expected)

    def test_complete_verification_reads_only_bounded_row_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = create_cache(root)
            n = 2051
            with h5py.File(path, 'r+') as handle:
                meta = json.loads(handle.attrs['metadata_json'])
                coords = np.zeros((n, 2), dtype=np.int64)
                for name, data in (('coords', coords), ('row_index', np.arange(n, dtype=np.int64)),
                                   ('transform_a', np.array([view_codes(meta['seed'], meta['slide_id'], i)[0] for i in range(n)])),
                                   ('transform_b', np.array([view_codes(meta['seed'], meta['slide_id'], i)[1] for i in range(n)]))):
                    del handle[name]
                    handle.create_dataset(name, data=data)
                for name in ('features_reference', 'features_a', 'features_b'):
                    data = np.tile(handle[name][0], (n, 1))
                    del handle[name]
                    ds = handle.create_dataset(name, data=data)
                    ds.attrs['sha256'] = hashlib.sha256(data.tobytes()).hexdigest()
                meta.update(rows=n, source_rows=n, coords_sha256=array_hash(coords), output_coords_sha256=array_hash(coords))
                handle.attrs['metadata_json'] = json.dumps(meta)
            original_getitem = h5py.Dataset.__getitem__
            reads = []
            def bounded(dataset, key):
                self.assertIsInstance(key, slice)
                self.assertIsNotNone(key.stop)
                self.assertLessEqual(key.stop-(key.start or 0), 2048)
                reads.append((dataset.name, key))
                return original_getitem(dataset, key)
            with patch.object(h5py.Dataset, '__getitem__', bounded):
                verify_output(path, require_reference=True)
            self.assertGreater(len(reads), 10)
            # Corruption in the final partial chunk must be caught as well.
            for name, column, error in (('coords', 0, 'Coordinate identity'),
                                         ('row_index', None, 'reordered'),
                                         ('transform_a', None, 'Transform record')):
                with h5py.File(path, 'r+') as handle:
                    key = (n-1, column) if column is not None else n-1
                    previous = handle[name][key]
                    handle[name][key] = previous + 1
                with self.assertRaisesRegex(ValueError, error):
                    verify_output(path, require_reference=True)
                with h5py.File(path, 'r+') as handle:
                    handle[name][key] = previous


if __name__ == '__main__':
    unittest.main()
