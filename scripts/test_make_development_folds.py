"""Synthetic contract checks; no real patient data or feature files are read."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    from .make_development_folds import make_folds
except ImportError:
    from make_development_folds import make_folds


def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class FoldContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = self.root / 'manifest.csv'
        self.original = self.root / 'original.csv'
        self.rows = [dict(patient_id=f'p{i:02}', split='train' if i < 8 else 'val',
                          event=str(i % 2), time=str(i + 1), rna_npy=f'rna/{i}.npy',
                          wsi_pt=json.dumps([f'patch/{i}.pt']), extra=f'meta-{i}')
                     for i in range(12)]
        self.original_rows = [dict(inner_train=f'p{i:02}' if i < 8 else '',
                                   inner_val=f'p{i + 8:02}' if i < 4 else '',
                                   outer_test=f't{i:02}' if i < 4 else '')
                              for i in range(8)]
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        write_csv(self.manifest, list(self.rows[0]), self.rows)
        write_csv(self.original, ['inner_train', 'inner_val', 'outer_test'], self.original_rows)

    def test_coverage_isolation_metadata_and_determinism(self):
        first = make_folds(self.manifest, self.original, self.root / 'first', 3, 17)
        second = make_folds(self.manifest, self.original, self.root / 'second', 3, 17)
        self.assertEqual(first, second)
        seen = []
        for fold in first['folds']:
            with (self.root / 'first' / fold['file']).open(newline='', encoding='utf-8') as fh:
                rows = list(csv.DictReader(fh))
            train = {row['patient_id'] for row in rows if row['split'] == 'train'}
            val = {row['patient_id'] for row in rows if row['split'] == 'val'}
            self.assertFalse(train & val)
            self.assertFalse((train | val) & {'t00', 't01', 't02', 't03'})
            self.assertEqual(len(train | val), 12)
            self.assertEqual(fold['val'], {'patients': 4, 'events': 2})
            for row in rows:
                old = next(r for r in self.rows if r['patient_id'] == row['patient_id'])
                self.assertEqual((row['event'], row['time'], row['extra']),
                                 (old['event'], old['time'], old['extra']))
                self.assertEqual(Path(row['rna_npy']), self.root / old['rna_npy'])
                self.assertEqual(json.loads(row['wsi_pt']), [str(self.root / json.loads(old['wsi_pt'])[0])])
            seen.extend(val)
        self.assertEqual(sorted(seen), sorted(row['patient_id'] for row in self.rows))
        with self.assertRaises(FileExistsError):
            make_folds(self.manifest, self.original, self.root / 'first', 3, 17)

    def test_rejects_outer_overlap_and_missing_development(self):
        self.original_rows[0]['outer_test'] = 'p00'
        self.save()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            make_folds(self.manifest, self.original, self.root / 'bad', 3, 17)
        self.original_rows[0]['outer_test'] = 't00'
        self.rows.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            make_folds(self.manifest, self.original, self.root / 'bad', 3, 17)
        self.assertFalse((self.root / 'bad').exists())


if __name__ == '__main__':
    unittest.main()
