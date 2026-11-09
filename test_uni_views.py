"""Synthetic pixels, fake WSI reader and fake encoder only; no patient/UNI jobs."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import torch

from cancer_data import load_manifest
from cancer_train import CancerConfig, view_batch
from uni_views import cli
from uni_views import core
from uni_views.core import (array_hash, extract_slide, inspect_slides, load_recipe,
                            load_slides, pixel_tensor_array, read_coordinates,
                            read_patch, transform_image, verify_output, view_codes)


RECIPE = load_recipe(Path(__file__).parent / 'uni_views' / 'recipe_uni_v1.json')


class FakeSlide:
    dimensions = (8, 8)
    properties = {'openslide.mpp-x': '0.25', 'openslide.objective-power': '40'}

    def __init__(self, path):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read_region(self, xy, level, size):
        self.calls.append((xy, level, size))
        w,h = size
        pixels = np.zeros((h, w, 4), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                xx,yy = xy[0]+x,xy[1]+y
                if xx < 8 and yy < 8:
                    pixels[y,x] = [xx*20, yy*20, (xx+yy)*10, 255]
        return Image.fromarray(pixels)


class FakeEncoder:
    def __call__(self, array):
        return np.ascontiguousarray(array.reshape(len(array), -1)[:, :1024])


def fixture(root, slide='slide0', patient='fake0', coords=None):
    source = root / 'sources'
    source.mkdir(exist_ok=True)
    (source / f'{slide}.svs').write_bytes(b'FAKE WSI: test fixture only')
    if coords is None:
        coords = [[0,0], [4,0], [0,4], [4,4]]
    with h5py.File(source / f'{slide}.h5', 'w') as f:
        ds = f.create_dataset('coords', data=np.array(coords, dtype=np.int64))
        ds.attrs['patch_size'] = 4
        ds.attrs['patch_level'] = 0
        ds.attrs['level_dim'] = [8,8]
    return {'patient_id': patient, 'slide_id': slide,
            'wsi_path': str(source / f'{slide}.svs'),
            'coords_h5': str(source / f'{slide}.h5')}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def create_cache(root, slide='slide0', patient='fake0', include_reference=True, limit=None):
    row = fixture(root, slide, patient)
    items, errors = inspect_slides([row], limit)
    if errors:
        raise AssertionError(errors)
    output = root / f'{slide}.h5'
    extract_slide(items[0], RECIPE, FakeEncoder(), FakeSlide, output,
                  'synthetic-run', 'synthetic-encoder', include_reference=include_reference,
                  batch_size=2)
    return output, items[0]


class UNIViewsTest(unittest.TestCase):
    def test_d4_keeps_every_pixel_and_has_eight_distinct_orientations(self):
        image = Image.fromarray(np.arange(4*4*3, dtype=np.uint8).reshape(4,4,3))
        outputs = [np.array(transform_image(image,c)) for c in range(8)]
        self.assertEqual(len({x.tobytes() for x in outputs}),8)
        expected = sorted(map(tuple,np.array(image).reshape(-1,3)))
        for result in outputs:
            self.assertEqual(result.shape,(4,4,3))
            self.assertEqual(sorted(map(tuple,result.reshape(-1,3))),expected)

    def test_rows_views_and_reference_are_exact_and_batch_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fixture(root)
            item = inspect_slides([row])[0][0]
            source_before = Path(row['coords_h5']).read_bytes()
            slide = FakeSlide('fake')
            for batch in (1,3):
                output = root / f'batch{batch}.h5'
                extract_slide(item,RECIPE,FakeEncoder(),lambda _:slide,output,'run','encoder',
                              batch_size=batch,include_reference=True)
                meta = verify_output(output,require_reference=True)
                self.assertEqual(meta['rows'],4)
                with h5py.File(output,'r') as f:
                    coords = f['coords'][:]
                    np.testing.assert_array_equal(f['row_index'][:],np.arange(4))
                    for index,xy in enumerate(coords):
                        image = read_patch(FakeSlide('fake'),xy,4)
                        for name,code in zip(('features_a','features_b'),view_codes(17,'slide0',index)):
                            expected = FakeEncoder()(pixel_tensor_array(transform_image(image,code),RECIPE)[None])[0]
                            np.testing.assert_array_equal(f[name][index],expected)
                        expected = FakeEncoder()(pixel_tensor_array(image,RECIPE)[None])[0]
                        np.testing.assert_array_equal(f['features_reference'][index],expected)
            self.assertEqual(len(slide.calls),8)  # once per original patch, not twice/three times
            with h5py.File(root/'batch1.h5') as a,h5py.File(root/'batch3.h5') as b:
                for name in ('features_a','features_b','features_reference'):
                    np.testing.assert_array_equal(a[name][:],b[name][:])
            self.assertEqual(source_before,Path(row['coords_h5']).read_bytes())
            self.assertFalse(list(root.rglob('*.png')))

    def test_boundary_policy_keeps_rows_and_composites_white(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fixture(root,coords=[[0,0],[6,6]])
            item = inspect_slides([row])[0][0]
            output = root/'views.h5'
            with self.assertRaisesRegex(ValueError,'boundary'):
                extract_slide(item,RECIPE,FakeEncoder(),FakeSlide,output,'run','encoder')
            self.assertFalse(output.exists())
            meta = extract_slide(item,RECIPE,FakeEncoder(),FakeSlide,output,'run','encoder',edge_policy='pad-white')
            self.assertEqual(meta['edge_rows'],1)
            self.assertEqual(meta['rows'],2)
            patch_image = np.array(read_patch(FakeSlide('fake'),[6,6],4))
            np.testing.assert_array_equal(patch_image[2:,2:],np.full((2,2,3),255))

    def test_source_changes_and_unsupported_geometry_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fixture(root)
            item = inspect_slides([row])[0][0]
            with h5py.File(row['coords_h5'],'r+') as f:
                f['coords'].attrs['patch_size'] = 5
            with self.assertRaisesRegex(ValueError,'changed'):
                extract_slide(item,RECIPE,FakeEncoder(),FakeSlide,root/'out.h5','run','encoder')
            with h5py.File(row['coords_h5'],'r+') as f:
                f['coords'].attrs['patch_level'] = 1
            with self.assertRaisesRegex(ValueError,'patch_level=0'):
                read_coordinates(row['coords_h5'])

    def test_corrupted_features_and_reordered_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output,_ = create_cache(root)
            with h5py.File(output,'r+') as f:
                f['features_a'][0,0] += 1
            with self.assertRaisesRegex(ValueError,'checksum'):
                verify_output(output)
            other,_ = create_cache(root,'slide1')
            with h5py.File(other,'r+') as f:
                f['row_index'][:] = [1,0,2,3]
            with self.assertRaisesRegex(ValueError,'reordered'):
                verify_output(other)

    def test_failed_encoder_never_produces_a_completed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fixture(root)
            item = inspect_slides([row])[0][0]
            def bad_encoder(array):
                return np.full((len(array),1024),np.nan,dtype=np.float32)
            output = root/'out.h5'
            with self.assertRaisesRegex(ValueError,'finite'):
                extract_slide(item,RECIPE,bad_encoder,FakeSlide,output,'run','encoder')
            self.assertFalse(output.exists())
            partial = list(root.glob('*.partial.*'))
            self.assertEqual(len(partial),1)
            with h5py.File(partial[0]) as f:
                self.assertEqual(f.attrs['complete'],0)

    def test_atomic_publication_never_overwrites_a_concurrent_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fixture(root)
            item = inspect_slides([row])[0][0]
            destination = root/'out.h5'
            real_link = os.link
            def concurrent_writer(source,target):
                Path(target).write_bytes(b'created by another process')
                return real_link(source,target)
            with patch.object(core.os,'link',side_effect=concurrent_writer):
                with self.assertRaises(FileExistsError):
                    extract_slide(item,RECIPE,FakeEncoder(),FakeSlide,destination,'run','encoder')
            self.assertEqual(destination.read_bytes(),b'created by another process')

    def test_plan_does_not_open_wsi_or_load_model_and_reports_storage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fixture(root)
            manifest = root/'slides.csv'
            write_csv(manifest,[row])
            stdout = io.StringIO()
            with (patch.object(cli,'open_slide',side_effect=AssertionError('pixel reader called')),
                 patch.object(cli,'FrozenUNI',side_effect=AssertionError('model loaded')),
                 contextlib.redirect_stdout(stdout)):
                result = cli.main(['plan','--manifest',str(manifest),'--recipe',
                                   str(Path(__file__).parent/'uni_views'/'recipe_uni_v1.json')])
            self.assertEqual(result,0)
            self.assertEqual(json.loads(stdout.getvalue())['feature_payload_bytes'],4*1024*4*2)
            row['wsi_path'] = str(root/'missing'/'slide0.svs')
            write_csv(manifest,[row])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['plan','--manifest',str(manifest),'--recipe',
                                 str(Path(__file__).parent/'uni_views'/'recipe_uni_v1.json')]),1)

    def test_resume_verifies_outputs_without_rerunning_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fixture(root)
            manifest = root/'slides.csv'
            write_csv(manifest,[row])
            weights = root/'fake.bin'
            weights.write_bytes(b'not actual UNI weights')
            args = ['extract','--manifest',str(manifest),'--recipe',
                    str(Path(__file__).parent/'uni_views'/'recipe_uni_v1.json'),
                    '--checkpoint',str(weights),'--out',str(root/'output'),'--include-reference']
            with (patch.object(cli,'open_slide',side_effect=FakeSlide),
                 patch.object(cli,'FrozenUNI',return_value=FakeEncoder()) as model,
                 patch.object(cli,'runtime_check',return_value={'test':'synthetic'}),
                 contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(cli.main(args),0)
                self.assertEqual(cli.main(args+['--resume']),0)
                self.assertEqual(model.call_count,1)
                with self.assertRaisesRegex(ValueError,'Resume contract'):
                    cli.main(args+['--resume','--seed','18'])
                with self.assertRaises(FileExistsError):
                    cli.main(args)
                with h5py.File(root/'output'/'slide0.h5','r+') as f:
                    f['features_b'][0,0] += 1
                with self.assertRaisesRegex(ValueError,'checksum'):
                    cli.main(args+['--resume'])
            self.assertFalse((root/'output'/'.extract.lock').exists())

    def test_doctor_checks_headers_without_pixels_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fixture(root)
            manifest = root/'slides.csv'
            write_csv(manifest,[row])
            weights = root/'fake.bin'
            weights.write_bytes(b'fake weights')
            fake_slide = FakeSlide('fake')
            with (patch.object(cli,'open_slide',return_value=fake_slide),
                  patch.object(cli,'FrozenUNI',side_effect=AssertionError('UNI should not load')),
                  patch.object(cli,'runtime_check',return_value={'test':'synthetic'}),
                  contextlib.redirect_stdout(io.StringIO())):
                code = cli.main(['doctor','--manifest',str(manifest),'--recipe',
                                 str(Path(__file__).parent/'uni_views'/'recipe_uni_v1.json'),
                                 '--checkpoint',str(weights)])
            self.assertEqual(code,0)
            self.assertEqual(fake_slide.calls,[])

    def test_actual_geometry_must_match_coordinate_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fixture(root)
            item = inspect_slides([row])[0][0]
            fake_slide = FakeSlide('fake')
            fake_slide.dimensions = (9,8)
            with self.assertRaisesRegex(ValueError,'dimensions'):
                extract_slide(item,RECIPE,FakeEncoder(),lambda _:fake_slide,
                              root/'out.h5','run','encoder')
            self.assertFalse((root/'out.h5').exists())

    def test_offline_encoder_wrapper_loads_strictly_freezes_and_uses_eval(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3,1024,bias=False)
            def forward(self,x):
                return self.linear(x.mean((2,3)))
        with tempfile.TemporaryDirectory() as temp:
            source = Tiny()
            weights = Path(temp)/'synthetic.bin'
            torch.save(source.state_dict(),weights)
            target = Tiny()
            def create_model(name, **kwargs):
                self.assertEqual(name,'vit_large_patch16_224')
                self.assertFalse(kwargs['pretrained'])
                self.assertEqual(kwargs['init_values'],1e-5)
                return target
            with patch.dict(sys.modules,{'timm':types.SimpleNamespace(create_model=create_model)}):
                encoder = cli.FrozenUNI(weights,'cpu','none')
                x = np.ones((2,3,4,4),dtype=np.float32)
                np.testing.assert_allclose(encoder(x),source(torch.from_numpy(x)).detach().numpy())
                self.assertFalse(target.training)
                self.assertTrue(all(not p.requires_grad for p in target.parameters()))

    def test_patient_slide_order_is_explicit_and_cross_patient_reuse_fails(self):
        from uni_views.cache import load_patient_cache
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a,_ = create_cache(root,'slide0','fake0')
            b,_ = create_cache(root,'slide1','fake0')
            base,views,provenance = load_patient_cache(json.dumps([b.name,a.name]),root,'fake0',True)
            self.assertEqual([p['slide_id'] for p in provenance],['slide1','slide0'])
            self.assertEqual(base.shape,(8,1024))
            with h5py.File(b) as f:
                np.testing.assert_array_equal(views[0][:4],f['features_a'][:])
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                load_patient_cache(json.dumps([a.name,a.name]),root,'fake0',True)

    def test_manifest_refuses_reusing_wsi_under_another_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = fixture(root)
            second = dict(first,patient_id='another',slide_id='another')
            manifest = root/'slides.csv'
            write_csv(manifest,[first,second])
            with self.assertRaisesRegex(ValueError,'listed twice'):
                load_slides(manifest)

    def test_cached_views_keep_all_mask_rows_and_do_not_mutate_inputs(self):
        x = torch.randn(3,4,1024)
        a,b = x+1,x-1
        g = torch.randn(3,5)
        mask = torch.tensor([[True,True,False,False],[True]*4,[True,True,True,False]])
        cfg = CancerConfig(path_views='cached_uni',patch_keep=1,rna_noise=0)
        result = view_batch(x,g,mask,cfg,(a,b))
        self.assertIs(result[0][0],a)
        self.assertIs(result[1][0],b)
        for _,rna,m in result:
            self.assertTrue(torch.equal(m,mask))
            self.assertTrue(torch.equal(rna,g))
        with self.assertRaises(ValueError):
            view_batch(x,g,mask,CancerConfig(path_views='cached_uni'),(a,b))

    def test_missing_reference_pilot_and_wrong_patient_are_rejected(self):
        from uni_views.cache import load_patient_cache
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output,_ = create_cache(root,include_reference=False)
            with self.assertRaisesRegex(ValueError,'reference'):
                load_patient_cache(str(output),root,'fake0',True)
            pilot,_ = create_cache(root,'slide1',limit=2)
            with self.assertRaisesRegex(ValueError,'Pilot'):
                load_patient_cache(str(pilot),root,'fake0',True)
            full,_ = create_cache(root,'slide2')
            with self.assertRaisesRegex(ValueError,'different patient'):
                load_patient_cache(str(full),root,'other',True)

    def test_cached_manifest_cli_and_label_free_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for i in range(6):
                cache,_ = create_cache(root,f'slide{i}',f'fake{i}')
                np.save(root/f'rna{i}.npy',np.arange(5,dtype=np.float32)+i)
                rows.append({'patient_id':f'fake{i}','split':'train' if i<3 else 'val' if i<5 else 'test',
                             'wsi_h5':cache.name,'rna_npy':f'rna{i}.npy','time':.5+i*.2,'event':i%2})
            manifest = root/'patients.csv'
            write_csv(manifest,rows)
            data = load_manifest(manifest,path_views='cached_uni')
            self.assertEqual(data['train']['patches_view1'].shape,(3,4,1024))
            self.assertNotIn('patches_view1',data['test'])
            env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            command = [sys.executable,'train_cancer.py','--manifest',str(manifest),'--out',str(root/'training'),
                       '--starts','0','1','--horizon','1','--time-unit','years','--endpoint','synthetic',
                       '--mode','stability','--path-views','cached_uni','--rna-noise','0','--epochs','2']
            result = subprocess.run(command,cwd=Path(__file__).parent,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report = json.loads((root/'training'/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['config']['patch_keep'],1)
            self.assertEqual(report['config']['path_views'],'cached_uni')
            self.assertEqual(len(report['view_provenance']['train']),3)
            self.assertTrue((root/'training'/'test_predictions.csv').is_file())


if __name__ == '__main__':
    unittest.main()
