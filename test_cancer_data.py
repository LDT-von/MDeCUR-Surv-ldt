import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from cancer_data import load_manifest
from cancer_model import CancerDeCUR, survival_at
from cancer_train import predict


class CancerDataTest(unittest.TestCase):
    def write_data(self,root):
        rng=np.random.default_rng(8)
        rows=[]
        for i in range(12):
            np.save(root/f'p{i}.npy',rng.normal(size=(3+i%2,4)).astype('float32'))
            np.save(root/f'g{i}.npy',rng.normal(size=5).astype('float32'))
            rows.append({'patient_id':f'fake-{i}','split':'train' if i<6 else 'val' if i<9 else 'test',
                         'wsi_npy':f'p{i}.npy','rna_npy':f'g{i}.npy','time':str(.2+i/10),'event':str(i%2)})
        with (root/'cohort.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        return rows

    def test_loader_and_end_to_end_synthetic_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.write_data(root)
            data=load_manifest(root/'cohort.csv')
            self.assertEqual([len(data[k]['ids']) for k in ('train','val','test')],[6,3,3])
            self.assertEqual(data['train']['mask'].sum(1).tolist(),[3,4,3,4,3,4])
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            command=[sys.executable,'train_cancer.py','--manifest',str(root/'cohort.csv'),
                     '--out',str(root/'out'),'--starts','0','1','--horizon','1',
                     '--time-unit','years','--endpoint','simulated','--epochs','3','--mode','stability']
            run=subprocess.run(command,cwd=Path(__file__).parent,env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads((root/'out'/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['patients']['test'],3)
            checkpoint=torch.load(root/'out'/'checkpoint.pt',map_location='cpu',weights_only=True)
            cfg=checkpoint['config']
            model=CancerDeCUR(checkpoint['path_dim'],checkpoint['rna_dim'],cfg['hidden'],cfg['common'],len(checkpoint['starts']))
            model.load_state_dict(checkpoint['model_state'])
            fitted={'model':model,'rna_mean':checkpoint['rna_mean'],'rna_scale':checkpoint['rna_scale'],'starts':checkpoint['starts']}
            with (root/'out'/'test_predictions.csv').open(encoding='utf-8',newline='') as f:
                rows=list(csv.DictReader(f))
            logits=predict(fitted,data['test'])['log_rates']
            expected=survival_at(logits,torch.tensor([1.]),checkpoint['starts'])[:,0]
            self.assertTrue(torch.allclose(expected,torch.tensor([float(r['survival_at_horizon']) for r in rows])))
            second=subprocess.run(command,cwd=Path(__file__).parent,env=env,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)

    def test_duplicate_patient_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=self.write_data(root)
            rows[-1]['patient_id']=rows[0]['patient_id']
            with (root/'cohort.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
            with self.assertRaises(ValueError):
                load_manifest(root/'cohort.csv')

if __name__ == '__main__':
    unittest.main()
