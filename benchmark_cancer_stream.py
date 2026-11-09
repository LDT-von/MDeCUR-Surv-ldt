"""CPU engineering benchmark on generated features, not patient efficacy.

Each backend runs in a fresh process. Peak memory includes Python/PyTorch and
allocator overhead. The generated bags do not measure H5 or real UNI throughput.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from cancer_auxiliary import patient_balanced_local_correlation
from cancer_model import CancerDeCUR, piecewise_nll
from cancer_stream import stream_forward
from model import correlation_loss


class GeneratedBag:
    def __init__(self, patient, rows, features):
        self.patient, self.n_patches, self.feature_dim = patient, rows, features
        self.max_rows_read = 0
        self.read_calls = 0

    def blocks(self, chunk_size):
        for start in range(0, self.n_patches, chunk_size):
            yield (self.patient, start, min(start+chunk_size, self.n_patches))

    def read_block(self, token, names):
        _, start, stop = token
        self.max_rows_read = max(self.max_rows_read, stop-start)
        self.read_calls += 1
        row = torch.arange(start, stop, dtype=torch.float32)[:, None]
        col = torch.arange(self.feature_dim, dtype=torch.float32)[None, :]
        base = torch.sin(row*.013 + col*.021 + self.patient*.37)
        return tuple(base if name == 'features_reference' else
                     base + (.03 if name == 'features_a' else -.02)*torch.cos(row*.007+col*.017)
                     for name in names)


def peak_memory_bytes():
    if sys.platform == 'win32':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not memory_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak*1024


def worker(args):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = CancerDeCUR(args.features, 6, hidden=16, common=8, bins=2,
                        prediction_space='encoder', local_dim=8)
    bags = [GeneratedBag(i, args.rows, args.features) for i in range(args.patients)]
    g = torch.randn(args.patients, 6)
    g1, g2 = g+.02*torch.randn_like(g), g+.02*torch.randn_like(g)
    started = time.perf_counter()
    if args.worker == 'stream':
        output = stream_forward(model, bags, g, args.chunk, True, True, rna_views=(g1, g2))
        log_rates, auxiliary, local = output['log_rates'], output['auxiliary'], output['local_loss']
    else:
        triples = [bag.read_block((bag.patient, 0, bag.n_patches),
                                 ('features_reference', 'features_a', 'features_b')) for bag in bags]
        x, a, b = [torch.stack([values[k] for values in triples]) for k in range(3)]
        del triples
        mask = torch.ones(args.patients, args.rows, dtype=torch.bool)
        log_rates = model(x, g, mask)['log_rates']
        first = model.encode_auxiliary(a, g1, mask, include_local=True)
        second = model.encode_auxiliary(b, g2, mask, include_local=True)
        auxiliary = dict(path1=first['path'], path2=second['path'], rna1=first['rna'], rna2=second['rna'])
        local = patient_balanced_local_correlation(first['local'], second['local'], mask)[0]
    nll = piecewise_nll(log_rates, torch.linspace(.2, 2., args.patients),
                        torch.arange(args.patients).remainder(2).float(), torch.tensor([0., 1.]))
    loss = nll + .001*(local + correlation_loss(auxiliary['path1'], auxiliary['path2'], 1)
                       + correlation_loss(auxiliary['rna1'], auxiliary['rna2'], 1)
                       + correlation_loss(auxiliary['path1'][:, :8], auxiliary['rna1'][:, :8], 1)
                       + correlation_loss(auxiliary['path1'][:, 8:], auxiliary['rna1'][:, 8:], 0))
    loss.backward()
    elapsed = time.perf_counter()-started
    gradient_norm = sum(p.grad.double().square().sum().item() for p in model.parameters() if p.grad is not None)**.5
    return dict(backend=args.worker, peak_process_memory_bytes=peak_memory_bytes(),
                forward_backward_seconds=elapsed, loss=loss.item(), gradient_l2=gradient_norm,
                max_feature_rows_per_read=max(b.max_rows_read for b in bags),
                feature_read_calls=sum(b.read_calls for b in bags))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out')
    parser.add_argument('--worker', choices=['padded', 'stream'])
    parser.add_argument('--patients', type=int, default=4)
    parser.add_argument('--rows', type=int, default=8192)
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--chunk', type=int, default=512)
    args = parser.parse_args()
    if min(args.rows, args.features, args.chunk) < 1 or args.patients < 2:
        raise ValueError('Positive dimensions and at least two synthetic patients required')
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if not args.out or Path(args.out).exists():
        raise ValueError('Provide a new --out JSON path')
    results = []
    for backend in ('padded', 'stream'):
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', backend,
                   '--patients', str(args.patients), '--rows', str(args.rows),
                   '--features', str(args.features), '--chunk', str(args.chunk)]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        results.append(json.loads(result.stdout))
    report = dict(status='synthetic_engineering_benchmark', patients=args.patients,
                  patches_per_patient=args.rows, feature_dim=args.features, chunk_size=args.chunk,
                  runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                               device='cpu', platform=platform.platform(), threads=1), results=results,
                  peak_memory_reduction_fraction=1-results[1]['peak_process_memory_bytes']/results[0]['peak_process_memory_bytes'],
                  caveats=['Generated tensors only; no H5 throughput, real WSI, UNI or patient evaluation',
                           'Fresh-process process peak includes runtime and allocator overhead',
                           'Same fixed patient batch; patient minibatch optimization not benchmarked',
                           'Stream uses checkpoint recomputation and float64 centered local moments'],
                  source_sha256={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ('cancer_model.py', 'cancer_auxiliary.py', 'cancer_stream.py',
                                              'benchmark_cancer_stream.py')})
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
