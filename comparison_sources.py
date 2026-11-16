"""Immutable upstream source registry and loader for the four comparators.

The official checkouts under ``papers/comparators/`` are treated as read-only
artifacts.  This module only records their identity (repository URL, commit and
per-file SHA256), verifies the recorded hashes before a run, and imports the
upstream innovation modules from those untouched trees.

Two upstream constraints are handled without editing the upstream files:

* ``PIBD`` hard-codes ``.cuda()`` inside its Gaussian samplers, so the adapter
  supplies a CPU/CUDA neutral replacement for exactly those samplers.
* ``DeReF``/``DIMAF`` upstream modules rely on package-relative imports and a
  few CUDA-only allocations; the adapter builds the upstream classes inside a
  temporary import context and never writes back to the checkout.
"""
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARATORS = REPO_ROOT / 'papers' / 'comparators'

# One upstream module set per repository, reused for the whole process.  See
# ``_cached`` for why re-executing the upstream files would be harmful.
_MODULE_CACHE = {}

UPSTREAM = {
    'PIBD': {
        'url': 'https://github.com/zylbuaa/PIBD',
        'commit': 'bd5bd94e6f8d48e7679c6e68209a3a65c9e56a78',
        'root': COMPARATORS / 'PIBD',
        'files': {
            'models/model_PIBD.py': 'b81b9b1598497d4dd23f5006aa024d50e60be32d71f48baa758afe2ad35d2fc7',
            'models/disentangle_transformer.py': '3bd1036ea6d9751b975e3dfdeacda3e20b172b51cf395b214b9eb32544d44811',
            'models/omics_encoder.py': 'cc241066041d1472e206a76f2dd42f31e089394d4942b92f0541f630f3cfe61e',
            'models/club.py': '995af2154276dcb973a0cc6e977bc3655d2361bf968a9db14f4915d270b6e12b',
            'utils/loss_func.py': '44f15b2c052f658b10366c6bb9c4ebd091d2bf1ce7751c3dff67e1a295c30ed3',
        },
    },
    'CAMR': {
        'url': 'https://github.com/wxq-ustc/CAMR',
        'commit': '8fbae0a0bc4e9da31547b5eab5d773e6b2049e93',
        'root': COMPARATORS / 'CAMR',
        'files': {
            'networks.py': '41c3881397500abc4e84f33a48c1a7b69a7ff8237efbaa9d9fc2d9a2a161da90',
            'fusion.py': '13370f73670b2e75f71177d0f679da8549cc2dbc4013b2362a63273f73446cf9',
            'utils.py': '8bb0bd341d208e3286ba8f3263d75f7eea172f4ff2960df6f761febd3461e720',
        },
    },
    'DeReF': {
        'url': 'https://github.com/ZJUMAI/DeReF',
        'commit': '6bdbf81e76d1c7fa1d49b05f1a8fec2fbfc7b6d3',
        'root': COMPARATORS / 'DeReF',
        'files': {
            'models/deref/network.py': '497f267d0c6a958282a5c643923862e2893e6c421b8c77474bf384e52c022063',
            'models/deref/util.py': '71a4233ff44f019b0a985b77224259b7807c9879180abb9f49b9ed0c04fe1610',
            'models/deref/engine.py': 'b1e4eed3527098ae87ccc380213f46df1bdbb7d0eef566462fee91c9a35d3146',
            'utils/loss.py': '4245d81db867e6a34719e0b581a446a40cccf65554c33d16cf01d665200839ab',
            'csv/signatures.csv': '0c9175b56331656998c63f8e21e55523a406a73dca31bb4c66b9357b8b5dbf58',
        },
    },
    'DIMAF': {
        'url': 'https://github.com/Trustworthy-AI-UU-NKI/DIMAF',
        'commit': '3fe81236aafac88977b730745314b8c9906cb4d3',
        'root': COMPARATORS / 'DIMAF',
        'files': {
            'src/models/DIMAF/main_model.py': '70c6cc2ff5f9dc279982e9197876583d633bbf31d610a18c35f1e538ac875025',
            'src/models/DIMAF/layers.py': '7721aaf58068fb1c937ab020ecc669a2af4e9e45a0d87371f5f3c53eb62f652a',
            'src/models/PANTHER/main_model.py': '9d8006961929fd2b304ceccdc77c0bba672a1a3f04d61f21270442f7f5fb8366',
            'src/models/PANTHER/layers.py': '9100ee11b565d576781a95e60c6df9c5956b321fc2d54030c25b094b1d50dd1e',
            'src/models/PANTHER/network.py': '5b7da673391f122f563cf3bef4ae6347de62f4ab860e1317d7c9c7dddb2bac89',
            'src/survival/losses.py': '497d991442b45f73ddd0c654a7340be162602574fa7db7222bdb3f3fa5ae8001',
            'src/survival/metrics.py': '77f8eb9093778ab47400b73aad5d25839f8849b688ae0605f77a608bbdeef5e6',
            'src/data/data_files/hallmarks_signatures.csv': '851c61997590be3dd001e54d75cc1d40ccc09dc8be03f58fa3392be6bf9e5db9',
            'src/embeddings/prototype.py': 'bb6fccc275b3b3b06ef999863c38966383d462b093db5feef8c4e734a6453e73',
        },
    },
}


def digest(path):
    """SHA256 of one file, as stored on disk."""
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def line_ending_variants(path):
    """Return the raw, LF and CRLF SHA256 of one file.

    The recorded upstream hashes were taken on 2026-11-13 from a CRLF
    checkout, while a fresh ``git clone`` on a Unix host produces LF working
    trees.  Both are legitimate representations of the same upstream commit, so
    the verifier accepts any of the three forms and records which one it
    matched.  The upstream files are never rewritten.
    """
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    crlf = lf.replace(b'\n', b'\r\n')
    return {'raw': hashlib.sha256(raw).hexdigest(),
            'lf': hashlib.sha256(lf).hexdigest(),
            'crlf': hashlib.sha256(crlf).hexdigest()}


def verify_upstream(name=None):
    """Verify recorded hashes of one or all upstream checkouts.

    Returns a ``{name: {relative_path: sha256}}`` mapping so the value can be
    embedded into an immutable comparison plan.  The recorded digest is the
    CRLF form; the reported digest is the form actually present on disk, so
    the plan records what was really read.
    """
    names = [name] if name else list(UPSTREAM)
    result = {}
    for entry in names:
        record = UPSTREAM[entry]
        root = Path(record['root'])
        if not root.is_dir():
            raise FileNotFoundError(f'Missing upstream checkout: {root}')
        files = {}
        for relative, expected in record['files'].items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f'Missing upstream file: {path}')
            variants = line_ending_variants(path)
            if expected not in variants.values():
                raise ValueError(f'Upstream file modified: {path}')
            files[relative] = variants['raw']
        result[entry] = files
    return result


def upstream_metadata(name):
    """Static metadata recorded for one upstream repository."""
    record = UPSTREAM[name]
    return {'name': name, 'url': record['url'], 'commit': record['commit'],
            'local_checkout': str(Path(record['root']).resolve()),
            'source_files_sha256': dict(record['files'])}


def _package_search_paths(root):
    """Import roots that make the upstream package-relative imports resolve."""
    root = Path(root)
    if (root / 'models' / 'deref').is_dir():
        return [str(root)]
    if (root / 'src' / 'models' / 'DIMAF').is_dir():
        return [str(root / 'src')]
    return [str(root)]


def _load_module_from_path(module_name, path, search_paths):
    """Import a file as ``module_name`` with the given import roots.

    Bytecode caching is disabled for the duration of the import: without it
    Python writes ``__pycache__`` directories into the upstream checkout, which
    would make a read-only third-party tree dirty.  The upstream *source*
    files are never modified either way; this only avoids littering them with
    generated caches.
    """
    saved_path = list(sys.path)
    saved_module = sys.modules.pop(module_name, None)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(p) for p in search_paths]
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load {module_name} from {path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = False


def _cached(name, loader):
    """Load one upstream module set once per process.

    Re-executing an upstream file produces *new* class objects, so an
    ``isinstance`` check against a freshly loaded copy would fail even though
    both copies come from the same untouched file.  The upstream modules are
    stateless at import time, so the first successful load is reused and the
    identity of the upstream classes stays stable for the whole process.
    """
    if name not in _MODULE_CACHE:
        _MODULE_CACHE[name] = loader()
    return _MODULE_CACHE[name]


def load_pibd_modules():
    """Import the upstream PIBD innovation modules.

    ``PIBD.models.model_PIBD`` is imported because it defines ``PIB``, ``PoE``
    and ``PIBD``.  Its ``.cuda()`` calls are patched at class level by the
    adapter, not by editing the file on disk.
    """
    return _cached('PIBD', _load_pibd_modules)


def _load_pibd_modules():
    root = Path(UPSTREAM['PIBD']['root'])
    search = _package_search_paths(root)
    modules = {}
    for name, relative in (('models.club', 'models/club.py'),
                           ('models.omics_encoder', 'models/omics_encoder.py'),
                           ('models.disentangle_transformer', 'models/disentangle_transformer.py'),
                           ('utils.loss_func', 'utils/loss_func.py')):
        modules[name] = _load_module_from_path(name, root / relative, search)
    modules['models.model_PIBD'] = _load_module_from_path(
        'models.model_PIBD', root / 'models' / 'model_PIBD.py', search)
    return modules


def load_camr_modules():
    """Import the upstream CAMR innovation modules.

    ``CAMR.networks.MisaLMFGatedRec`` is the upstream shared/private encoder with
    cross-modal bilinear gating and low-rank fusion; ``CAMR.fusion`` provides
    ``LMF`` and ``Discriminator``; ``CAMR.utils`` provides ``DiffLoss``, ``MSE``
    and ``regularize_weights``.

    ``fusion.py`` executes ``options.parse_args()`` **at import time**, which
    would parse the caller's command line and write an options file into the
    checkout.  To keep the third-party tree byte-identical and side-effect
    free, a benign ``options`` module is injected into ``sys.modules`` before
    the import; the upstream ``options.py`` file itself is never executed and
    never written to.
    """
    return _cached('CAMR', _load_camr_modules)


def _load_camr_modules():
    root = Path(UPSTREAM['CAMR']['root'])
    previous = {name: sys.modules.get(name) for name in ('options', 'fusion', 'networks', 'utils')}
    stub = types.ModuleType('options')
    stub.parse_args = lambda: types.SimpleNamespace(
        gpu_ids=[], input_size=80, label_dim=1, dropout_rate=.3, hidden_size=512,
        measure=1, verbose=1, beta1=.9, beta2=.999, niter=0, niter_decay=80,
        epoch_count=1, batch_size=400, lr=4e-4, lambda_reg=2e-6, weight_decay=1e-6,
        lr_policy='linear', optimizer_type='adam', model_save='', results='',
        exp_name='camr', model_name='camr', dataroot='', datatype='')
    sys.modules['options'] = stub
    search = [str(root)]
    modules = {}
    try:
        for name, relative in (('utils', 'utils.py'), ('fusion', 'fusion.py'),
                               ('networks', 'networks.py')):
            modules[name] = _load_module_from_path(name, root / relative, search)
    finally:
        # The upstream module names are generic; they must not shadow anything
        # else in the running process once the classes have been captured.
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return modules


def load_deref_modules():
    """Import the upstream DeReF network modules."""
    return _cached('DeReF', _load_deref_modules)


def _load_deref_modules():
    root = Path(UPSTREAM['DeReF']['root'])
    search = _package_search_paths(root)
    package = types.ModuleType('models')
    package.__path__ = [str(root / 'models')]
    sys.modules.setdefault('models', package)
    deref = types.ModuleType('models.deref')
    deref.__path__ = [str(root / 'models' / 'deref')]
    sys.modules.setdefault('models.deref', deref)
    modules = {}
    for name, relative in (('models.deref.util', 'models/deref/util.py'),
                           ('models.deref.network', 'models/deref/network.py')):
        modules[name] = _load_module_from_path(name, root / relative, search)
    return modules


def load_dimaf_modules():
    """Import the upstream DIMAF attention, PANTHER and loss modules."""
    return _cached('DIMAF', _load_dimaf_modules)


def _load_dimaf_modules():
    root = Path(UPSTREAM['DIMAF']['root'])
    src = root / 'src'
    search = [str(src)]
    modules = {}
    for name, relative in (
            ('models.DIMAF.layers', 'models/DIMAF/layers.py'),
            ('models.PANTHER.network', 'models/PANTHER/network.py'),
            ('models.PANTHER.layers', 'models/PANTHER/layers.py'),
            ('survival.losses', 'survival/losses.py')):
        modules[name] = _load_module_from_path(name, src / relative, search)
    modules['models.DIMAF.main_model'] = _load_module_from_path(
        'models.DIMAF.main_model', src / 'models' / 'DIMAF' / 'main_model.py', search)
    return modules


def device_agnostic(module):
    """Return a device-agnostic Gaussian sampler for the upstream PIBD classes.

    The upstream ``gaussian_noise`` helpers call ``.cuda()`` unconditionally.
    This replacement reproduces the same sampling semantics with an explicit
    device, so the upstream innovation module is exercised unchanged on CPU and
    GPU.  Evaluation keeps the upstream fixed-seed generator semantics.
    """
    def gaussian_noise(self, samples, K=None, seed=None, k=None):
        # Upstream calls this as ``gaussian_noise(samples=..., k=..., seed=...)``
        # from ``PoE`` and as ``gaussian_noise(samples=..., K=..., seed=...)``
        # from ``PIB``; both spellings are accepted here.
        dimension = K if K is not None else k
        if dimension is None:
            raise TypeError('gaussian_noise requires the latent dimension as K or k')
        generator = None
        if not self.training:
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
        samples = torch.zeros(*samples, dimension, device='cpu')
        noise = torch.normal(samples, torch.ones_like(samples), generator=generator)
        parameter = next(self.parameters())
        return noise.to(device=parameter.device, dtype=parameter.dtype)

    module.PoE.gaussian_noise = gaussian_noise
    module.PIB.gaussian_noise = gaussian_noise
    return module


class device_redirect:
    """Scoped redirect of ``Tensor.cuda()`` to an explicit device.

    Several upstream files call ``.cuda()`` unconditionally (``PIBD`` proxy
    masks, ``DeReF`` class-token expansions).  Editing those files would break
    the promise that the third-party checkouts stay byte-identical, so the
    call is redirected at runtime instead: ``.cuda()`` becomes a move to the
    configured device.

    The redirect is restored on exit, and it is only ever active inside the
    upstream forward pass, never during data loading.
    """

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self._original = None

    def __enter__(self):
        original = torch.Tensor.cuda

        def redirected(tensor, *args, **kwargs):
            return tensor.to(self.device)

        self._original = original
        torch.Tensor.cuda = redirected
        return self

    def __exit__(self, *exc):
        if self._original is not None:
            torch.Tensor.cuda = self._original
            self._original = None
        return False
