"""Freeze human top-level Reactome GMT entries, without accessing patient data.

The two official inputs are hash-pinned so an upstream `current` update cannot
silently alter this research resource.  Reactome data exports are CC0.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


BASE = 'https://reactome.org/download/current/'
INPUTS = {
    'ReactomePathways.gmt.zip': '6056fcd95dc8f09740c8bb64c2765bea1386ce91b9b19cd10bcd45d30451805c',
    'ReactomePathwaysRelation.txt': 'aa3609ee852442899b400bf4817b36311d820883ecd75259ed77531d93d8f1dc',
}


def fetch(name):
    request = urllib.request.Request(BASE + name, headers={'User-Agent': 'DeCUR-survival-research/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != INPUTS[name]:
        raise ValueError(f'Reactome current file changed: {name}: {digest}; audit a new release explicitly')
    return data


def main():
    destination = Path(__file__).resolve().parent.parent / 'resources'
    destination.mkdir(exist_ok=True)
    output, metadata = destination/'reactome_human_roots.gmt', destination/'reactome_human_roots.json'
    if output.exists() or metadata.exists():
        raise FileExistsError('Frozen resource already exists; do not overwrite it')
    zdata, relation = (fetch(name) for name in INPUTS)
    gmt = zipfile.ZipFile(io.BytesIO(zdata)).read('ReactomePathways.gmt').decode('utf-8')
    edges = [row.split('\t') for row in relation.decode('utf-8').splitlines()]
    edges = [(p, c) for p, c, *_ in edges if p.startswith('R-HSA-') and c.startswith('R-HSA-')]
    parents, children = {p for p, _ in edges}, {c for _, c in edges}
    entries = [row.split('\t') for row in gmt.splitlines()]
    roots = [row for row in entries if len(row) >= 3 and row[1] in parents and row[1] not in children]
    roots.sort(key=lambda x: x[1])
    if len(roots) != 29 or len({row[1] for row in roots}) != 29:
        raise ValueError('Unexpected human root count or duplicate identifiers')
    data = ''.join('\t'.join([row[0], row[1], *dict.fromkeys(row[2:])])+'\n' for row in roots).encode('utf-8')
    output.write_bytes(data)
    info = {
        'source': 'Reactome human top-level pathway gene set and pathway hierarchy',
        'source_urls': [BASE+name for name in INPUTS],
        'source_sha256': INPUTS,
        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'selection': 'Human GMT pathway IDs occurring as parents but never as children in human hierarchy; sorted by stable Reactome ID; duplicate symbols within a set removed in source order.',
        'root_count': len(roots),
        'output_sha256': hashlib.sha256(data).hexdigest(),
        'gene_identifier': 'HUGO gene symbol as provided by Reactome GMT; exact matching only',
        'licence': 'Reactome data and derived files: CC0; https://reactome.org/license',
        'patient_data_used': False,
    }
    metadata.write_text(json.dumps(info, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
