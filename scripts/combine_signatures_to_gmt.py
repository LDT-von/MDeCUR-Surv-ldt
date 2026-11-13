"""Convert the reference pathway signature table into an explicit GMT file.

The reference comparison pipeline stores its pathways as a wide CSV whose
columns are pathway names and whose cells are gene symbols.  This script
rewrites that table as a GMT file with the same membership rule: exact gene
symbol intersection with the ordered RNA schema, empty intersections dropped.

The conversion depends only on the signature table and the gene schema.  It
never reads expression values, patient labels or model outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def convert(signature_csv, genes_path, out_gmt, min_genes=1):
    genes = [line.strip() for line in Path(genes_path).read_text(encoding='utf-8-sig').splitlines()]
    if not genes or len(set(genes)) != len(genes):
        raise ValueError('The gene schema must contain unique nonempty gene symbols')
    schema = set(genes)
    table = pd.read_csv(signature_csv)
    records, skipped = [], []
    for column in table.columns:
        members = [str(gene).strip() for gene in table[column].dropna().tolist()]
        matched = sorted(set(members) & schema)
        item = {'name': str(column), 'source_genes': len(set(members)), 'matched_genes': len(matched)}
        if len(matched) < min_genes:
            skipped.append(item)
            continue
        records.append({'name': item['name'], 'genes': matched})
    if not records:
        raise ValueError('No pathway survived the gene-schema intersection')
    out = Path(out_gmt)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='\n') as handle:
        for record in records:
            handle.write('\t'.join([record['name'], record['name'], *record['genes']])+'\n')
    summary = {
        'signature_csv': str(Path(signature_csv).resolve()),
        'signature_sha256': sha256_file(signature_csv),
        'genes_file': str(Path(genes_path).resolve()),
        'genes_sha256': sha256_file(genes_path),
        'gene_count': len(genes),
        'minimum_matched_genes': min_genes,
        'pathways_written': len(records),
        'pathways_skipped': len(skipped),
        'skipped': skipped,
        'matched_gene_slots': sum(len(r['genes']) for r in records),
        'gmt': str(out.resolve()),
        'gmt_sha256': sha256_file(out),
    }
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                        encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signature-csv', required=True)
    parser.add_argument('--genes', required=True)
    parser.add_argument('--out-gmt', required=True)
    parser.add_argument('--min-genes', type=int, default=1,
                        help='Drop a pathway only when it has fewer exact gene-schema matches')
    args = parser.parse_args()
    summary = convert(args.signature_csv, args.genes, args.out_gmt, args.min_genes)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
