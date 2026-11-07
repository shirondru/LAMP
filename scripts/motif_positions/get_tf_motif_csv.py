"""
Get all hits of one JASPAR 2024 motif inside each gene's TSS-centered window and write them to a csv.gz,
the input to make_training_motif_positions.ipynb.

Hits come from JASPAR's genome-wide hg38 predictions for the motif (~200 MB for STAT1), so nothing else needs downloading.
The `score` column is the score shown in UCSC's JASPAR 2024 track.

Usage (from the LAMP root):
    python scripts/motif_positions/get_tf_motif_csv.py --motif_id MA0137.4 --out data/STAT1.csv.gz
JASPAR motif IDs are listed in data/human_CORE_JASPAR_TFs.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

JASPAR_URL = 'https://mencius-pub.uio.no/JASPAR/JASPAR_TFBSs/2024/hg38/{motif_id}.tsv.gz'
DEFAULT_REGIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data',
                               'Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval_NoDuplicatesNoOverlappingBins.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif_id", type=str, required=True, help="JASPAR 2024 motif ID, e.g. MA0137.4 (STAT1)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--regions_csv", type=str, default=DEFAULT_REGIONS, help="196,608 bp TSS-centered window for each gene")
    args = parser.parse_args()

    hits = pd.read_csv(JASPAR_URL.format(motif_id=args.motif_id), sep='\t', header=None,
                       names=['chrom', 'chromStart', 'chromEnd', 'TFName', 'jaspar_score', 'score', 'strand'])
    hits['name'] = args.motif_id

    regions = pd.read_csv(args.regions_csv)
    regions['gene_name'] = regions['gene_name'].str.replace('/', '_')  # genes sharing a TSS bin are named gene1/gene2
    per_gene = []
    for chrom, chrom_regions in regions.groupby('seqnames'):
        chrom_hits = hits[hits['chrom'] == chrom].sort_values('chromStart')
        starts = chrom_hits['chromStart'].values
        for _, region in chrom_regions.iterrows():
            # hits overlapping the window. The -100 only widens the lookup; motifs are much shorter than 100 bp
            nearby = chrom_hits.iloc[np.searchsorted(starts, region['starts'] - 100):np.searchsorted(starts, region['ends'])]
            overlapping = nearby[(nearby['chromEnd'] > region['starts']) & (nearby['chromStart'] < region['ends'])]
            per_gene.append(overlapping.assign(gene_name=region['gene_name']))
    cols = ['gene_name', 'chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'TFName']
    pd.concat(per_gene)[cols].reset_index(drop=True).to_csv(args.out)


if __name__ == '__main__':
    main()
