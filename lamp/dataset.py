import os
from collections import defaultdict

import numpy as np
import pandas as pd
import random
import xarray as xr
from bionemo.noodles.nvfaidx import NvFaidx
from lightning.pytorch import LightningDataModule
from tangermeme.ersatz import substitute
from tangermeme.io import read_meme
from tangermeme.utils import one_hot_encode, reverse_complement
from torch import randint, stack
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

DATA_DIR = os.path.join(os.environ.get('LAMP_ROOT', ''), "data")
REF_GENOME_PATH = os.path.join(DATA_DIR, "hg38_genome.fa")
DEFAULT_BACKGROUND_SNVS = os.path.join(DATA_DIR, "background_snvs", "background_snvs_Whole_Blood.csv.gz")
ENFORMER_SEQ_LEN = 196608


def prep_GTExDataset(fold, split):
    print("Using Scaled Expression value")
    expression_filepath = os.path.join(DATA_DIR, "expression", f"concat_fold{fold}.nc")
    conc = xr.open_dataarray(expression_filepath, engine='netcdf4')

    donor_dir = os.path.join(DATA_DIR, 'cross_validation_folds', 'gtex', 'cv_folds')
    split_name = {'train': 'train', 'valid': 'val', 'test': 'test'}[split]
    donor_path = os.path.join(donor_dir, f"person_ids-{split_name}-fold{fold}.txt")
    return DATA_DIR, donor_path, conc, expression_filepath


class GTExDataset(Dataset):
    def __init__(self,
                 tissues_to_train: list,
                 requested_regions: list,
                 desired_seq_len: int,
                 num_individuals_per_gene: int,
                 gene_expression_df,
                 donor_list_path: str,
                 DATA_DIR: str) -> None:
        """
        Args:
            tissues_to_train (list): GTEx tissues to train on.
            requested_regions (list): Genes to use.
            desired_seq_len (int): Length of the TSS-centered input DNA sequence.
            num_individuals_per_gene (int): Number of people sampled consecutively for one gene before moving to the next gene. -1 uses all people (validation/test).
            gene_expression_df (xr.DataArray): Expression values with dims (gene_name, donor_id, tissue).
            donor_list_path (str): Line-separated txt file of GTEx donor IDs in this cross-validation split.
            DATA_DIR (str): Data directory.
        """
        self.DATA_DIR = DATA_DIR
        self.desired_seq_len = desired_seq_len
        self.donor_list_path = donor_list_path
        self.tissues_to_train = tissues_to_train
        self.filtered_metadata = pd.read_csv(os.path.join(self.DATA_DIR, "filtered_metadata.csv"))

        assert type(tissues_to_train) == list
        assert type(requested_regions) == list
        # genes that overlap the same 128bp bin are named `gene1/gene2` in the regions file and are not supported
        assert all(['/' not in x for x in requested_regions])

        self.gene_expression_df = gene_expression_df
        genes_in_df = set([x.item() for x in self.gene_expression_df.gene_name.values])
        requested_regions = sorted(list(set(requested_regions) & genes_in_df))
        self.all_regions = pd.read_csv(os.path.join(self.DATA_DIR, "Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval_NoDuplicatesNoOverlappingBins.csv"))
        self.genomic_regions_df = self.all_regions[self.all_regions['gene_name'].isin(requested_regions)].reset_index(drop=True)
        self.genes_in_dataset = list(self.genomic_regions_df['gene_name'].unique())

        # donors with WGS & RNA-seq in the desired tissue(s) that belong to this split
        self.individuals_in_split = self.select_donors()
        if num_individuals_per_gene == -1:
            self.num_individuals_per_gene = len(self.individuals_in_split)
        else:
            self.num_individuals_per_gene = num_individuals_per_gene
        # number of groups of `num_individuals_per_gene` people per gene in one epoch
        self.n_gene_replicate_batches_per_epoch = len(self.individuals_in_split) // self.num_individuals_per_gene
        self.consensus_seq_dir = os.path.join(self.DATA_DIR, "ConsensusSeqs_SNPsOnlyUnphased")
        self.fastas = self._cache_fastas()

    def _cache_fastas(self):
        fastas = defaultdict(lambda: {})
        for gtex_id in self.individuals_in_split:
            for haplotype in [1, 2]:
                gtex_path = self.get_path_to_consensus_seq(gtex_id, haplotype)
                fastas[gtex_id][f'H{haplotype}'] = NvFaidx(gtex_path, faidx_path=gtex_path + ".fai", ignore_existing_fai=False)
        return fastas

    def _get_gene_and_individual_from_idx(self, idx):
        # consecutive idx values give different people for the same gene until num_individuals_per_gene is reached
        region_idx = idx // self.num_individuals_per_gene
        region_row = self.region_rows_in_epoch[region_idx]
        individual_idx = idx % self.num_individuals_per_gene
        individual = self.indivs_per_epoch[region_idx][individual_idx]
        return region_row, individual

    def select_donors(self):
        """
        Donors from this cross-validation split that have expression data in ANY of the desired tissue(s).
        """
        with open(self.donor_list_path, 'r') as f:
            GTEx_data_split_IDs = [ID for ID in f.read().split("\n") if ID != '']

        donors_per_tissue = []
        for tissue in self.tissues_to_train:
            donors_per_tissue.extend(self.filtered_metadata[self.filtered_metadata['clean_tissue'] == tissue]['donor_id'].tolist())
        donors_w_expr_data_any_tissue = list(set(donors_per_tissue))
        desired_donors = [x for x in donors_w_expr_data_any_tissue if x in GTEx_data_split_IDs]
        return sorted(desired_donors)

    def get_path_to_consensus_seq(self, donor_id, haplotype_num):
        return os.path.join(self.consensus_seq_dir, f"{donor_id}_consensus_H{haplotype_num}.fa")

    def shuffle_and_define_epoch(self):
        """
        Shuffles the dataset so each batch contains only one gene. The same gene appears in consecutive batches until
        num_individuals_per_gene people have been used, then a new gene starts. If there are enough people, each gene
        gets n_gene_replicate_batches_per_epoch such groups per epoch, which need not be consecutive.
        """
        self.genomic_regions_df = self.genomic_regions_df.sample(frac=1).reset_index(drop=True)

        self.indivs_per_epoch = []  # list of lists: people assigned to each gene group
        self.region_rows_in_epoch = []  # gene for each gene group
        for i in range(0, len(self.genomic_regions_df), 1):
            indivs_for_gene = random.sample(self.individuals_in_split, self.num_individuals_per_gene * self.n_gene_replicate_batches_per_epoch)
            split_indivs_for_gene = np.array_split(indivs_for_gene, self.n_gene_replicate_batches_per_epoch)
            for j in range(self.n_gene_replicate_batches_per_epoch):
                self.region_rows_in_epoch.append(self.genomic_regions_df.iloc[i])
                self.indivs_per_epoch.append(split_indivs_for_gene[j].tolist())

        # shuffle the order of gene groups so the same gene's groups aren't back to back
        shuffled_idxs = np.random.permutation(len(self.region_rows_in_epoch))
        self.region_rows_in_epoch = [self.region_rows_in_epoch[i] for i in shuffled_idxs]
        self.indivs_per_epoch = [self.indivs_per_epoch[i] for i in shuffled_idxs]

    def __len__(self):
        return len(self.genomic_regions_df) * self.num_individuals_per_gene * self.n_gene_replicate_batches_per_epoch

    def _close_fastas(self):
        if hasattr(self, 'fastas'):
            del self.fastas

    def __del__(self):
        self._close_fastas()


class PersonalGenomeSimulatorPropEffectSize(GTExDataset):
    """
    Personal genomes with one synthetic SNV per gene, placed inside a TF motif. Each person is randomly assigned the
    effect or other allele (with probability effect_allele_freq). Expression is shifted by `delta` for effect allele
    carriers, where delta is chosen so the synthetic variant explains `target_r2` of expression variance.
    """
    def __init__(self,
                 seq_len,
                 fold,
                 split,
                 gene_list,
                 target_r2,
                 position_idx,
                 effect_allele_freq,
                 n_per_gene,
                 tissues_to_train,
                 motif_positions,
                 rc_rate=0.5,
                 expression_delta=None):

        self.target_r2 = target_r2
        self.seq_len = seq_len
        self.fold = fold
        self.split = split
        self.gene_list = gene_list
        self.effect_allele_freq = effect_allele_freq
        self.n_per_gene = n_per_gene
        self.rc_rate = rc_rate
        self.motif_positions = motif_positions
        self.pfm_dir = os.path.join(DATA_DIR, "jaspar_pfms")
        DATA_DIR_, donor_path, gene_expression_df, expression_filepath = self.prepare_GTExDataset(fold, split)
        self.expression_delta = expression_delta
        self.expression_filepath = expression_filepath

        super().__init__(tissues_to_train, gene_list, self.seq_len, n_per_gene, gene_expression_df, donor_path, DATA_DIR_)
        self.genomic_regions_df = self._redefine_regions_by_seq_len(self.genomic_regions_df)
        self.define_motif_positions()
        self.define_alleles(position_idx)
        self.define_genotypes()
        self.define_expression_vals()
        self._test_expectations()

    def prepare_GTExDataset(self, fold, split):
        return prep_GTExDataset(fold, split)

    def _test_expectations(self):
        print(f"Testing expectations for Dataset of type {type(self)}")
        assert all(self.motif_positions.gene_name.value_counts() == 1), "Only one synthetic variant per gene is supported"
        assert len(self.tissues_to_train) == 1, "Only single tissue expression values are supported"

    def _redefine_regions_by_seq_len(self, regions):
        return redefine_regions_by_seq_len(regions, self.seq_len)

    def define_motif_positions(self):
        """
        Keep only genes that have a pre-defined motif position, then re-define the epoch over those genes.
        """
        self.genomic_regions_df = self.genomic_regions_df[self.genomic_regions_df['gene_name'].isin(self.motif_positions['gene_name'].unique())]
        self.motif_positions = self.motif_positions[self.motif_positions['gene_name'].isin(self.genomic_regions_df['gene_name'].unique())]
        self.genes_in_dataset = list(self.genomic_regions_df['gene_name'].unique())
        assert sorted(self.genes_in_dataset) == sorted(self.motif_positions['gene_name'].unique())
        assert self.motif_positions.sort_values(by='gene_name')[['gene_name', 'short_start0', 'short_end0', 'gene_start0']].drop_duplicates().reset_index(drop=True).equals(
            self.genomic_regions_df.sort_values(by='gene_name')[['gene_name', 'short_start0', 'short_end0', 'gene_start0']].reset_index(drop=True)
        )
        self.shuffle_and_define_epoch()
        print(f"Used {len(self.genes_in_dataset)} genes in {self.split} dataset.")

    def define_alleles(self, position_idx):
        self.motif_positions = define_alleles(self.motif_positions, position_idx, self.pfm_dir)

    def define_genotypes(self):
        """
        Assign each person the effect or other allele for each gene's synthetic variant
        """
        self.genotype_dict = {}
        for gene in self.genes_in_dataset:
            if gene not in self.genotype_dict:
                self.genotype_dict[gene] = {}
            allele_freq = self.effect_allele_freq[gene]
            for indiv in self.individuals_in_split:
                allele_choice = np.random.choice(['effect_allele', 'other_allele'], p=[allele_freq, 1 - allele_freq])
                self.genotype_dict[gene][indiv] = allele_choice

    def define_expression_vals(self):
        """
        expression = original expression + delta * genotype, where genotype is 1 for the effect allele and 0 otherwise.
        delta = sqrt(r2 / (1 - r2) * var(y) / var(g)), unless a delta for the gene was passed in.
        """
        if self.expression_delta is None:
            self.expression_delta = {}
        self.expression_vals = {}
        for gene in self.genes_in_dataset:
            self.expression_vals[gene] = {}
            donor_list = list(self.genotype_dict[gene].keys())
            genotype_arr = (np.array([self.genotype_dict[gene][indiv] for indiv in donor_list]) == 'effect_allele').astype(int)
            y = self.gene_expression_df.sel(gene_name=gene, tissue=self.tissues_to_train, donor_id=donor_list).to_dataframe().reset_index()
            assert all(y['donor_id'] == donor_list)
            y = y['expression']

            target_r2 = self.target_r2[gene]
            # if no one has the synthetic allele, or target R2 is 0, expression is not changed
            if (target_r2 == 0) or np.all(genotype_arr == 0):
                self.expression_delta[gene] = 0

            if gene not in self.expression_delta:
                var_y = np.var(y, ddof=1)
                var_g = np.var(genotype_arr, ddof=1)
                delta = np.sqrt((target_r2 / (1 - target_r2)) * (var_y / var_g))
            else:
                delta = self.expression_delta[gene]
            self.expression_delta[gene] = delta
            y_modified = y + (delta * genotype_arr)
            for i, indiv in enumerate(donor_list):
                self.expression_vals[gene][indiv] = y_modified[i].item()

    def _get_both_seq_str(self, gtex_id, region_chr, region_start, region_end):
        seq1 = self.fastas[gtex_id]['H1'][region_chr][region_start:region_end].upper()
        seq2 = self.fastas[gtex_id]['H2'][region_chr][region_start:region_end].upper()
        return seq1, seq2

    def _get_single_GTEx_donor_sequence(self, gene_name, gtex_id, region_chr, region_start, region_end, use_rc):
        """
        Regions are 0-based, half open.
        Returns the diploid one-hot sequence (heterozygous positions are 0.5/0.5), shape (seq_len, 4).
        """
        seq1, seq2 = self._get_both_seq_str(gtex_id, region_chr, region_start, region_end)
        assert len(seq1) == len(seq2) == self.desired_seq_len, f"Offending region: {region_chr}:{region_start}-{region_end}"
        seq1_one_hot = one_hot_encode(seq1).unsqueeze(0)
        seq2_one_hot = one_hot_encode(seq2).unsqueeze(0)
        seq1_one_hot, seq2_one_hot, allele_choice = self._substitute_allele(seq1_one_hot, seq2_one_hot, gene_name, gtex_id)
        diploid_one_hot = ((seq1_one_hot + seq2_one_hot) / 2).squeeze(0)
        if use_rc:
            diploid_one_hot = reverse_complement(diploid_one_hot)
        diploid_one_hot = diploid_one_hot.permute(1, 0)
        return diploid_one_hot, allele_choice

    def _get_single_GTEx_donor_expression(self, sampled_individual, gene_name):
        return np.array(self.expression_vals[gene_name][sampled_individual])[np.newaxis]  # tissue axis

    def _substitute_allele(self, seq1_one_hot, seq2_one_hot, gene_name, indiv):
        gene_motif = self.motif_positions[self.motif_positions['gene_name'] == gene_name]
        allele_choice = self.genotype_dict[gene_name][indiv]
        allele = gene_motif[allele_choice].item()  # 'effect_allele' or 'other_allele' column
        synthetic_snp_position = gene_motif['synthetic_snp_position'].item()
        seq1_one_hot = substitute(seq1_one_hot, allele, start=synthetic_snp_position)
        seq2_one_hot = substitute(seq2_one_hot, allele, start=synthetic_snp_position)
        return seq1_one_hot, seq2_one_hot, allele_choice

    def generate_train_batch_one_gene(self, sampled_individual, region_chr, short_start0, short_end0, gene_name):
        use_rc = np.random.choice([True, False], p=[self.rc_rate, 1 - self.rc_rate])
        dna_seq, allele_choice = self._get_single_GTEx_donor_sequence(gene_name, sampled_individual, region_chr, short_start0, short_end0, use_rc)
        gene_expression = self._get_single_GTEx_donor_expression(sampled_individual, gene_name)
        return dna_seq, gene_expression, use_rc

    def __getitem__(self, idx):
        region_row, individual = self._get_gene_and_individual_from_idx(idx)
        dna_vector, expression_vector, use_rc = self.generate_train_batch_one_gene(
            individual,
            region_row['seqnames'],
            region_row['short_start0'],
            region_row['short_end0'],
            region_row['gene_name']
        )
        return dna_vector, expression_vector, region_row['gene_name'], individual, use_rc.item(), idx


def augment_slide_motif_w_flank(gene_motif, seq1_one_hot, seq2_one_hot, synthetic_snp_position, flank, min_tss_dist):
    """
    Take the causal motif plus `flank` nt on either side of it and slide it closer to or further from the TSS, staying on the same side of the TSS.
    The nucleotides it slides past get pushed into the space the motif + flank left behind, so nothing is deleted or duplicated.
    The same shift is used for both haplotypes. The motif + flank can't be slid within `min_tss_dist` nt of the TSS.
    """
    len_seq = seq1_one_hot.shape[-1]
    center = len_seq // 2  # TSS
    motif_rel_start = (gene_motif['chromStart'] - gene_motif['short_start0']).item()
    motif_rel_end = (gene_motif['chromEnd'] - gene_motif['short_start0']).item()
    motif_w_flank_start = max(0, motif_rel_start - flank)
    motif_w_flank_end = min(len_seq, motif_rel_end + flank)

    # smallest (furthest left) and largest (furthest right) shift that keeps the motif + flank on the same side of the TSS.
    # crossing the TSS would push the TSS off the center of the sequence
    if motif_rel_start < center:
        min_shift = -motif_w_flank_start
        max_shift = center - min_tss_dist - motif_w_flank_end
    else:
        min_shift = center + min_tss_dist - motif_w_flank_start
        max_shift = len_seq - motif_w_flank_end
    # motif is already within min_tss_dist of the TSS: there is no valid place to move it
    if min_shift > max_shift:
        return seq1_one_hot, seq2_one_hot, synthetic_snp_position

    shift = randint(low=min_shift, high=max_shift + 1, size=(1,)).item()  # negative = slide left, positive = slide right

    # stretch = motif + flank plus all the nt it slides past. Rolling the stretch by `shift` moves the motif + flank to the
    # other end of the stretch and moves the nt it slid past into the space it left behind. Sequence outside the stretch is unchanged
    stretch_start = min(motif_w_flank_start, motif_w_flank_start + shift)
    stretch_end = max(motif_w_flank_end, motif_w_flank_end + shift)
    seq1_one_hot[..., stretch_start:stretch_end] = seq1_one_hot[..., stretch_start:stretch_end].roll(shift, dims=-1)
    seq2_one_hot[..., stretch_start:stretch_end] = seq2_one_hot[..., stretch_start:stretch_end].roll(shift, dims=-1)

    synthetic_snp_position = synthetic_snp_position + shift  # causal allele follows the motif
    return seq1_one_hot, seq2_one_hot, synthetic_snp_position


class PersonalGenomeSlideMotifWithFlank(PersonalGenomeSimulatorPropEffectSize):
    """
    Training augmentation: every time a sequence is drawn, slide the causal motif + `flank` nt to a random position on the
    same side of the TSS (see `augment_slide_motif_w_flank`), then place the synthetic allele.
    """
    def __init__(self, *args, flank, min_tss_dist, **kwargs):
        self.flank = flank
        self.min_tss_dist = min_tss_dist
        super().__init__(*args, **kwargs)

    def _substitute_allele(self, seq1_one_hot, seq2_one_hot, gene_name, indiv):
        gene_motif = self.motif_positions[self.motif_positions['gene_name'] == gene_name]
        allele_choice = self.genotype_dict[gene_name][indiv]
        allele = gene_motif[allele_choice].item()
        synthetic_snp_position = gene_motif['synthetic_snp_position'].item()
        seq1_one_hot, seq2_one_hot, synthetic_snp_position = augment_slide_motif_w_flank(
            gene_motif, seq1_one_hot, seq2_one_hot, synthetic_snp_position, self.flank, self.min_tss_dist)
        seq1_one_hot = substitute(seq1_one_hot, allele, start=synthetic_snp_position)
        seq2_one_hot = substitute(seq2_one_hot, allele, start=synthetic_snp_position)
        return seq1_one_hot, seq2_one_hot, allele_choice


class TwinDifferencePersonalGenomeSimulatorPropEffectSize(PersonalGenomeSimulatorPropEffectSize):
    """
    TwinNet evaluation dataset. Returns two sequences: the personal genome (with the effect or other allele) and the
    reference genome with the other allele. The target is the person's expression (the average is not subtracted).
    """
    def _cache_fastas(self):
        fastas = super()._cache_fastas()
        fastas['ref']['H1'] = NvFaidx(REF_GENOME_PATH, faidx_path=REF_GENOME_PATH + ".fai", ignore_existing_fai=False)
        fastas['ref']['H2'] = fastas['ref']['H1']
        return fastas

    def _substitute_allele(self, seq1_one_hot, seq2_one_hot, gene_name, indiv):
        """
        The reference genome always gets the other allele. Personal genomes get their assigned allele.
        """
        gene_motif = self.motif_positions[self.motif_positions['gene_name'] == gene_name]
        allele_choice = 'other_allele' if indiv == 'ref' else self.genotype_dict[gene_name][indiv]
        allele = gene_motif[allele_choice].item()
        synthetic_snp_position = gene_motif['synthetic_snp_position'].item()
        seq1_one_hot = substitute(seq1_one_hot, allele, start=synthetic_snp_position)
        seq2_one_hot = substitute(seq2_one_hot, allele, start=synthetic_snp_position)
        return seq1_one_hot, seq2_one_hot, allele_choice

    def __getitem__(self, idx):
        region_row, individual = self._get_gene_and_individual_from_idx(idx)
        pers_genome, expression_vector, use_rc = self.generate_train_batch_one_gene(
            individual,
            region_row['seqnames'],
            region_row['short_start0'],
            region_row['short_end0'],
            region_row['gene_name']
        )
        ref_genome, _ = self._get_single_GTEx_donor_sequence(
            region_row['gene_name'], 'ref', region_row['seqnames'], region_row['short_start0'], region_row['short_end0'], use_rc)
        X = stack([pers_genome, ref_genome], dim=0)  # X[0] is the personal genome (alt), X[1] is the reference
        return X, expression_vector, region_row['gene_name'], individual, use_rc.item(), idx


class RefGenomeVariantEffectDataset(Dataset):
    """
    Train on variant effects using the reference genome (hg38).

    GTEx donors in the split are used only to compute each synthetic variant's effect size (delta, same formula as
    PersonalGenomeSimulatorPropEffectSize) and each gene's mean expression. Each training example is then the reference
    sequence with one variant substituted:
        - synthetic variant: one per gene, inside the TF motif, effect size = delta
        - background SNVs: natural SNVs with effect size = `background_delta`, read from the `background_snvs` CSV.
          Used by default. background_snvs='none' trains on synthetic variants only, which makes the model overfit
          and predict large effect sizes for all variants
    Genes whose synthetic variant has an effect allele frequency of 0 (rare-variant genes) are left out.

    paired=False (Variformer): each variant gives two examples. 'effect_allele' (target = gene mean + delta) and
        'other_allele' (target = gene mean).
    paired=True (TwinNet): each variant gives one example, stack([effect, other]) with target = delta.

    background_sampling:
        'equal': synthetic variants are duplicated so synthetic and background examples are 1:1 per epoch
        'capped': each gene keeps at most `max_background_per_variant` of its own background SNVs
    """
    def __init__(self,
                 seq_len,
                 fold,
                 split,
                 gene_list,
                 target_r2,
                 position_idx,
                 effect_allele_freq,
                 n_per_gene,  # unused, kept so all datasets share the same positional arguments
                 tissues_to_train,
                 motif_positions,
                 rc_rate=0.5,
                 expression_delta=None,
                 background_snvs=DEFAULT_BACKGROUND_SNVS,
                 background_sampling='equal',
                 max_background_per_variant=None,
                 paired=False):
        self.seq_len = seq_len
        self.desired_seq_len = seq_len
        self.fold = fold
        self.split = split
        self.target_r2 = target_r2
        self.effect_allele_freq = effect_allele_freq
        self.tissues_to_train = tissues_to_train
        self.rc_rate = rc_rate
        self.paired = paired
        self.pfm_dir = os.path.join(DATA_DIR, "jaspar_pfms")
        assert len(tissues_to_train) == 1, "Only single tissue expression values are supported"

        _, self.donor_list_path, self.gene_expression_df, self.expression_filepath = prep_GTExDataset(fold, split)
        self.filtered_metadata = pd.read_csv(os.path.join(DATA_DIR, "filtered_metadata.csv"))
        self.donors = GTExDataset.select_donors(self)

        regions = pd.read_csv(os.path.join(DATA_DIR, "Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval_NoDuplicatesNoOverlappingBins.csv"))
        regions = regions[regions['gene_name'].isin(gene_list) & regions['gene_name'].isin(motif_positions['gene_name'])]
        regions = regions[regions['gene_name'].isin([x.item() for x in self.gene_expression_df.gene_name.values])]
        self.regions = redefine_regions_by_seq_len(regions.reset_index(drop=True), seq_len)
        self.genes_in_dataset = list(self.regions['gene_name'].unique())
        window_cols = ['gene_name', 'short_start0', 'short_end0']
        assert motif_positions[motif_positions['gene_name'].isin(self.genes_in_dataset)][window_cols].drop_duplicates().sort_values('gene_name').reset_index(drop=True).equals(
            self.regions[window_cols].sort_values('gene_name').reset_index(drop=True)
        ), f"motif_positions windows don't match seq_len={seq_len}"
        print(f"Used {len(self.genes_in_dataset)} genes in {self.split} dataset.")
        self.motif_positions = define_alleles(motif_positions[motif_positions['gene_name'].isin(self.genes_in_dataset)], position_idx, self.pfm_dir)

        self.define_synthetic_effects(expression_delta)
        self.remove_rare_var_genes()
        self.build_variant_table(background_snvs, background_sampling, max_background_per_variant)

        self.individuals_in_split = ['effect_allele'] if paired else ['effect_allele', 'other_allele']
        self.num_individuals_per_gene = len(self.individuals_in_split)
        self.n_gene_replicate_batches_per_epoch = 1
        self.fastas = self._cache_fastas()
        self.shuffle_and_define_epoch()

    def define_synthetic_effects(self, expression_delta):
        """
        Assign synthetic genotypes to GTEx donors and compute delta for each gene (same as PersonalGenomeSimulatorPropEffectSize),
        plus each gene's mean expression across donors.
        """
        self.expression_delta = {} if expression_delta is None else expression_delta
        self.avg_expression = {}
        for gene in self.genes_in_dataset:
            allele_freq = self.effect_allele_freq[gene]
            genotype_arr = np.array([np.random.choice([1, 0], p=[allele_freq, 1 - allele_freq]) for _ in self.donors])
            y = self.gene_expression_df.sel(gene_name=gene, tissue=self.tissues_to_train[0], donor_id=self.donors).values
            self.avg_expression[gene] = float(y.mean())
            target_r2 = self.target_r2[gene]
            if (target_r2 == 0) or np.all(genotype_arr == 0):
                self.expression_delta[gene] = 0
            if gene not in self.expression_delta:
                self.expression_delta[gene] = np.sqrt((target_r2 / (1 - target_r2)) * (np.var(y, ddof=1) / np.var(genotype_arr, ddof=1)))

        self.genotype_dict = {gene: {'effect_allele': 'effect_allele', 'other_allele': 'other_allele'} for gene in self.genes_in_dataset}
        self.expression_vals = {gene: {'effect_allele': self.avg_expression[gene] + self.expression_delta[gene],
                                       'other_allele': self.avg_expression[gene]} for gene in self.genes_in_dataset}

    def remove_rare_var_genes(self):
        """
        Genes with effect_allele_freq == 0 are held out of variant training (they're only seen at validation/test)
        """
        rare_var_genes = [gene for gene in self.genes_in_dataset if self.effect_allele_freq[gene] == 0]
        self.genes_in_dataset = [x for x in self.genes_in_dataset if x not in rare_var_genes]
        self.regions = self.regions[~self.regions['gene_name'].isin(rare_var_genes)]
        self.motif_positions = self.motif_positions[~self.motif_positions['gene_name'].isin(rare_var_genes)]
        for gene in rare_var_genes:
            for d in [self.expression_delta, self.expression_vals, self.genotype_dict, self.avg_expression]:
                del d[gene]

    def build_variant_table(self, background_snvs, background_sampling, max_background_per_variant):
        """
        One row per (gene, variant) with the variant's position in the sequence, its alleles, and its effect size.
        """
        region_cols = ['gene_name', 'seqnames', 'short_start0', 'short_end0']
        synthetic = self.motif_positions[['gene_name', 'synthetic_snp_position', 'effect_allele', 'other_allele']].copy()
        synthetic['delta'] = synthetic['gene_name'].map(self.expression_delta)
        synthetic['snp_type'] = 'synthetic'
        if str(background_snvs).lower() == 'none':
            variants = synthetic
        else:
            background_snvs = pd.read_csv(background_snvs)
            background = background_snvs[background_snvs['gene_name'].isin(self.genes_in_dataset)]
            background = background[['gene_name', 'synthetic_snp_position', 'effect_allele', 'other_allele', 'background_delta']].rename(columns={'background_delta': 'delta'})
            background['snp_type'] = 'background'
            if background_sampling == 'equal':
                # duplicate synthetic variants and trim background SNVs so they are 1:1
                n_repeats = len(background) // len(synthetic)
                background = background.sample(n=n_repeats * len(synthetic))
                synthetic = pd.concat([synthetic] * n_repeats)
            elif background_sampling == 'capped':
                background = background.sample(frac=1).groupby('gene_name').head(max_background_per_variant)
            else:
                raise ValueError(f"background_sampling {background_sampling} is not supported")
            variants = pd.concat([synthetic, background])
        self.genomic_regions_df = variants.merge(self.regions[region_cols], on='gene_name').reset_index(drop=True)
        print(f"Variant table: {self.genomic_regions_df['snp_type'].value_counts().to_dict()}")

    def _cache_fastas(self):
        return {'ref': NvFaidx(REF_GENOME_PATH, faidx_path=REF_GENOME_PATH + ".fai", ignore_existing_fai=False)}

    def shuffle_and_define_epoch(self):
        self.genomic_regions_df = self.genomic_regions_df.sample(frac=1).reset_index(drop=True)

    def _get_sequence(self, row, allele, use_rc):
        seq = self.fastas['ref'][row['seqnames']][row['short_start0']:row['short_end0']].upper()
        assert len(seq) == self.seq_len
        one_hot = substitute(one_hot_encode(seq).unsqueeze(0), allele, start=row['synthetic_snp_position']).squeeze(0).float()
        if use_rc:
            one_hot = reverse_complement(one_hot)
        return one_hot.permute(1, 0)

    def __getitem__(self, idx):
        row = self.genomic_regions_df.iloc[idx // self.num_individuals_per_gene]
        individual = self.individuals_in_split[idx % self.num_individuals_per_gene]
        use_rc = np.random.choice([True, False], p=[self.rc_rate, 1 - self.rc_rate])
        mean = self.avg_expression[row['gene_name']]
        if self.paired:
            X = stack([self._get_sequence(row, row['effect_allele'], use_rc), self._get_sequence(row, row['other_allele'], use_rc)], dim=0)
            y = row['delta']
        else:
            X = self._get_sequence(row, row[individual], use_rc)
            y = mean + row['delta'] if individual == 'effect_allele' else mean
        return X, np.array(y)[np.newaxis], row['gene_name'], individual, use_rc.item(), idx

    def __len__(self):
        return len(self.genomic_regions_df) * self.num_individuals_per_gene

    def _close_fastas(self):
        if hasattr(self, 'fastas'):
            del self.fastas

    def __del__(self):
        self._close_fastas()


def redefine_regions_by_seq_len(regions, seq_len):
    """
    Regions file is 1-based, half open and 196,608 bp long. Convert to 0-based and add TSS-centered `short_start0`/`short_end0` of length seq_len
    """
    assert all((regions.ends - regions.starts) == ENFORMER_SEQ_LEN)
    regions[['starts', 'ends', 'gene_start']] = regions[['starts', 'ends', 'gene_start']].astype(int) - 1
    regions = regions.rename(columns={'starts': 'start_0', 'ends': 'end_0', 'gene_start': 'gene_start0'})
    region_center = regions.end_0 - (ENFORMER_SEQ_LEN // 2)
    regions['short_start0'] = region_center - (seq_len // 2)
    regions['short_end0'] = region_center + (seq_len // 2)
    return regions


def define_alleles(motif_positions, position_idx, pfm_dir):
    """
    For the motif position `position_idx`, the effect allele is the nucleotide with the highest PFM value and the
    other allele is the nucleotide with the lowest. For motifs on the - strand, the position and alleles are reverse complemented.
    Skipped if motif_positions already defines the alleles.
    """
    if all(col in motif_positions.columns for col in ['effect_allele', 'other_allele', 'position_in_motif', 'synthetic_snp_position']):
        print("Not re-defining Allele information")
        return motif_positions
    print("Re-defining Allele information")
    motif_positions = motif_positions.copy()
    for motif_id in motif_positions['name'].unique():
        motif_name = motif_positions[motif_positions['name'] == motif_id]['TFName'].unique().item()
        pfm = read_meme(os.path.join(pfm_dir, f"{motif_id}.meme"))[f"{motif_id} {motif_name}"]
        nts = ['A', 'C', 'G', 'T']
        effect_allele = nts[pfm[:, position_idx].argmax()]
        other_allele = nts[pfm[:, position_idx].argmin()]
        position_idx_rc = pfm.shape[1] - position_idx - 1
        is_motif = motif_positions['name'] == motif_id
        motif_positions.loc[(motif_positions['strand'] == '+') & is_motif, ['effect_allele', 'other_allele', 'position_in_motif']] = [effect_allele, other_allele, position_idx]
        motif_positions.loc[(motif_positions['strand'] == '-') & is_motif, ['effect_allele', 'other_allele', 'position_in_motif']] = [reverse_complement(effect_allele), reverse_complement(other_allele), position_idx_rc]
    motif_positions['synthetic_snp_position'] = (motif_positions['chromStart'] + motif_positions['position_in_motif'] - motif_positions['short_start0']).astype(int)
    return motif_positions


class DoubleRCDataset(Dataset):
    """
    Returns every example from the input dataset twice, without and with reverse complementation
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.genomic_regions_df = self.base_dataset.genomic_regions_df
        self.num_individuals_per_gene = self.base_dataset.num_individuals_per_gene * 2
        self.n_gene_replicate_batches_per_epoch = self.base_dataset.n_gene_replicate_batches_per_epoch

    def __len__(self):
        return len(self.base_dataset) * 2

    def __getitem__(self, idx):
        self.base_dataset.rc_rate = 1 if idx % 2 else 0
        return self.base_dataset[idx // 2]

    def shuffle_and_define_epoch(self):
        self.base_dataset.shuffle_and_define_epoch()

    def _close_fastas(self):
        self.base_dataset._close_fastas()

    def __del__(self):
        self.base_dataset.__del__()


class CustomDistributedSampler(DistributedSampler):
    """
    Splits genes across GPUs so each GPU's batches contain only one gene. Indices are not shuffled here because the
    dataset reshuffles genes and people every epoch (`shuffle_and_define_epoch`).

    Example: 937 genes, 6 people per gene, 4 GPUs. Each GPU gets 937 // 4 = 234 genes (one gene is dropped).
    GPU 0 gets indices 0-1403, GPU 1 gets 1404-2807, and so on.
    """
    def __init__(self, dataset, shuffle=False, **kwargs):
        super().__init__(dataset, shuffle=shuffle, **kwargs)
        self.dataset = dataset
        self.num_individuals_per_gene = self.dataset.num_individuals_per_gene
        self.genes_per_replica = (len(self.dataset.genomic_regions_df) // self.num_replicas) * self.dataset.n_gene_replicate_batches_per_epoch
        self.num_samples_per_replica = self.genes_per_replica * self.num_individuals_per_gene
        print(f"CustomDistributedSampler (rank {self.rank} of {self.num_replicas}): {len(self.dataset.genomic_regions_df)} genes in dataset, "
              f"{self.genes_per_replica} gene groups x {self.num_individuals_per_gene} samples per gene on this rank = {self.num_samples_per_replica} samples")

    def __iter__(self):
        start_group = self.rank * self.genes_per_replica
        end_group = start_group + self.genes_per_replica
        indices = []
        for group in range(start_group, end_group):
            group_start_idx = group * self.num_individuals_per_gene
            indices.extend(range(group_start_idx, group_start_idx + self.num_individuals_per_gene))
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica


class CustomDataModule(LightningDataModule):
    def __init__(self, train_dataset, valid_dataset, test_dataset, train_batch_size, shuffle=False):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.train_batch_size = train_batch_size
        self.shuffle = shuffle

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=8, sampler=CustomDistributedSampler(self.train_dataset, shuffle=self.shuffle))

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=8, sampler=CustomDistributedSampler(self.valid_dataset, shuffle=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=8, sampler=CustomDistributedSampler(self.test_dataset, shuffle=False))
