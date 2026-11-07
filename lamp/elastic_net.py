"""
Elastic net baseline for a trained run. Variant runs use the personal genome train dataset saved for this (train_ds_pg.pt).

For each gene, an ElasticNetCV is fit on train donors using SNP genotypes in the gene's input window as features
(the synthetic SNP plus natural SNPs from the GTEx VCF) and the simulated expression as the target. It is then evaluated
on the same validation/test donors, genotypes and expression values as the deep learning model.

Feature sets ('snps_used_for_training'):
    all: synthetic + natural SNPs. Also evaluated with all natural SNPs set to 0 ('snps_used_for_eval' = only_synthetic)
    only_synthetic: synthetic SNP only
    only_natural: natural SNPs only (train genes)

Gene groups:
    train: train genes whose synthetic variant was seen in training
    rare-train: train genes whose synthetic variant had an allele frequency of 0 during training. The EN is fit as if
        the variant had the common allele frequency, giving an upper bound on performance for these genes
    valid / test: genes not used for training. Train donors are given synthetic genotypes (with the effect sizes used
        for those genes during validation/test) so the EN can be fit

Outputs EN_pred_{valid|test}_donor_SNPs_{gene group}_genes.csv in the run's save_dir.

Usage: python elastic_net.py --save_dir <results/.../{run id}> --vcf <SNPs-only GTEx VCF/BCF>
"""
import argparse
import os
import warnings
from ast import literal_eval

import numpy as np
import pandas as pd
import pysam
import torch
import xarray as xr
from lightning.pytorch import seed_everything
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV

import dataset  # noqa: F401  (needed to unpickle saved datasets)

warnings.filterwarnings('ignore', category=ConvergenceWarning)
RANDOM_STATE = 0


def get_genotype_df(vcf, sample_names, chromosome, start_pos, end_pos):
    """
    Genotypes (0, 1, 2 alt alleles) of every variant in the region, one column per variant. Missing genotypes are set
    to 0, matching the reference allele the deep learning model sees for missing genotypes
    """
    variant_ids = []
    genotype_matrix = []
    for record in vcf.fetch(chromosome, start_pos, end_pos):
        variant_ids.append(record.id)
        genotypes = []
        for sample in sample_names:
            gt = record.samples[sample]["GT"]
            genotypes.append(0 if gt == (None, None) else sum(gt))
        genotype_matrix.append(genotypes)
    return pd.DataFrame(genotype_matrix, index=variant_ids, columns=sample_names).T


def get_gene_df(vcf, gene, genotype, expr, positions, seq_len):
    gene_genotype = genotype[genotype['gene'] == gene]
    gene_df = gene_genotype.merge(expr, on=['gene', 'individual']).rename(columns={'genotype': 'synthetic_snp'})
    gene_df.synthetic_snp = gene_df.synthetic_snp.map({'effect_allele': 2, 'other_allele': 0})  # homozygous
    chrom, start, end = positions[positions['gene_name'] == gene][['chrom', 'short_start0', 'short_end0']].drop_duplicates().values.squeeze()
    assert end - start == seq_len
    genotype_df = get_genotype_df(vcf, list(gene_df.individual), chrom, start, end)
    return gene_df.set_index('individual').join(genotype_df)


def get_X_y(gene_df, func_str):
    y = gene_df['expression']
    if func_str == 'all':
        X = gene_df.drop(columns=['gene', 'expression'])
    elif func_str == 'only_natural':
        X = gene_df.drop(columns=['gene', 'expression', 'synthetic_snp'])
    elif func_str == 'only_synthetic':
        X = gene_df[['synthetic_snp']]
    return X, y


def eval_en(gene_df, func_str, regr, selector):
    natural_snp_cols = [x for x in list(gene_df.columns) if x.startswith('chr')]
    X_test, _ = get_X_y(gene_df, func_str)
    out = gene_df[[x for x in gene_df.columns if x not in natural_snp_cols]]
    out = out.assign(y_pred=regr.predict(selector.transform(X_test)))
    out['snps_used_for_eval'] = func_str
    if func_str == 'all':
        # how well the model fit on all SNPs predicts the synthetic SNP's effect alone: set all natural SNPs to 0
        gene_df.loc[:, natural_snp_cols] = 0
        X_test, _ = get_X_y(gene_df, func_str)
        out2 = gene_df[[x for x in gene_df.columns if x not in natural_snp_cols]]
        out2 = out2.assign(y_pred=regr.predict(selector.transform(X_test)))
        out2['snps_used_for_eval'] = 'only_synthetic'
        out = pd.concat([out, out2])
    out['snps_used_for_training'] = func_str
    return out


def get_expression_and_genotype(ds):
    expression = pd.DataFrame.from_dict(ds.expression_vals, orient='index')
    expression = expression.reset_index().melt(id_vars='index', var_name='individual', value_name='expression').rename(columns={'index': 'gene'})
    genotypes = pd.DataFrame.from_dict(ds.genotype_dict, orient='index')
    genotypes = genotypes.reset_index().melt(id_vars='index', var_name='individual', value_name='genotype').rename(columns={'index': 'gene'})
    return expression, genotypes


def eval_genes(vcf, genes_to_eval, gene_str, train_genotype, train_expr, positions, seq_len, unseen_str, unseen_genotype,
               unseen_expr, save_dir, eval_types, seed, n_jobs):
    print(f"Evaluating on {gene_str} genes using {unseen_str} donors")
    out_path = os.path.join(save_dir, f'EN_pred_{unseen_str}_donor_SNPs_{gene_str}_genes.csv')
    preds_list = []
    for i, gene in enumerate(genes_to_eval):
        for func_str in eval_types:
            seed_everything(seed)
            regr = ElasticNetCV(max_iter=2000, random_state=RANDOM_STATE, n_jobs=n_jobs)
            selector = VarianceThreshold()
            X, y = get_X_y(get_gene_df(vcf, gene, train_genotype, train_expr, positions, seq_len), func_str)
            regr.fit(selector.fit_transform(X), y)
            preds_list.append(eval_en(get_gene_df(vcf, gene, unseen_genotype, unseen_expr, positions, seq_len), func_str, regr, selector))
        if i % 25 == 0:  # save results as they come
            pd.concat(preds_list).to_csv(out_path)
            print(f"Gene {i} / {len(genes_to_eval)}")
    pd.concat(preds_list).to_csv(out_path)


EXPRESSION = {}  # open each expression file once; re-opening the same netCDF file can crash


def load_ds(save_dir, split, config, filename=None):
    ds = torch.load(os.path.join(save_dir, filename or f'{split}_ds.pt'), weights_only=False)  # a pickled dataset object, not just weights (torch>=2.6 defaults to True)
    ds = getattr(ds, 'base_dataset', ds)  # valid/test are wrapped in DoubleRCDataset
    if not hasattr(ds, 'gene_expression_df'):  # removed before saving
        path = config[f'{split}_expression_filepath']
        if path not in EXPRESSION:
            EXPRESSION[path] = xr.open_dataarray(path, engine='netcdf4')
        ds.gene_expression_df = EXPRESSION[path]
    return ds


def eval_unseen_genes(vcf, split, train_indivs, save_dir, genes, seq_len, unseen_genotype, unseen_expr, config, majority_af, seed, n_jobs):
    """
    Unseen (valid/test) genes have no synthetic genotypes or expression for train donors. Assign them using the
    validation/test dataset's effect sizes and the common allele frequency, so the EN can be fit on train donors and
    evaluated on the same people, genotypes and expression as the deep learning model
    """
    print(f"Preparing {split} genes")
    seed_everything(seed)
    unseen_ds = load_ds(save_dir, split, config)
    for gene in unseen_ds.effect_allele_freq.keys():
        unseen_ds.effect_allele_freq[gene] = majority_af
    unseen_ds.individuals_in_split = train_indivs
    unseen_ds.define_genotypes()
    unseen_ds.define_expression_vals()  # expression_delta is already defined, so the same effect sizes are used
    train_expr, train_genotype = get_expression_and_genotype(unseen_ds)
    assert sorted(train_genotype.individual.unique()) == train_indivs
    unseen_positions = pd.read_csv(os.path.join(save_dir, f"{split}-dataset_motif-positions.csv"), index_col=0)
    eval_genes(vcf, genes, split, train_genotype, train_expr, unseen_positions, seq_len, split, unseen_genotype, unseen_expr,
               save_dir, ['all', 'only_synthetic'], seed, n_jobs)


GENE_GROUPS = ['train', 'rare-train', 'valid', 'test']


def main():
    parser = argparse.ArgumentParser(description="Fit elastic nets on SNPs for a trained run")
    parser.add_argument("--save_dir", type=str, required=True, help="Results folder of a training run (contains config.csv)")
    parser.add_argument("--vcf", type=str, default=os.path.join(os.environ.get('LAMP_ROOT', ''), "data", "GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz"),
                        help="Indexed SNPs-only GTEx VCF/BCF")
    parser.add_argument("--gene_groups", type=str, nargs='+', default=GENE_GROUPS, choices=GENE_GROUPS)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    save_dir = args.save_dir
    vcf = pysam.VariantFile(args.vcf)
    config = pd.read_csv(os.path.join(save_dir, "config.csv"), index_col=0).iloc[0]
    seed = int(config['seed'])
    seq_len = int(config['seq_len'])
    valid_genes = sorted(literal_eval(config['valid_genes']))
    test_genes = sorted(literal_eval(config['test_genes']))
    seed_everything(seed)

    if config['mode'] == 'personal_genome':
        train_ds = load_ds(save_dir, 'train', config)
        train_genotype = pd.read_csv(os.path.join(save_dir, "train_ds_synthetic_genotype.csv"), index_col=0)
        train_expr = pd.read_csv(os.path.join(save_dir, "train_ds_expression_vals.csv"), index_col=0)
        train_positions = pd.read_csv(os.path.join(save_dir, "train-dataset_motif-positions.csv"), index_col=0)
    else:  # variant runs have no train donors. Use the personal genome train dataset saved for the elastic net (save_pg_train_dataset in train.py)
        train_ds = load_ds(save_dir, 'train', config, filename='train_ds_pg.pt')
        train_expr, train_genotype = get_expression_and_genotype(train_ds)
        train_positions = train_ds.motif_positions

    # rare-variant genes: give train donors synthetic genotypes at the common allele frequency, as if the variant had been seen
    majority_af = train_positions[train_positions['effect_allele_freq'] != 0]['effect_allele_freq'].unique().item()
    rare_var_genes = list(train_positions[train_positions['effect_allele_freq'] == 0]['gene_name'].unique())
    if rare_var_genes:  # not present in variant runs, which leave rare-variant genes out of training. False when no rare vars
        for gene in rare_var_genes:
            assert train_ds.expression_delta[gene] == 0
            del train_ds.genotype_dict[gene]
            del train_ds.expression_delta[gene]
            del train_ds.expression_vals[gene]
            train_ds.effect_allele_freq[gene] = majority_af
        train_ds.define_genotypes()  # re-draws genotypes for every gene; only the rare-variant genes are kept below
        if not pd.isna(config.get('expression_delta')):
            # --expression_delta runs (target r2 = -1): reuse the fixed effect size; the r2 formula would give NaN
            for gene in rare_var_genes:
                train_ds.expression_delta[gene] = float(config['expression_delta'])
        train_ds.define_expression_vals()
        expression, genotypes = get_expression_and_genotype(train_ds)
        train_expr = pd.concat([train_expr[~train_expr['gene'].isin(rare_var_genes)], expression[expression['gene'].isin(rare_var_genes)]])
        train_genotype = pd.concat([train_genotype[~train_genotype['gene'].isin(rare_var_genes)], genotypes[genotypes['gene'].isin(rare_var_genes)]])
    train_indivs = sorted(train_genotype.individual.unique())

    valid_genotype = pd.read_csv(os.path.join(save_dir, "valid_ds_synthetic_genotype.csv"), index_col=0)
    valid_expr = pd.read_csv(os.path.join(save_dir, "valid_ds_expression_vals.csv"), index_col=0)
    test_genotype = pd.read_csv(os.path.join(save_dir, "test_ds_synthetic_genotype.csv"), index_col=0)
    test_expr = pd.read_csv(os.path.join(save_dir, "test_ds_expression_vals.csv"), index_col=0)
    common = dict(save_dir=save_dir, seed=seed, n_jobs=args.n_jobs)

    if 'rare-train' in args.gene_groups and rare_var_genes:
        for unseen_str, genotype, expr in [('valid', valid_genotype, valid_expr), ('test', test_genotype, test_expr)]:
            eval_genes(vcf, rare_var_genes, 'rare-train', train_genotype, train_expr, train_positions, seq_len, unseen_str, genotype, expr,
                       eval_types=['all', 'only_synthetic'], **common)
    if 'valid' in args.gene_groups:
        eval_unseen_genes(vcf, 'valid', train_indivs, save_dir, valid_genes, seq_len, valid_genotype, valid_expr, config, majority_af, seed, args.n_jobs)
    if 'test' in args.gene_groups:
        eval_unseen_genes(vcf, 'test', train_indivs, save_dir, test_genes, seq_len, test_genotype, test_expr, config, majority_af, seed, args.n_jobs)
    if 'train' in args.gene_groups:
        remaining_train_genes = sorted([x for x in train_positions['gene_name'].unique() if x not in rare_var_genes])
        for unseen_str, genotype, expr in [('valid', valid_genotype, valid_expr), ('test', test_genotype, test_expr)]:
            eval_genes(vcf, remaining_train_genes, 'train', train_genotype, train_expr, train_positions, seq_len, unseen_str, genotype, expr,
                       eval_types=['all', 'only_synthetic', 'only_natural'], **common)


if __name__ == '__main__':
    main()
