import argparse
import os
import warnings
from functools import partial

import lightning.pytorch as pl
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from dataset import (PersonalGenomeSimulatorPropEffectSize, PersonalGenomeSlideMotifWithFlank,
                     TwinDifferencePersonalGenomeSimulatorPropEffectSize, RefGenomeVariantEffectDataset,
                     DoubleRCDataset, CustomDataModule, DEFAULT_BACKGROUND_SNVS)
from utils import (load_config, get_genes, ensure_no_gene_overlap, ensure_no_donor_overlap, load_trainer,
                   load_trainer_multi_gpu, load_model, delete_checkpoint, save_config_file, log_expression_filepath,
                   parse_tissues, save_datasets, save_motif_positions, update_learning_rate)

warnings.filterwarnings('ignore', '.*does not have many workers.*')

MODES = ['personal_genome', 'variant_effect', 'variant_effect_twinnet']
VARIANT_MODES = ['variant_effect', 'variant_effect_twinnet']


def save_ds_meta(ds, name, save_dir):
    df = pd.DataFrame.from_dict(ds.expression_vals, orient='index')
    df = df.reset_index().melt(id_vars='index', var_name='individual', value_name='expression').rename(columns={'index': 'gene'})
    df.to_csv(os.path.join(save_dir, f"{name}_expression_vals.csv"))

    df = pd.DataFrame.from_dict(ds.genotype_dict, orient='index')
    df = df.reset_index().melt(id_vars='index', var_name='individual', value_name='genotype').rename(columns={'index': 'gene'})
    df.to_csv(os.path.join(save_dir, f"{name}_synthetic_genotype.csv"))

    df = pd.DataFrame.from_dict(ds.expression_delta, orient='index').reset_index()
    df = df.rename(columns={'index': 'gene', 0: 'expression_delta'})
    df.to_csv(os.path.join(save_dir, f"{name}_expression_delta.csv"))


def get_n_per_gene(config):
    """
    One batch contains only one gene
    """
    return config.train_batch_size


def get_train_ds_kwargs(config):
    """
    Extra arguments passed only to the train dataset
    """
    if config.mode in VARIANT_MODES:
        # Background SNVs are on by default. Without them every training example has a large effect (the synthetic variants),
        # so the model overfits and predicts large effect sizes for all variants. Use --background_snvs none to omit them.
        return {'background_snvs': config.background_snvs,
                'background_sampling': config.background_sampling,
                'max_background_per_variant': config.max_background_per_variant}
    if config.motif_flank is not None:
        return {'flank': config.motif_flank, 'min_tss_dist': config.min_tss_dist}
    return {}


def get_prop_effect_size_ds(config, train_genes, valid_genes, test_genes, motif_positions, ds):
    """
    Build train/valid/test datasets.

    Rare-variant genes (effect_allele_freq == 0 in motif_positions) are train genes whose synthetic variant is not seen
    during training. Their delta is defined in the validation set, and in valid/test every gene's synthetic variant
    gets the common allele frequency so all variant effects can be evaluated in unseen people.
    Deltas defined in an earlier split are reused in later splits so each variant's effect size stays the same.
    """
    print("Also simulating rare variants for train genes")
    train_ds, valid_ds, test_ds = ds
    seq_len = config.seq_len
    fold = config.fold
    position_idx = config.position_idx
    n_per_gene = get_n_per_gene(config)
    tissues_to_train = parse_tissues(config)
    r2_dict = {}
    for gene in train_genes:
        r2_dict[gene] = float(config.train_target_r2)
    for gene in valid_genes:
        r2_dict[gene] = float(config.valid_target_r2)
    for gene in test_genes:
        r2_dict[gene] = float(config.test_target_r2)

    effect_allele_freq_dict = {}
    for gene in train_genes + valid_genes + test_genes:
        effect_allele_freq_dict[gene] = motif_positions[motif_positions['gene_name'] == gene]['effect_allele_freq'].unique().item()

    if config.get('expression_delta') is not None:
        assert config['train_target_r2'] == -1 == config['valid_target_r2'] == config['test_target_r2']  # -1: delta is set directly, not by r2
        expression_delta = {gene: config['expression_delta'] for gene in effect_allele_freq_dict.keys()}
    else:
        expression_delta = {}
    train_ds = train_ds(seq_len, fold, 'train', train_genes, r2_dict, position_idx, effect_allele_freq_dict, n_per_gene,
                        tissues_to_train, motif_positions, expression_delta=expression_delta, **get_train_ds_kwargs(config))

    # rare-variant genes have a delta of 0 in the train set. Remove it so it's re-defined in the validation set
    genes_w_rare_var = list(motif_positions[motif_positions['effect_allele_freq'] == 0]['gene_name'].unique())
    train_gene_expression_delta = train_ds.expression_delta.copy()
    for gene in genes_w_rare_var:
        if gene in train_ds.genes_in_dataset:
            del train_gene_expression_delta[gene]

    train_effect_allele_freq_dict = effect_allele_freq_dict
    # in valid/test, every synthetic variant gets the common allele frequency
    observed_allele_freq = motif_positions[motif_positions['effect_allele_freq'] != 0]['effect_allele_freq'].unique().item()
    effect_allele_freq_dict = {gene: observed_allele_freq for gene in train_genes + valid_genes + test_genes}
    if config.get('expression_delta') is not None:
        for gene in genes_w_rare_var + valid_genes:
            train_gene_expression_delta[gene] = expression_delta[gene]

    valid_ds = valid_ds(seq_len, fold, 'valid', valid_genes + train_genes, r2_dict, position_idx, effect_allele_freq_dict, -1,
                        tissues_to_train, motif_positions, expression_delta=train_gene_expression_delta)

    train_val_expression_delta = valid_ds.expression_delta.copy()
    if config.get('expression_delta') is not None:
        for gene in genes_w_rare_var + valid_genes:
            train_val_expression_delta[gene] = expression_delta[gene]
    test_ds = test_ds(seq_len, fold, 'test', test_genes + valid_genes + train_genes, r2_dict, position_idx, effect_allele_freq_dict, -1,
                      tissues_to_train, motif_positions, expression_delta=train_val_expression_delta)
    if config.mode not in VARIANT_MODES:  # variant training has no donors
        ensure_no_donor_overlap(train_ds, valid_ds, test_ds)

    os.makedirs(config.save_dir, exist_ok=True)
    if config.mode in VARIANT_MODES and not config.skip_ds_save:
        # built last so it doesn't change the random genotypes of the valid/test datasets
        save_pg_train_dataset(config, train_ds, r2_dict, train_effect_allele_freq_dict, n_per_gene, tissues_to_train, motif_positions)
    save_ds_meta(train_ds, "train_ds", config.save_dir)
    save_ds_meta(valid_ds, "valid_ds", config.save_dir)
    save_ds_meta(test_ds, "test_ds", config.save_dir)
    config.update({'train_ds_type': type(train_ds).__name__,
                   'valid_ds_type': type(valid_ds).__name__,
                   'test_ds_type': type(test_ds).__name__})
    valid_ds = DoubleRCDataset(valid_ds)  # evaluate every sequence in both orientations
    test_ds = DoubleRCDataset(test_ds)
    train_ds, valid_ds, test_ds = update_ds_hook(config, train_ds, valid_ds, test_ds)
    config.update({'rc_during_validation': 'both'})
    save_datasets(config, train_ds, valid_ds, test_ds)
    return train_ds, valid_ds, test_ds


def save_pg_train_dataset(config, train_ds, r2_dict, effect_allele_freq_dict, n_per_gene, tissues_to_train, motif_positions):
    """
    Variant modes: save a personal genome train dataset (train_ds_pg.pt) with the same genes and synthetic variant effect
    sizes as the variant train dataset, giving train donors synthetic genotypes. Only used by elastic_net.py.
    """
    pg_ds = PersonalGenomeSimulatorPropEffectSize(config.seq_len, config.fold, 'train', train_ds.genes_in_dataset, r2_dict, config.position_idx,
                                                  effect_allele_freq_dict, n_per_gene, tissues_to_train, motif_positions,
                                                  expression_delta=dict(train_ds.expression_delta))
    pg_ds._close_fastas()
    del pg_ds.gene_expression_df
    torch.save(pg_ds, os.path.join(config.save_dir, "train_ds_pg.pt"))


def choose_ds(config):
    """
    Returns the (train, valid, test) dataset classes for config.mode. Override to use your own datasets.
    """
    if config.mode == 'personal_genome':
        train_ds = PersonalGenomeSlideMotifWithFlank if config.motif_flank is not None else PersonalGenomeSimulatorPropEffectSize
        return train_ds, PersonalGenomeSimulatorPropEffectSize, PersonalGenomeSimulatorPropEffectSize
    elif config.mode == 'variant_effect':
        return RefGenomeVariantEffectDataset, PersonalGenomeSimulatorPropEffectSize, PersonalGenomeSimulatorPropEffectSize
    elif config.mode == 'variant_effect_twinnet':
        return (partial(RefGenomeVariantEffectDataset, paired=True),
                TwinDifferencePersonalGenomeSimulatorPropEffectSize,
                TwinDifferencePersonalGenomeSimulatorPropEffectSize)


def update_ds_hook(config, train_ds, valid_ds, test_ds, **kwargs):
    """
    Extension point: override to modify datasets after they are built. Does nothing by default.
    """
    return train_ds, valid_ds, test_ds


def update_args_for_config_hook(parser):
    """
    Extension point: override to add command line arguments. Does nothing by default.
    """
    pass


def update_config_hook(config, args):
    """
    Extension point: override to modify the config using parsed arguments. Does nothing by default.
    """
    pass


def train(config, train_genes, valid_genes, test_genes, keep_checkpoint, motif_positions):
    ensure_no_gene_overlap(train_genes, valid_genes, test_genes)
    tissues_to_train = parse_tissues(config)
    train_ds, valid_ds, test_ds = get_prop_effect_size_ds(config, train_genes, valid_genes, test_genes, motif_positions, choose_ds(config))
    log_expression_filepath(config, train_ds, valid_ds, test_ds)

    if config.n_gpus > 1:
        print("Using MultiGPU Train Function")
        data_module = CustomDataModule(train_ds, valid_ds, test_ds, int(config.train_batch_size), shuffle=config.shuffle)
        config = update_learning_rate(config)
        trainer = load_trainer_multi_gpu(config, train_genes, valid_genes, test_genes)
    else:
        trainer = load_trainer(config, train_genes, valid_genes, test_genes)
    model = load_model(config, tissues_to_train, train_ds)
    save_config_file(config)
    save_motif_positions(train_ds, valid_ds, test_ds, config)
    ckpt_path = config.get('train_ckpt_resume_path')  # resume training, e.g. if a SLURM job ran out of time

    if config.n_gpus > 1:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)
        trainer.test(model, datamodule=data_module, ckpt_path='best')
    else:
        trainer.fit(model=model,
                    train_dataloaders=DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=config.shuffle, num_workers=8),
                    val_dataloaders=DataLoader(valid_ds, batch_size=config.train_batch_size, num_workers=8),
                    ckpt_path=ckpt_path)
        trainer.test(model, DataLoader(test_ds, batch_size=config.train_batch_size, num_workers=8), ckpt_path='best')
    if not keep_checkpoint:
        delete_checkpoint(trainer)


def harmonize_genes_and_positions(train_genes, valid_genes, test_genes, motif_positions):
    train_genes = sorted(list(set(train_genes) & set(motif_positions['gene_name'])))
    valid_genes = sorted(list(set(valid_genes) & set(motif_positions['gene_name'])))
    test_genes = sorted(list(set(test_genes) & set(motif_positions['gene_name'])))
    motif_positions = motif_positions[motif_positions['gene_name'].isin(train_genes + valid_genes + test_genes)]
    return train_genes, valid_genes, test_genes, motif_positions


def main():
    parser = argparse.ArgumentParser(description="Train Enformer on simulated variant effects")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=MODES, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--motif_positions", type=str, required=True, help="CSV of motif positions (one synthetic variant per gene) for train, valid, and test genes")
    parser.add_argument("--background_snvs", type=str,
                        default=DEFAULT_BACKGROUND_SNVS,
                        help="Variant modes: CSV of background SNVs (from scripts/background_snvs/make_background_snvs.ipynb), used by default. "
                             "'none' trains on synthetic variants only, which causes the model to overfit and predict large effect sizes for all variants")
    parser.add_argument("--background_sampling", type=str, choices=['equal', 'capped'], default='equal', help="Variant modes: 'equal' = 1:1 synthetic:background examples; 'capped' = at most --max_background_per_variant background SNVs per gene")
    parser.add_argument("--max_background_per_variant", type=int, help="Variant modes with --background_sampling capped")
    parser.add_argument("--motif_flank", type=int, help="personal_genome mode: slide the causal motif plus this many flanking nt during training")
    parser.add_argument("--min_tss_dist", type=int, help="personal_genome mode with --motif_flank: the slid motif + flank stays this many nt from the TSS")
    parser.add_argument("--keep_checkpoint", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0, help="1: single GPU, don't save datasets, log to a `_test` W&B project")
    parser.add_argument("--n_gpus", type=int)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--target_r2", type=float, help="Overrides train/valid/test target r2 in the config")
    parser.add_argument("--effect_allele_freq", type=float)
    parser.add_argument("--expression_delta", type=float, help="Set the synthetic effect size directly. Requires target r2 of -1")
    parser.add_argument("--position_idx", type=int, help="Position within the motif of the synthetic variant")
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--metadata_path", type=str, help="Recorded in the config for tracking")
    parser.add_argument("--ckpt_path", type=str, help="Resume training from this checkpoint")
    parser.add_argument("--train_gene_path", type=str)
    parser.add_argument("--valid_gene_path", type=str)
    parser.add_argument("--test_gene_path", type=str)
    update_args_for_config_hook(parser)
    args = parser.parse_args()

    assert args.keep_checkpoint in [0, 1]
    assert args.debug in [0, 1]
    if args.mode in VARIANT_MODES:
        assert args.background_sampling == 'equal' or args.max_background_per_variant is not None, "--background_sampling capped requires --max_background_per_variant"
    if args.motif_flank is not None:
        assert args.mode == 'personal_genome' and args.min_tss_dist is not None, "--motif_flank is for personal_genome mode and requires --min_tss_dist"

    config = load_config(args.config_path)
    config['mode'] = args.mode
    config['config_path'] = args.config_path
    config['PreDefinedMotifPositionsPath'] = args.motif_positions
    config['shuffle'] = bool(args.shuffle)
    for arg in ['background_snvs', 'background_sampling', 'max_background_per_variant', 'motif_flank', 'min_tss_dist']:
        config[arg] = getattr(args, arg)
    # optional overrides of config values
    for arg in ['n_gpus', 'seq_len', 'patience', 'max_epochs', 'experiment_name', 'project', 'position_idx',
                'train_gene_path', 'valid_gene_path', 'test_gene_path', 'metadata_path', 'learning_rate']:
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    if args.target_r2 is not None:
        config['train_target_r2'] = config['valid_target_r2'] = config['test_target_r2'] = args.target_r2
    if args.effect_allele_freq is not None:
        config['train_effect_allele_freq'] = args.effect_allele_freq
    if args.expression_delta is not None:
        assert config['train_target_r2'] == -1 == config['valid_target_r2'] == config['test_target_r2']
        config['expression_delta'] = args.expression_delta
    update_config_hook(config, args)

    if args.debug:
        suffix = '_test'
        config['n_gpus'] = 1
        config['skip_ds_save'] = True
    else:
        suffix = ''
        config['skip_ds_save'] = False

    fold, seed = args.fold, args.seed
    if args.ckpt_path is not None:
        config['resumed'] = True
        config['train_ckpt_resume_path'] = args.ckpt_path
        experiment_name = args.ckpt_path.split('/')[-6]  # {save_dir}/{experiment_name}/Fold-{fold}/Seed-{seed}/{run_id}/checkpoints/{ckpt}
        wandb.init(project=config['project'] + suffix, name=f"RESUME-{experiment_name}_Fold-{fold}_Seed-{seed}", group=experiment_name, config=config, resume='allow')
    else:
        experiment_name = config['experiment_name'].replace('::', '--')
        wandb.init(project=config['project'] + suffix, name=f"{experiment_name}_Fold-{fold}_Seed-{seed}", group=experiment_name, config=config)

    motif_positions = pd.read_csv(args.motif_positions)
    train_genes, valid_genes, test_genes = get_genes(wandb.config.train_gene_path, wandb.config.valid_gene_path, wandb.config.test_gene_path)
    train_genes, valid_genes, test_genes, motif_positions = harmonize_genes_and_positions(train_genes, valid_genes, test_genes, motif_positions)
    wandb.config.update({'fold': fold, 'seed': seed, 'train_genes': train_genes, 'valid_genes': valid_genes, 'test_genes': test_genes})
    if args.tag is not None:
        wandb.run.tags = wandb.run.tags + (args.tag,)
    save_dir = f"{config['save_dir']}/{experiment_name}/Fold-{fold}/Seed-{seed}/{wandb.run.id}"
    if args.ckpt_path is not None:
        save_dir += "/resume_from_checkpoint"
    wandb.config.update({'save_dir': save_dir}, allow_val_change=True)
    pl.seed_everything(int(wandb.config.seed), workers=True)
    torch.use_deterministic_algorithms(True)
    train(wandb.config, train_genes, valid_genes, test_genes, bool(args.keep_checkpoint), motif_positions)
    wandb.finish()


if __name__ == '__main__':
    main()
