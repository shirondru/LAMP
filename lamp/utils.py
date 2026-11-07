import os

import pandas as pd
import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from models import LitModelHeadAdapterWrapper_MSE, TwinNet, MetricLogger


def load_config(config_path):
    """
    Load a YAML config file and substitute ${LAMP_ROOT} with the LAMP_ROOT environment variable (set in paths.config)
    """
    lamp_root = os.environ.get('LAMP_ROOT')
    if lamp_root is None:
        raise EnvironmentError("LAMP_ROOT is not set. Run `source paths.config` first.")
    with open(config_path, 'r') as f:
        content = f.read().replace('${LAMP_ROOT}', lamp_root)
    return yaml.safe_load(content)


def parse_gene_files(filepath):
    """
    txt file with one gene name per row
    """
    with open(filepath, 'r') as file:
        return [gene.strip() for gene in file]


def get_genes(train_gene_path, valid_gene_path, test_gene_path):
    return parse_gene_files(train_gene_path), parse_gene_files(valid_gene_path), parse_gene_files(test_gene_path)


def ensure_no_gene_overlap(train_genes, val_genes, test_genes):
    assert len(set(train_genes) & set(val_genes)) == 0, "There is overlap between train and valid genes"
    assert len(set(train_genes) & set(test_genes)) == 0, "There is overlap between train and test genes"
    assert len(set(val_genes) & set(test_genes)) == 0, "There is overlap between valid and test genes"
    assert len(train_genes) > 0, "You have no genes to train on!"


def ensure_no_donor_overlap(train_ds, val_ds, test_ds):
    train_set, val_set, test_set = set(train_ds.individuals_in_split), set(val_ds.individuals_in_split), set(test_ds.individuals_in_split)
    assert len(train_set & val_set) == 0 and len(train_set & test_set) == 0 and len(val_set & test_set) == 0


def load_callbacks(config, train_genes, valid_genes, test_genes):
    checkpoint_dir = os.path.join(config.save_dir, 'checkpoints_after_resuming' if hasattr(config, 'train_ckpt_resume_path') else 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, save_top_k=1, monitor=config.monitor, mode=config.monitor_mode, verbose=True)
    early_stopper = EarlyStopping(monitor=config.monitor, mode=config.monitor_mode, min_delta=0, patience=int(config.patience), verbose=True)
    metric_logger = MetricLogger(train_genes, valid_genes, test_genes, config.save_dir)
    return metric_logger, early_stopper, checkpoint_callback


def define_grad_acc(config):
    return int(config.gradient_accumulation)


def load_trainer(config, train_genes, valid_genes, test_genes):
    metric_logger, early_stopper, checkpoint_callback = load_callbacks(config, train_genes, valid_genes, test_genes)
    return Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        accumulate_grad_batches=define_grad_acc(config),
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[checkpoint_callback, metric_logger, early_stopper],
        logger=WandbLogger(),
        num_sanity_val_steps=0,  # no validation before training
        log_every_n_steps=1,
        check_val_every_n_epoch=config.valid_metrics_save_freq
    )


def load_trainer_multi_gpu(config, train_genes, valid_genes, test_genes):
    metric_logger, early_stopper, checkpoint_callback = load_callbacks(config, train_genes, valid_genes, test_genes)
    return Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        accumulate_grad_batches=define_grad_acc(config),
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[checkpoint_callback, metric_logger, early_stopper],
        logger=WandbLogger(),
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=int(config.n_gpus),
        check_val_every_n_epoch=config.valid_metrics_save_freq,
        strategy='ddp_find_unused_parameters_true'
    )


def delete_checkpoint(trainer):
    for checkpoint_callback in [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]:
        os.remove(checkpoint_callback.best_model_path)


def save_config_file(config):
    pd.DataFrame({k: str(v) for k, v in config.items()}, index=[0]).to_csv(os.path.join(config.save_dir, "config.csv"))


def load_model(config, tissues_to_train, train_ds):
    model_class = TwinNet if config.mode == 'variant_effect_twinnet' else LitModelHeadAdapterWrapper_MSE
    model = model_class(tissues_to_train, float(config.learning_rate), train_ds)
    config.update({'model_type': type(model).__name__})
    return model


def log_expression_filepath(config, train_ds, valid_ds, test_ds):
    config.update({'train_expression_filepath': train_ds.expression_filepath,
                   'valid_expression_filepath': valid_ds.base_dataset.expression_filepath,
                   'test_expression_filepath': test_ds.base_dataset.expression_filepath})


def parse_tissues(config):
    return config.tissues_to_train.split(',')  # comma-delimited string in the config file


def save_datasets(config, train_ds, valid_ds, test_ds):
    """
    Save datasets for later use. Open FASTA handles and the expression array are removed first so the objects can be pickled,
    then the FASTAs are re-opened for training.
    """
    if config.skip_ds_save:
        return
    for ds in [train_ds, valid_ds.base_dataset, test_ds.base_dataset]:
        ds._close_fastas()
        if hasattr(ds, 'gene_expression_df'):
            del ds.gene_expression_df
    torch.save(train_ds, os.path.join(config.save_dir, "train_ds.pt"))
    torch.save(valid_ds, os.path.join(config.save_dir, "valid_ds.pt"))
    torch.save(test_ds, os.path.join(config.save_dir, "test_ds.pt"))
    for ds in [train_ds, valid_ds.base_dataset, test_ds.base_dataset]:
        ds.fastas = ds._cache_fastas()


def save_motif_positions(train_ds, valid_ds, test_ds, config):
    train_ds.motif_positions.to_csv(os.path.join(config.save_dir, "train-dataset_motif-positions.csv"))
    valid_ds.base_dataset.motif_positions.to_csv(os.path.join(config.save_dir, "valid-dataset_motif-positions.csv"))
    test_ds.base_dataset.motif_positions.to_csv(os.path.join(config.save_dir, "test-dataset_motif-positions.csv"))


def update_learning_rate(config):
    """
    Scale the learning rate linearly by the number of GPUs, to account for the larger effective batch size
    """
    config.update({'old_learning_rate': config.learning_rate})
    config.update({'learning_rate': float(config.learning_rate) * int(config.n_gpus)}, allow_val_change=True)
    return config
