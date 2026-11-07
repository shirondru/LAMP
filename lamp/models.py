import os
import pickle

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper, freeze_batchnorms_
from torchmetrics.regression import PearsonCorrCoef, R2Score


def load_pretrained_enformer():
    try:
        enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=-1)  # no cropping, for shorter sequences
        print("Loaded Enformer using URL")
    except Exception as e:
        # offline fallback: a local copy saved with Enformer.save_pretrained
        local_dir = os.path.join(os.environ.get('LAMP_ROOT', ''), "data", "Enformer-from_pretrained")
        print(f"Online load failed ({e}). Loading from {local_dir}")
        enformer = Enformer.from_pretrained(local_dir, target_length=-1)
    return enformer


def masked_mse_loss(y_hat, y):
    """
    MSE ignoring NaN targets (e.g., when someone is missing data from a tissue)
    """
    mask = torch.isnan(y)
    if all(torch.isnan(y)):  # no gradient from batches that are all NaN
        return torch.mean(y_hat * 0.0)
    return torch.mean((y_hat[~mask] - y[~mask]) ** 2)


class MultiTissueLitModel(pl.LightningModule):
    def __init__(self, tissues_to_train, learning_rate, train_dataset):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset"])
        self.pred_dict = {}
        self.target_dict = {}
        self.donor_dict = {}
        self.rank_dict = {}
        self.rc_dict = {}
        self.tissues_to_train = tissues_to_train
        self.learning_rate = learning_rate
        self.train_dataset = train_dataset  # for shuffling after each train epoch
        assert tissues_to_train == train_dataset.tissues_to_train, "if order isn't the same, there will be a mismatch between labeled tissue and true tissue"

    def loss_fn(self, y_hat, y):
        return masked_mse_loss(y_hat, y)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]
        y_hat = self(x)
        return y_hat[:, y_hat.shape[1] // 2, :]  # keep the prediction at the center of the sequence (TSS)

    def training_step(self, batch, batch_idx):
        x, y, genes, donor, rc, _ = batch
        y_hat = self.predict_step(batch, batch_idx)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, batch_size=y.shape[0], on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self._eval_step(batch, batch_idx)

    def _eval_step(self, batch, batch_idx):
        """
        Store predictions for each person. Metrics are computed across all people at the end of the epoch by MetricLogger
        """
        x, y, gene_name, donor, rc, _ = batch
        y_hat = self.predict_step(batch, batch_idx)
        rank = self.trainer.global_rank
        for i in range(len(gene_name)):
            self.save_eval_results(y_hat[i:i + 1], y[i:i + 1], donor[i], rank, gene_name[i], rc[i])

    def on_train_epoch_end(self):
        self.train_dataset.shuffle_and_define_epoch()  # reshuffle genes/people on the main process

    def configure_optimizers(self):
        return {'optimizer': torch.optim.AdamW(self.parameters(), lr=self.learning_rate)}

    def save_eval_results(self, y_hat, y, donor, rank, gene_name, rc):
        if gene_name not in self.pred_dict:
            for d in [self.pred_dict, self.target_dict, self.donor_dict, self.rank_dict, self.rc_dict]:
                d[gene_name] = {tissue: [] for tissue in self.tissues_to_train}

        tissue_indices_without_data = torch.isnan(y).nonzero(as_tuple=True)[1]
        for tissue_idx, tissue in enumerate(self.tissues_to_train):
            if tissue_idx not in tissue_indices_without_data:
                self.pred_dict[gene_name][tissue].append(y_hat[:, tissue_idx])
                self.target_dict[gene_name][tissue].append(y[:, tissue_idx])
                self.donor_dict[gene_name][tissue].append(donor)
                self.rank_dict[gene_name][tissue].append(rank)
                self.rc_dict[gene_name][tissue].append(rc)


class LitModelHeadAdapterWrapper_MSE(MultiTissueLitModel):
    """
    Pre-trained Enformer with a new linear output head (one output per tissue), trained with MSE
    """
    def __init__(self, tissues_to_train, learning_rate, train_dataset):
        super().__init__(tissues_to_train, learning_rate, train_dataset)
        self.model = HeadAdapterWrapper(
            enformer=load_pretrained_enformer(),
            num_tracks=len(self.tissues_to_train),
            post_transformer_embed=False,  # important to keep False
            output_activation=nn.Identity()
        )
        # remove unused layers and freeze batchnorm during init so DDP works
        del self.model.enformer._heads
        freeze_batchnorms_(self)
        print("Deleted original output heads and froze batchnorm layers")

    def forward(self, x):
        return self.model(x, freeze_enformer=False)


class TwinNet(MultiTissueLitModel):
    """
    Input: (batch, 2, seq_len, 4) where x[:, 0] is the alt (effect allele) sequence and x[:, 1] is the ref (other allele) sequence.
    Output: mean over positions and channels of trunk(alt) - trunk(ref). No output activation.
    """
    def __init__(self, tissues_to_train, learning_rate, train_dataset):
        super().__init__(tissues_to_train, learning_rate, train_dataset)
        self.enformer = load_pretrained_enformer()

    def forward(self, x):
        assert x.ndim == 4
        output = self.enformer._trunk(x[:, 0, ...]) - self.enformer._trunk(x[:, 1, ...])  # alt - ref
        return torch.mean(output, axis=(1, 2)).unsqueeze(1)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


class MetricLogger(pl.Callback):
    """
    Saves raw predictions and per-gene cross-individual metrics (Pearson R of z-scored values, R2, MSE) from validation/test loops
    """
    def __init__(self, genes_for_training, genes_for_valid, genes_for_test, save_dir, output_type='tissue'):
        super().__init__()
        self.genes_for_training = genes_for_training
        self.genes_for_valid = genes_for_valid
        self.genes_for_test = genes_for_test
        self.save_dir = save_dir
        self.output_type = output_type
        self.reset_metrics_history()

    def reset_metrics_history(self):
        self.metrics_history = {'epoch': [], 'pearsonr': [], 'r2': [], 'gene_name': [], self.output_type: [], 'per_gene_loss': [], 'gene_split': [], 'donor_split': [], 'sample_size': []}

    def _compute_loss(self, pl_module, preds_df):
        self.loss_vals = {}
        self.total_loss = {}
        self.per_output_type_loss = {}
        preds_df = preds_df[['gene', self.output_type, 'donor', 'y_pred', 'y_true', 'gene_split']]
        preds_df = preds_df.loc[~((preds_df['gene'].isnull()) & (preds_df['donor'].isnull()))]
        for gene in preds_df['gene'].unique():
            df = preds_df[preds_df['gene'] == gene].groupby(['gene', self.output_type, 'donor'])[['y_pred', 'y_true']].mean().reset_index()  # average over +/- strands
            y = df.pivot_table(index='donor', values='y_true', columns=self.output_type).values
            y_hat = df.pivot_table(index='donor', values='y_pred', columns=self.output_type).values
            self.loss_vals[gene] = masked_mse_loss(torch.from_numpy(y_hat.astype('float')), torch.from_numpy(y.astype('float'))).item()

        for gene_split in preds_df['gene_split'].unique():
            self.per_output_type_loss[gene_split] = {}
            df = preds_df[preds_df['gene_split'] == gene_split].groupby(['gene', self.output_type, 'donor'])[['y_pred', 'y_true']].mean().reset_index()
            y = df.pivot_table(index=['donor', 'gene'], values='y_true', columns=self.output_type)
            y_hat = df.pivot_table(index=['donor', 'gene'], values='y_pred', columns=self.output_type)
            self.total_loss[gene_split] = masked_mse_loss(torch.from_numpy(y_hat.values.astype('float')), torch.from_numpy(y.values.astype('float'))).item()
            for output_type in preds_df[self.output_type].unique():
                self.per_output_type_loss[gene_split][output_type] = masked_mse_loss(
                    torch.from_numpy(y_hat[[output_type]].values.astype('float')), torch.from_numpy(y[[output_type]].values.astype('float'))).item()

    def _dict_to_long_df(self, d, value_name, to_float=False):
        df = pd.DataFrame(d).reset_index(names=self.output_type).melt(id_vars=[self.output_type], var_name='gene', value_name=value_name)
        if to_float:
            df[value_name] = df[value_name].apply(lambda vals: [x.item() for x in vals])
        return df.explode(value_name)

    def log_predictions(self, trainer, pl_module, donor_split):
        pred_df = self._dict_to_long_df(pl_module.pred_dict, 'y_pred', to_float=True)
        target_df = self._dict_to_long_df(pl_module.target_dict, 'y_true', to_float=True)
        donor_df = self._dict_to_long_df(pl_module.donor_dict, 'donor')
        rank_df = self._dict_to_long_df(pl_module.rank_dict, 'rank')
        rc_df = self._dict_to_long_df(pl_module.rc_dict, 'used_rc', to_float=True)

        df = pd.concat([pred_df.reset_index(), target_df[['y_true']].reset_index(), donor_df[['donor']].reset_index(), rank_df[['rank']].reset_index(), rc_df[['used_rc']].reset_index()], axis=1)
        df['end_of_epoch'] = self.epoch
        df.loc[df['gene'].isin(self.genes_for_training), 'gene_split'] = 'train'
        df.loc[df['gene'].isin(self.genes_for_valid), 'gene_split'] = 'valid'
        df.loc[df['gene'].isin(self.genes_for_test), 'gene_split'] = 'test'
        self._compute_loss(pl_module, df)
        global_step = pl_module.global_step
        df['global_step'] = global_step
        df.to_csv(os.path.join(self.save_dir, f"Prediction_Results_{self.epoch}_Step-{global_step}_in_{donor_split}_donors.csv"), index=False)

    def organize_cross_indiv_vals(self, pl_module, gene_name, tissue, pearson, r2_score):
        all_predictions = torch.cat(pl_module.pred_dict[gene_name][tissue]).view(-1, 2)
        all_targets = torch.cat(pl_module.target_dict[gene_name][tissue]).view(-1, 2)
        assert all(all_targets.diff(dim=1) == 0), "Forward and RC from same individual are not consecutive"
        # average over forward and RC
        all_predictions = all_predictions.mean(axis=1)
        all_targets = all_targets.mean(axis=1)
        # z-score before Pearson R, which can be unstable when predictions have low variance
        all_predictions_z = (all_predictions - all_predictions.mean()) / all_predictions.std()
        all_targets_z = (all_targets - all_targets.mean()) / all_targets.std()
        self.metrics_history['pearsonr'].append(pearson(all_predictions_z, all_targets_z).detach().cpu().numpy().item())
        self.metrics_history['r2'].append(r2_score(all_predictions, all_targets).detach().cpu().numpy().item())

    def add_to_metrics_history(self, gene_name, tissue, pl_module, gene_split, pearson, r2_score, donor_split):
        sample_size = len(pl_module.target_dict[gene_name][tissue])
        if sample_size > 1:
            self.organize_cross_indiv_vals(pl_module, gene_name, tissue, pearson, r2_score)
            self.metrics_history[self.output_type].append(tissue)
            self.metrics_history['gene_name'].append(gene_name)
            self.metrics_history['epoch'].append(self.epoch)
            self.metrics_history['per_gene_loss'].append(self.loss_vals[gene_name])
            self.metrics_history['gene_split'].append(gene_split)
            self.metrics_history['donor_split'].append(donor_split)
            self.metrics_history['sample_size'].append(sample_size)

    def log_per_gene_per_tissue_metrics(self, trainer, pl_module, donor_split):
        df = pd.DataFrame(self.metrics_history)
        df = df[df['donor_split'] == donor_split].copy()
        df[['pearsonr', 'r2', 'per_gene_loss']] = df[['pearsonr', 'r2', 'per_gene_loss']].astype(float)
        global_step = pl_module.global_step
        df['global_step'] = global_step
        df.to_csv(os.path.join(self.save_dir, f"CrossIndivMetrics_{donor_split}_donors_Epoch{self.epoch}_Step-{global_step}.csv"), index=False)

        grouped = df.groupby(['gene_split', 'donor_split']).agg(
            mean_pearsonr=('pearsonr', 'mean'),
            mean_r2=('r2', 'mean'),
            mean_loss=('per_gene_loss', 'mean')
        ).reset_index()
        results_dict = {}
        for _, row in grouped.iterrows():
            suffix = f"across_{row['gene_split']}_genes_across_{row['donor_split']}_donors"
            results_dict[f"mean_pearsonrz_{suffix}"] = row['mean_pearsonr']
            results_dict[f"mean_r2_{suffix}"] = row['mean_r2']
            results_dict[f"mean_MSEloss_{suffix}"] = row['mean_loss']
        for gene_split, total_loss in self.total_loss.items():
            results_dict[f"mean_global_MSEloss_across_{gene_split}_genes_across_{donor_split}_donors"] = total_loss
            for output_type, output_type_loss in self.per_output_type_loss[gene_split].items():
                results_dict[f"mean_{output_type}_MSEloss_across_{gene_split}_genes_across_{donor_split}_donors"] = output_type_loss
        pl_module.log_dict(results_dict, sync_dist=trainer.num_devices > 1)
        with open(os.path.join(self.save_dir, f"ResultsDict_Epoch{self.epoch}.pickle"), 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def add_all_genes_to_metrics_history(self, pl_module, donor_split):
        pearson = PearsonCorrCoef(num_outputs=1).to(pl_module.device)
        r2_score = R2Score().to(pl_module.device)
        for gene_name in pl_module.pred_dict.keys():
            if gene_name in self.genes_for_training:
                gene_split = 'train'
            elif gene_name in self.genes_for_valid:
                gene_split = 'valid'
            elif gene_name in self.genes_for_test:
                gene_split = 'test'
            else:
                raise Exception(f"Gene {gene_name} not in desired train set, valid set, or test set")
            for tissue in pl_module.tissues_to_train:
                self.add_to_metrics_history(gene_name, tissue, pl_module, gene_split, pearson, r2_score, donor_split)

    def log_and_save_eval(self, trainer, pl_module, donor_split):
        self.epoch = -1 if pl_module.global_step == 0 else pl_module.current_epoch
        self.log_predictions(trainer, pl_module, donor_split)
        self.add_all_genes_to_metrics_history(pl_module, donor_split)
        self.log_per_gene_per_tissue_metrics(trainer, pl_module, donor_split)
        # clear for the next epoch
        pl_module.pred_dict = {}
        pl_module.target_dict = {}
        pl_module.donor_dict = {}
        pl_module.rank_dict = {}
        pl_module.rc_dict = {}
        self.reset_metrics_history()

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_and_save_eval(trainer, pl_module, 'valid')

    def on_test_epoch_end(self, trainer, pl_module):
        self.log_and_save_eval(trainer, pl_module, 'test')
