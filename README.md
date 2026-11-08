# LAMP
This repository implements the artificial SNV (aSNV) training framework we detail in our manuscript: **LAMP sheds light on how sequence to function models learn variant effects during post-training**


LAMP fine-tunes Enformer on simulated genetic effects on gene expression, using GTEx whole-blood expression and GTEx personal genomes.

For each gene, one aSNV is placed inside a TF motif (STAT1 by default). The effect size of this aSNV is set to be deliberately large by default, such that it explains most of the variance in gene expression across individuals (R² = 0.5 by default). This incentivizes the model to learn its effect. By placing the aSNV in the same relative position of the same TF motif over and over throughout the genome, the pattern should also be simple to learn. This strategy (LAMP) can then be used to track how models learn these variant effects. This implementation uses Enformer to fine-tune on these aSNV effects, but in theory other sequence-to-function models can be used instead, by inheriting from `dataset.MultiTissueLitModel` and defining a new model class. There are three ways to train:

| `--mode` | Trains on | Evaluates on | Model |
|---|---|---|---|
| `personal_genome` | GTEx personal genomes carrying the synthetic variant | personal genomes of held-out genes and people (HOGP) | Enformer + linear head |
| `variant_effect` | The reference genome (hg38) with each aSNV's 'effect allele' and 'other allele' as separate examples. There are also background training SNV, where the alt and ref alleles are separate examples. These are GTEx-observed SNVs with their eQTL-test betas as effect sizes. We include these during training so the model doesn't overfit to only seeing aSNVs, and nothing else, during training. Without them, we observe this leads to memorization of the aSNV effect size while paying little attention to the precise motif-perturbation pattern that causes it. | HOGP | Enformer + linear head |
| `variant_effect_twinnet` | Pairs of (effect allele, other allele) reference sequences for each aSNV, with the expression difference as the target. For background SNVs, we use pairs of (alt, ref). Here, each pair of alleles (2 sequences) is one contrastive training example | HOGP | TwinNet: Enformer trunk(alt) − trunk(ref) |

## 1. Setup

### Define paths that are referenced by various scripts

- `LAMP_ROOT`: the absolute path of this repository, e.g. `/home/you/LAMP`. Data is read from `$LAMP_ROOT/data` and results go to `$LAMP_ROOT/results`. Files refer to it as `${LAMP_ROOT}`.
- `CONDA_PATH`: the base directory of your conda installation (the folder that contains `bin/conda`, e.g. `~/miniforge3`). The SLURM scripts use it to activate the `lamp` environment.

Edit `paths.config` to your paths:
```bash
export LAMP_ROOT=/home/you/LAMP
export CONDA_PATH=/home/you/miniforge3
```

Then add these two lines to the end of your `~/.bashrc`, and run `source ~/.bashrc`:
```bash
export LAMP_CONFIG=/home/you/LAMP/paths.config
source $LAMP_CONFIG
```
Check with `echo $LAMP_ROOT $CONDA_PATH`. The SLURM scripts run `source ${LAMP_CONFIG}` themselves.

### Environment
With `CONDA_PATH` set, create the `lamp` conda environment:
```bash
bash install_reqs.sh
conda activate lamp
```
This installs PyTorch built for CUDA 12.8. If your system needs a different CUDA version, change `cu128` in `install_reqs.sh` to match (see https://pytorch.org/get-started/locally/).

Alternatively, install the exact environment LAMP was developed with (PyTorch 2.3.1, CUDA 12.1):
```bash
conda env create -f environment.yml
conda activate lamp
```

### Weights & Biases
I use W&B to log training runs. Follow the [Weights & Biases quickstart](https://docs.coreweave.com/products/wandb/quickstart). Be sure to run `wandb login` and paste your API key, to avoid wandb issues when launching training runs. If your cluster has no access to the internet, set `export WANDB_MODE=offline` within the job script (more details on these job scripts below). 

## 2. Data

### Included in `data/`
| File | Contents |
|---|---|
| `expression/concat_fold0.nc` | GTEx v8 whole-blood expression (gene × donor × tissue), rescaled so each gene has SD ≈ 1, so that the same effect size is derived to explain the same variance in gene expression across individuals, no matter the gene. This is the expression data we used in our manuscript, where we detail these decisions in greater detail. |
| `filtered_metadata.csv` | GTEx donors with RNA-seq data in each tissue |
| `cross_validation_folds/gtex/cv_folds/` | train/valid/test donor IDs for fold 0 |
| `Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval_NoDuplicatesNoOverlappingBins.csv` | 196,608 bp TSS-centered window for each gene (1-based), respecting Enformer's original train/validation/test splits |
| `gene_lists/` | `train_genes_3k` (2,725), `train_genes_279`, `all_pc_train_genes` (12,111), `valid_genes` (92), `all_blood_test_genes` (1,467), `all_blood_genes` |
| `motif_positions/STAT1_uniform_motif_positions_49kb.csv` | CSV where each row corresponds to a STAT1 aSNV, its MAF, and the genomic position of the motif it perturbs. These motif positions come from pre-processing in step 6 |
| `jaspar_pfms/` | JASPAR 2024 PFMs (MEME format) |
| `human_CORE_JASPAR_TFs.csv` | JASPAR TF names and motif IDs |
| `gencode_v26_gene_id_to_name.tsv` | GTEx v8 gene IDs to gene names |
| `background_snvs/background_snvs_Whole_Blood.csv.gz` | background SNVs for training models on direct variant effects. This was created in scripts/background_snvs/make_background_snvs.ipynb  |

### Data to download and/or preprocess
1. **Reference genome:** Please download and place the hg38 reference genome, plus its `.fai` index in `data/hg38_genome.fa[i]`

2. **Personal genomes** (controlled access). Unfortunately, we cannot release GTEx Whole Genome Sequences, because these data are protected. To make use of LAMP, you must have access to whole genome data from GTEx. Alternatively, if you have whole-genome sequencing with paired expression from another source (e.g., GEUVADIS) this repository can be adapted to make use of this. 
 LAMP expects two unphased consensus FASTAs per donor, one per haplotype, at
   `data/ConsensusSeqs_SNPsOnlyUnphased/{donor_id}_consensus_H{1,2}.fa` (each with a `.fai`).

   To make the SNPs-only VCF/BCF and the consensus FASTAs from `GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.vcf.gz`, follow the repository's instructions to use the scripts in [`code/process_vcf/`](https://github.com/shirondru/enformer_fine_tuning/tree/master/code/process_vcf) of [enformer_fine_tuning](https://github.com/shirondru/enformer_fine_tuning) (written for an SGE cluster). 

3. **Enformer weights.** These download automatically from Hugging Face (`EleutherAI/enformer-official-rough`). If your cluster runs without access to the internet, you will need to pre-download the weights from huggingface and pass in the path to the downloaded weights to Enformer.from_pretrained(path_to_weights)


## 3. Training

### On SLURM
Job scripts expect a SLURM scheduler, which was used to manage our cluster. Please treat these job scripts as examples that will likely need to be slightly modified to conform to your clusters specifications. 
```bash
mkdir -p logs
sbatch scripts/slurm_train_personal_genome.sh
sbatch scripts/slurm_train_personal_genome.sh --motif_flank 50 --min_tss_dist 1000      # with slide-motif augmentation
sbatch scripts/slurm_train_variant_effect.sh variant_effect
sbatch scripts/slurm_train_variant_effect.sh variant_effect_twinnet
sbatch scripts/slurm_train_variant_effect.sh variant_effect --background_sampling capped --max_background_per_variant 10
```
These job scripts can accept additional arguments that can be passed to `lamp/train.py`, if desired. See some important arguments below. 

**IMPORTANT**: the `--n_gpus` argument in `lamp/train.py`  (or `n_gpus` in the config) must match the GPUs requested from SLURM. You can choose to specify n_gpus either with `--n_gpus` or within the config. 

### Directly
```bash
cd $LAMP_ROOT/lamp
python train.py --mode personal_genome \
    --config_path ../configs/personal_genome.yaml \
    --motif_positions ../data/motif_positions/STAT1_uniform_motif_positions_49kb.csv \
    --fold 0 --seed 0 --n_gpus 1
```
For a quick test, add `--debug 1 --max_epochs 1 --train_gene_path ../data/gene_lists/train_genes_279.txt`. Debug mode uses one GPU, skips saving datasets to avoid generating unecessary files, and logs to the W&B project `LAMP_test` (to avoid wandb bloat)

### Main arguments
| Argument | Meaning |
|---|---|
| `--mode` | `personal_genome`, `variant_effect`, or `variant_effect_twinnet` |
| `--config_path` | YAML config (`configs/`). Command line arguments can override config values |
| `--motif_positions` | CSV where each row corresponds to an aSNV, its MAF, and the genomic position of the motif it perturbs. These motif positions come from pre-processing in step 6 |
| `--background_snvs` | variant modes: background SNV CSV. Defaults to `data/background_snvs/background_snvs_Whole_Blood.csv.gz`. `none` trains on synthetic variants only, which makes the model overfit and predict large effect sizes for all variants |
| `--background_sampling` | variant modes: `equal` (default) duplicates synthetic variants so synthetic and background examples are 1:1; `capped` keeps at most `--max_background_per_variant` background SNVs per gene |
| `--motif_flank`, `--min_tss_dist` | personal_genome mode: slide the causal motif plus `motif_flank` nt during training, keeping it at least `min_tss_dist` nt from the TSS |
| `--target_r2` | variance explained by each aSNV (default is 50% in the configs, same as in the paper) |
| `--expression_delta` | set the synthetic effect size directly instead (requires you to specify the target R² to be -1 in the config) |
| `--train_gene_path`, `--valid_gene_path`, `--test_gene_path` | override the gene lists to this path instead |
| `--ckpt_path` | resume training from a checkpoint |

`train.py` also has four hook functions (`choose_ds`, `update_ds_hook`, `update_args_for_config_hook`, `update_config_hook`). They do nothing by default and are there so you can customise an experiment from your own script without editing the base implementation.

### Effect sizes
- **aSNVs.** At R² = 0.5 and allele frequency 0.5 this is about 2 (see details on why in our paper's Methods section). You can inspect the effect size via the `expression_delta` attribute in a dataset object. 

### Minor allele frequency (MAF)
The minor allele frequency of each aSNV is established by the `effect_allele_freq` column of the motif_positions csv file (e.g., motif_positions/STAT1_uniform_motif_positions_49kb.csv). By default, most aSNVs are given an MAF of 50%. For some aSNVs (100 of them in `STAT1_uniform_motif_positions_49kb.csv`) the MAF is set to 0%. These aSNVs are therefore not observed during training but appear around these genes for the first time during validation and test time. 

For models trained directly on variant effects, this MAF is used to help establish the effect size of the aSNV, because the effect size is calculated to satisfy a target R², and the effect size that will yield a given R² depends on the MAF of the aSNV. But, for variant effect style models, the MAF is not the actual frequency the aSNV appears during training. How often the aSNV appears relative to background SNVs depends on --background_sampling: with `equal`, every aSNV is repeated so that aSNV and background examples are 1:1 across the training set. This means the same aSNV is appears many times during each training epoch. With `capped`, each aSNV appears once alongside at most `--max_background_per_variant` of its gene's background SNVs.


### Motif sliding augmentations during training
In `personal_genome` mode, the optional slide-motif augmentation (`--motif_flank`, `--min_tss_dist`) moves the causal motif plus flanking sequence to a random position on the same side of the TSS every time a sequence is drawn. We observed this helps promote model invariance to the aSNV's distance from the TSS, leading to better learning and generalization of aSNV effects, no matter where in the sequence they appear during test time. We reported this in our manuscript. This option enables its usage. 

### Config files
These .yaml files (some default ones are included in `configs/`) specify various training hyerparameters, like learning rate, patience, target R², and sequence length. Note that the sequence length specified in this file must match the window size between `short_end0` and `short_start0` in the `motif_positions` csv.

These files also specify `position_idx` which determines the relative position of the aSNV within a perturbed motif. 

## 4. Outputs
- **Local files.** Each run writes to `$LAMP_ROOT/results/{experiment_name}/Fold-{fold}/Seed-{seed}/{wandb run id}/`:
  - `Prediction_Results_*`: every prediction, with `y_pred`, `y_true`, `gene`, and `donor` for both strand orientations.
  - `CrossIndivMetrics_*`: per-gene cross-individual Pearson R, R², and MSE.
  - the config, the motif positions, the synthetic genotypes (`*_synthetic_genotype.csv`) and effect sizes (`*_expression_delta.csv`), and the best checkpoint. Best checkpoint is based on the metric that was monitored during the validation loop, defined within a config `.yaml` file. By default it is cross-individual correlation using individuals and genes in the validation set (HOGP).
- **W&B.** Runs go to the project in the config (`LAMP`), grouped by experiment name. They contain the training loss and the validation/test metrics averaged over genes, e.g. `mean_pearsonrz_across_valid_genes_across_valid_donors`, which is also used for checkpointing and early stopping. Wandb also logs GPU performance, so you can monitor how efficiently the GPU is being used in your setup.
- **Multi-GPU runs.** When training on multiple GPUs, each GPU gets its own wandb run id. Training is performed with data parallelism, meaning each GPU gets an identical copy of the model weights, but different portions of the data. Gradients updates are averaged over the GPUs. During validation and test time, each GPU gets a different portion of the validation and test set (but each GPU gets all individuals from a given gene). The metrics and predictions for each GPU are saved under the subdirectory corresponding to that GPU's wandb run id. This means predictions and metrics for different genes will be saved under different subdirectories, and you will need to combine those results. On W&B.com, the performance metrics will be aggregated automatically, but will only be displayed for the run corresponding to the main process (GPU 0).

- **TwinNet metrics.** TwinNet predicts the effect allele minus the other allele, while evaluation targets during the validation/test loops are each person's absolute expression (modified by the aSNV). We have the TwinNet form a prediction by passing in, for a given gene, a personal genome sequence and the reference genome sequence. It predicts the difference between the two. When individuals have the aSNV, this difference should be greater. But the absolute expression value may not be on the same scale as an individual's absolute expression. Thus we recommend evaluating these models only using cross-individual correlation, rather than R² and MSE, which will be sensitive to this absolute expression difference. Cross individual correlation will only be sensitive to whether the model correctly understands which individuals have higher or lower expression. 

## 5. Elastic net baseline
`lamp/elastic_net.py` fits a per-gene elastic net (`ElasticNetCV`) on train donors. Its features are SNV genotypes in the gene's input window, including the aSNV plus natural SNVs from the GTEx VCF seen in train donors. It is meant to very closely match the Enformer aSNV training setup, with two differences. (1) The elastic net is fit directly on SNV genotypes, and not pure sequence like Enformer. (2) We fit the elastic net directly on the genes that are held out for Enformer, to establish a baseline performance a model could reasonably achieve without needing to infer the effects of important variants (e.g., aSNVs) from what the model learned in other parts of the genome. See the Methods section in our manuscript for more details. **We highly recommend comparing fine-tuned Enformer's HOGP performance (cross-individual correlation) against an elastic net baseline, as we did in the paper, using this implementation.**


 It is evaluated on the same validation/test donors, genotypes and expression values as fine-tuned Enformer. It needs the indexed SNPs-only VCF from step 2 of Data, read by default from `data/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze_SNPsOnly.vcf.gz` .
```bash
cd $LAMP_ROOT/lamp
python elastic_net.py --save_dir ../results/{experiment_name}/Fold-0/Seed-0/{run id} --n_jobs 8
```
**Arguments**
- **--n_jobs**: Number of CPUs to use for cross validation. Gets passed to ElasticNetCV. For speeding things up.
- **--gene_groups** (default: all four): Which kind of genes to fit and evaluate the elastic net on. The gene groups here are named with respect to Enformer's fine-tuning set. For example, `test` means these were Enformer's test genes for LAMP. But the elastic net is fit directly on these genes, and then evaluated on held out people. In every case, the elastic net is fit directly on the genes, and evaluated on held out people. 
  - `train`: train genes whose synthetic variant was seen by Enformer during fine-tuning.
  - `rare-train`: rare-variant genes that Enformer saw during fine-tuning, but the aSNV was held-out. The elastic net will see the aSNV during training. The point of the elastic net here is to compare Enformer's accuracy on held-out people after not observing it during training, to an elastic net that did observe it during training. This group only exists for models trained on personal genomes, not variant effects.
  - `valid`, `test`: Enformer's unseen genes from the validaton or test sets. The point of the elastic net here is to compare Enformer's accuracy on held-out genes and people to a model that observed the held-out gene and aSNV during training. As always, both models are evaluated on the same held-out people. **This is the most important baseline to compare Enformer's HOGP accuracy to.**
- **--save_dir**: The path to the directory containing a fine-tuned Enformer model's results. This directory also contains the config file, individuals' genotypes and expression values, etc. This is key to fitting an elastic net model on identical data for maximum comparability.

**Variant sets**: The elastic net is fit one gene at a time. Two elastic nets are also fit for each gene. One is fit using all SNVs (aSNV + Natural SNVs, `snps_used_for_training` is set to `all`) and evaluated by predicting each individual's expression using on either their genotype for all SNVs (`snps_used_for_eval` set to `all`) or setting the value for all natural SNVs to 0 and only defining the genotype for the aSNV (`snps_used_for_eval` set to `only_synthetic`). Another elastic net is fit on only each person's aSNV genotype and evaluated using each held-out person's aSNV genotype (`snps_used_for_training` and `snps_used_for_eval` set to `only_synthetic` in the output prediction files). In the paper, we evaluated mainly using `snps_used_for_training` == `only_synthetic` and `snps_used_for_eval` == `only_synthetic`


**Output:** predictions are saved in the run folder as `EN_pred_{valid|test}_donor_SNPs_{gene group}_genes.csv`.

For a multi-GPU run, use any one of its run folders: the datasets are the same on every GPU.

## 6. Specifying motif (and aSNV) locations

A "motif_positions" file that specifies the positioning and allele frequency of each aSNV is necessary for any training run. In most experiments in our paper, we use `data/motif_positions/STAT1_uniform_motif_positions_49kb.csv` which was generated by randomly sampling one naturally occuring candidate STAT1 motif per Whole Blood expressed gene. The position of the aSNV is then placed in the same relative position of each selected motif.

You may choose to use this file for your own experiments. You will likely want to also deviate from this and specify experiments that position the aSNV in particular locations. For example, you may wish to place the aSNV only very far from the target gene. To do this, you will need to create your own motif_positions file. We have a template notebook to support this in `scripts/motif_positions/make_training_motif_positions.ipynb`. You can play around with this notebook and make the changes you desire to generate your own motif_positions file. 

The notebook needs a file with the genomic positions of all JASPAR 2024 matches to the TF's motif within each gene's window. Generate it with `scripts/motif_positions/get_tf_motif_csv.py`, which downloads the motif's genome-wide hg38 matches from JASPAR (~200 MB for STAT1) and keeps those in each gene's window. It takes about 5 minutes. For STAT1:
```bash
python scripts/motif_positions/get_tf_motif_csv.py --motif_id MA0137.4 --out data/STAT1.csv.gz
```
This reproduces the STAT1 matches we used to make `STAT1_uniform_motif_positions_49kb.csv`. To perturb another TF's motif, pass its JASPAR motif ID instead, e.g. for GATA1:
```bash
python scripts/motif_positions/get_tf_motif_csv.py --motif_id MA0035.5 --out data/GATA1.csv.gz
```
Look up TF names and motif IDs in `data/human_CORE_JASPAR_TFs.csv`.

Then use `scripts/motif_positions/make_training_motif_positions.ipynb` and set `TF_NAME`, `WINDOW_SIZE`, and `MIN_SCORE` in the first cell. It picks one hit per gene at random and marks 100 rare-variant genes. Edit the marked cell to choose motifs differently, e.g. by distance to the TSS.


## LLM Disclosure
LAMP code was originally written without LLM assistance by Shiron Drusinsky as part of a larger research codebase. To turn that into a simple release that keeps only code and data needed to run LAMP simulations without extra experiment-specific files and code that most users won't need or want, I used Claude Opus 5.5. I verified behavior didn't change, and results remained the same (i.e, no different than from changing the random seed). 
