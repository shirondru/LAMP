#!/bin/bash
# Fallback if `conda env create -f environment.yml` fails on your system.
# LAMP was tested with the exact versions in environment.yml; this script leaves most versions unpinned.
set -e
eval "$(${CONDA_PATH}/bin/conda shell.bash hook)"
conda create -y -n lamp python=3.12
conda activate lamp

# PyTorch with CUDA. Change cu128 to match your system (`nvidia-smi` shows the highest CUDA version your driver supports;
# available versions: https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# enformer_pytorch is pinned because it defines the model (and breaks on transformers 5); tangermeme>=1.6.0 handles N's in personal genomes
pip install \
  lightning \
  torchmetrics \
  "enformer_pytorch==0.8.8" \
  "tangermeme>=1.6.0" \
  bionemo-noodles \
  "transformers<5" \
  numpy \
  pandas \
  pyarrow \
  scikit-learn \
  xarray \
  netCDF4 \
  pyyaml \
  pysam \
  wandb \
  jupyter \
  ipdb
