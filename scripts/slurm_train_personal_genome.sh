#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=400G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --job-name=LAMP_PG
# Submit from the LAMP root: sbatch scripts/slurm_train_personal_genome.sh
# Any extra arguments are passed to train.py. Slide-motif augmentation:
#   sbatch scripts/slurm_train_personal_genome.sh --motif_flank 50 --min_tss_dist 1000
source ${LAMP_CONFIG}
eval "$(${CONDA_PATH}/bin/conda shell.bash hook)"
conda activate lamp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ulimit -n 4096

extra_args=("$@")
cd ${LAMP_ROOT}/lamp
srun python train.py \
    --mode personal_genome \
    --config_path ${LAMP_ROOT}/configs/personal_genome.yaml \
    --motif_positions ${LAMP_ROOT}/data/motif_positions/STAT1_uniform_motif_positions_49kb.csv \
    --fold 0 \
    --seed 0 \
    "${extra_args[@]}"
