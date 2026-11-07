#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=300G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --job-name=LAMP_VE
# Submit from the LAMP root:
#   sbatch scripts/slurm_train_variant_effect.sh variant_effect
#   sbatch scripts/slurm_train_variant_effect.sh variant_effect_twinnet
# Any arguments after the mode are passed to train.py, e.g. capped background sampling:
#   sbatch scripts/slurm_train_variant_effect.sh variant_effect --background_sampling capped --max_background_per_variant 10
source ${LAMP_CONFIG}
eval "$(${CONDA_PATH}/bin/conda shell.bash hook)"
conda activate lamp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ulimit -n 4096

mode=${1:-variant_effect}
extra_args=("${@:2}")
cd ${LAMP_ROOT}/lamp
srun python train.py \
    --mode $mode \
    --config_path ${LAMP_ROOT}/configs/${mode}.yaml \
    --motif_positions ${LAMP_ROOT}/data/motif_positions/STAT1_uniform_motif_positions_49kb.csv \
    --background_snvs ${LAMP_ROOT}/data/background_snvs/background_snvs_Whole_Blood.csv.gz \
    --background_sampling equal \
    --fold 0 \
    --seed 0 \
    "${extra_args[@]}"
