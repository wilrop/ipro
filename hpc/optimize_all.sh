#!/bin/bash

#SBATCH --job-name=optimize
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err
#SBATCH --array=1-100

# Load the necessary modules.
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load PyTorch/1.12.0-foss-2022a
module load pygmo/2.18.0-foss-2022a
module load Optuna/2.10.0-foss-2021b
module load tensorboard/2.10.0-foss-2022a
module load wandb/0.13.4-GCCcore-11.3.0

pip install --user gymnasium
pip install --user mo-gymnasium
pip install --user highway-env

export OMP_NUM_THREADS=1

# Define variables.
NUM_LINES=$(wc -l < ${VSC_HOME}/geohunt/hpc/yaml_files.txt)
LINE=$(( ${SLURM_ARRAY_TASK_ID} % ${NUM_LINES} + 1 ))
YAML_FILE=$(head -${LINE} ${VSC_HOME}/geohunt/hpc/yaml_files.txt | tail -1)
OPTIMIZATION_DIR="${VSC_HOME}/geohunt/optimization"

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:$VSC_HOME/geohunt"

# Set wandb directory.
export WANDB_DIR=$VSC_SCRATCH

# Run the experiments.
python3 ${OPTIMIZATION_DIR}/search.py \
--params ${OPTIMIZATION_DIR}/${YAML_FILE} \
--log_dir ${VSC_SCRATCH}
