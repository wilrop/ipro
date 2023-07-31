#!/bin/bash

#SBATCH --job-name=optimize
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err
#SBATCH --array=1-6

# Load the necessary modules.
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load PyTorch/1.12.0-foss-2022a
module load pygmo/2.18.0-foss-2022a
module load Optuna/2.10.0-foss-2021b
module load tensorboard/2.10.0-foss-2022a

pip install --user gymnasium
pip install --user mo-gymnasium
pip install --user highway-env
pip install --user wandb

# Define variables.
YAML_FILE="${VSC_HOME}/geohunt/hpc/yaml_files.txt"

export PYTHONPATH="${PYTHONPATH}:$VSC_HOME/geohunt"

# Run the experiments.
python3 $VSC_HOME/geohunt/optimization/search.py --params $(head -${SLURM_ARRAY_TASK_ID} $YAML_FILE | tail -1)
