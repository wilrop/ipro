#!/bin/bash

#SBATCH --job-name=opt_dqn_highway
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err

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

# Define the log directory.
LOGDIR="${VSC_SCRATCH}/IPRO"

export PYTHONPATH="${PYTHONPATH}:$VSC_HOME/geohunt"

# Run the experiments.
python3 $VSC_HOME/geohunt/experiments/opt_dqn_highway.py --log_dir "$LOGDIR"
