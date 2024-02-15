#!/bin/bash

#SBATCH --job-name=grid_search
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --array=1-540

# Load the necessary modules.
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load PyTorch/1.12.0-foss-2022a
module load pygmo/2.18.0-foss-2022a
module load wandb/0.13.4-GCCcore-11.3.0
module load MuJoCo/2.2.2-GCCcore-11.3.0

export OMP_NUM_THREADS=1

# Define variables.
OPTIMISATION_DIR="${VSC_HOME}/ipro/optimisation"

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:$VSC_HOME/ipro"

# Set wandb directory.
export WANDB_DIR=$VSC_SCRATCH

# Sleep for a random number of seconds to avoid overloading the file system.
sleep $((($RANDOM % 240) + 1))s

# Run the experiments.
python3 ${OPTIMISATION_DIR}/grid_search.py \
  --config ${OPTIMISATION_DIR}/grid_a2c_dst.yaml \
  --exp_id ${SLURM_ARRAY_TASK_ID} \
  --offset 0
