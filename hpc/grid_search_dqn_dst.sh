#!/bin/bash

#SBATCH --job-name=gs_dqn_dst
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --array=1-100

# Load the necessary modules.
module load Python/3.10.4-GCCcore-11.3.0 \
 SciPy-bundle/2022.05-foss-2022a \
 PyTorch/1.12.0-foss-2022a \
 pygmo/2.18.0-foss-2022a \
 wandb/0.13.4-GCCcore-11.3.0 \
 MuJoCo/2.2.2-GCCcore-11.3.0

export OMP_NUM_THREADS=1

# Define variables.
OPTIMISATION_DIR="${VSC_HOME}/ipro/optimisation"
CONFIG_DIR="${VSC_HOME}/ipro/configs"
FN_TYPE="increasing_cumsum"
U_DIR="${VSC_HOME}/ipro/utility_function/utility_fns/${FN_TYPE}"

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:$VSC_HOME/ipro"

# Set wandb directory.
export WANDB_DIR=$VSC_SCRATCH

# This forces the jobs to start sequentially. The startup time is estimated at 2 seconds.
sleep $(((${SLURM_ARRAY_TASK_ID} - 1) *2))s

# Run the experiments.
python3 ${OPTIMISATION_DIR}/grid_search.py \
  --config ${CONFIG_DIR}/sn_dqn_dst.yaml \
  --u_dir ${U_DIR} \
  --exp_id ${SLURM_ARRAY_TASK_ID} \
  --offset 0
