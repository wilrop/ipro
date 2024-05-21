#!/bin/bash

#SBATCH --job-name=grid_search
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/gs_%A_%a.out
#SBATCH --array=1-60

# Load the necessary modules.
module load Python/3.10.4-GCCcore-11.3.0 \
  SciPy-bundle/2022.05-foss-2022a \
  PyTorch/1.12.0-foss-2022a \
  pygmo/2.18.0-foss-2022a \
  wandb/0.13.4-GCCcore-11.3.0 \
  MuJoCo/2.2.2-GCCcore-11.3.0

export OMP_NUM_THREADS=1

# Define variables.
IPRO_DIR="${VSC_HOME}/ipro"
OPTIMISATION_DIR="${IPRO_DIR}/optimisation"
NUM_LINES=$(wc -l <${IPRO_DIR}/hpc/sweep_ids.txt)
LINE=$((${SLURM_ARRAY_TASK_ID} % ${NUM_LINES} + 1))
SWEEP_ID=$(head -${LINE} ${IPRO_DIR}/hpc/sweep_ids.txt | tail -1)
FN_TYPE="increasing_cumsum"
U_DIR="${IPRO_DIR}/utility_function/utility_fns/${FN_TYPE}"

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:$VSC_HOME/ipro"

# Set wandb directory.
export WANDB_DIR=$VSC_SCRATCH

# This forces the jobs to start sequentially. The startup time is estimated at 2 seconds.
sleep $(((${SLURM_ARRAY_TASK_ID} - 1) * 2))s

# Run the experiments.
python3 ${OPTIMISATION_DIR}/run_agent.py \
  --project IPRO_dst_grid \
  --sweep_id ${SWEEP_ID} \
  --u_dir ${U_DIR}
