#!/bin/bash

#SBATCH --job-name=grid_search
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4gb
#SBATCH --partition=skylake,skylake_mpi
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/gs_%A_%a.out
#SBATCH --array=1-60

# Load the necessary modules.
module load Python/3.10.4-GCCcore-11.3.0

export OMP_NUM_THREADS=1

# Define paths.
PROJ_DIR="${VSC_SCRATCH}/ipro"
IPRO_DIR="${PROJ_DIR}/ipro"
OPTIMISATION_DIR="${IPRO_DIR}/optimisation"
FN_TYPE="increasing_cumsum"
U_DIR="${IPRO_DIR}/utility_function/utility_fns/${FN_TYPE}"

# load virtual environments and set some variables.
source ${PROJ_DIR}/venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$VSC_SCRATCH/ipro"
export WANDB_DIR=$VSC_SCRATCH

# Define the sweep id.
NUM_LINES=$(wc -l <${PROJ_DIR}/hpc/sweep_ids.txt)
LINE=$((${SLURM_ARRAY_TASK_ID} % ${NUM_LINES} + 1))
SWEEP_ID=$(head -${LINE} ${PROJ_DIR}/hpc/sweep_ids.txt | tail -1)

# This forces the jobs to start sequentially. The startup time is estimated at 2 seconds.
sleep $(((${SLURM_ARRAY_TASK_ID} - 1) * 2))s

# Run the experiments.
python3 ${OPTIMISATION_DIR}/run_agent.py \
  --wandb_project_name IPRO_dst_grid \
  --sweep_id ${SWEEP_ID} \
  --u_dir ${U_DIR}
