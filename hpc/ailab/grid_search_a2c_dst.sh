#!/bin/bash

#SBATCH --job-name=gs_a2c_dst
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --nodelist=node103
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --output=logs/gs_a2c_dst_%A_%a.out
#SBATCH --array=1-640

# Load the necessary modules.
module load Python/3.11.5-GCCcore-13.2.0

export OMP_NUM_THREADS=1

# Define variables.
IPRO_DIR="${HOME}/ipro"
VENV_DIR="${IPRO_DIR}/venv"
OPTIMISATION_DIR="${IPRO_DIR}/optimisation"
CONFIG_DIR="${IPRO_DIR}/configs"
FN_TYPE="increasing_cumsum"
U_DIR="${IPRO_DIR}/utility_function/utility_fns/${FN_TYPE}"

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:$HOME/ipro"

source "${VENV_DIR}/bin/activate"

# Sleep for a random number of seconds to avoid overloading the file system.
sleep $((($RANDOM % 240) + 1))s

# Run the experiments.
python3 ${OPTIMISATION_DIR}/grid_search.py \
  --config ${CONFIG_DIR}/sn_a2c_dst.yaml \
  --u_dir ${U_DIR} \
  --exp_id ${SLURM_ARRAY_TASK_ID} \
  --offset 0
