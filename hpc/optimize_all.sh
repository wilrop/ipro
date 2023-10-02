#!/bin/bash

#SBATCH --job-name=optimize
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2gb
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL
#SBATCH --array=1-100

# Load the necessary modules.
module load Python/3.10.4-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load PyTorch/1.12.0-foss-2022a
module load pygmo/2.18.0-foss-2022a
module load Optuna/3.1.0-foss-2022a
module load tensorboard/2.10.0-foss-2022a
module load wandb/0.13.4-GCCcore-11.3.0
module load MuJoCo/2.2.2-GCCcore-11.3.0

pip install --user gymnasium
pip install --user mo-gymnasium
pip install --user highway-env

export OMP_NUM_THREADS=1

# Define variables.
NUM_LINES=$(wc -l <${VSC_HOME}/ipro/hpc/yaml_files.txt)
LINE=$((${SLURM_ARRAY_TASK_ID} % ${NUM_LINES} + 1))
YAML_FILE=$(head -${LINE} ${VSC_HOME}/ipro/hpc/yaml_files.txt | tail -1)
OPTIMIZATION_DIR="${VSC_HOME}/ipro/optimization"

# Set pythonpath
export PYTHONPATH="${PYTHONPATH}:$VSC_HOME/ipro"

# Set wandb directory.
export WANDB_DIR=$VSC_SCRATCH

# Sleep for a random number of seconds to avoid overloading the file system.
sleep $((($RANDOM % 20) + 1))s

# Run the experiments.
python3 ${OPTIMIZATION_DIR}/search.py \
  --params ${OPTIMIZATION_DIR}/${YAML_FILE} \
  --log_dir ${VSC_SCRATCH}
