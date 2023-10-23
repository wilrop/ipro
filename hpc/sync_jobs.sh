#!/bin/bash -l

#SBATCH --job-name=wandbsync
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-user=willem.ropke@vub.be
#SBATCH --mail-type=ALL

module load wandb/0.13.4-GCCcore-11.3.0

wandb login 8966d6039f5932959dbc2d8d40621f3da0952c48
export WANDB_DIR=$VSC_SCRATCH
export OMP_NUM_THREADS=1

FILE_LIST="${VSC_SCRATCH}/finished_logs.txt"

while read -r l1; do
  read -r l2
  read -r l3
  read -r l4
  read -r l5
  wandb sync $l1 &
  wandb sync $l2 &
  wandb sync $l3 &
  wandb sync $l4 &
  wandb sync $l5
  wait
done <${FILE_LIST}
