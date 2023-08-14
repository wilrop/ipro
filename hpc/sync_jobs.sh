#!/bin/bash -l

#SBATCH --job-name=wandbsync
#SBATCH --time=4:00:00
#SBATCH --partition=skylake,skylake_mpi
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G

module load wandb/0.13.4-GCCcore-11.3.0

wandb login 8966d6039f5932959dbc2d8d40621f3da0952c48
export WANDB_DIR=$VSC_SCRATCH
export OMP_NUM_THREADS=1

while read -r l1; do
  read -r l2
  read -r l3
  read -r l4
  read -r l5
  read -r l6
  read -r l7
  read -r l8
  read -r l9
  read -r l10
  wandb sync $l1 &
  wandb sync $l2 &
  wandb sync $l3 &
  wandb sync $l4 &
  wandb sync $l5 &
  wandb sync $l6 &
  wandb sync $l7 &
  wandb sync $l8 &
  wandb sync $l9 &
  wandb sync $l10
  wait
done <${VSC_SCRATCH}/finished_logs.txt
