# IPRO - Iterated Pareto Referent Optimisation
## Installation
1. Change line 105 in ``outer_loops/outer.py`` to reflect the correct directory.
2. You need to create a simlink from the ``wandb`` folder to ``wandb-cache`` that we make ourselves. This is because we will overwrite internal debugging logs to ``null``. Symlink: ``mkdir $VSC_SCRATCH/wandb-cache; cd ~/.cache; ln -s $VSC_SCRATCH/wandb-cache wandb``

## Running experiments
1. Run ``sbatch geohunt/hpc/optimize_all.sh`` 
2. Once finished, go to ``$VSC_SCRATCH`` and run ``$VSC_HOME/geohunt/hpc/list_runs.sh`` to list the finished runs
3. Sync the finished logs to wandb with ``sbatch geohunt/hpc/sync_jobs.sh``

