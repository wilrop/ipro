# IPRO - Iterated Pareto Referent Optimisation

Iterated Pareto Referent Optimisation (IPRO) is an algorithm that decomposes learning the Pareto front into a sequence
of single-objective problems, each of which is solved by an oracle and leads to a non-dominated solution. IPRO is
guaranteed to converge and comes with an upper bound on the distance to the Pareto front at each iteration. You can 
checkout the paper [here](https://arxiv.org/abs/2202.10794).

## Structure

The repository is structured as follows:

- `analysis/` contains scripts to analyse the results of the experiments.
- `configs/` contains the configuration files for the experiments.
- `environments/` contains metadata on the environments and a one hot observation wrapper.
- `experiments/` contains the scripts to run the experiments and baselines.
- `linear_solvers/` contains the solvers for the initial (linear) problems in the sequence generated by IPRO.
- `optimisation/` contains the code to initialise wandb sweeps and run the grid search.
- `oracles/` contains the oracles that can be used in IPRO to solve the single-objective problems.
- `outer_loops/` contains the implementation of IPRO and IPRO-2D.
- `utility_function/` contains the code to generate and perform the utility-based evaluation.
- `utils/` contains miscellaneous utility functions.

## Running the code
> [!NOTE]
> To reproduce the exact results from the paper, please run IPRO from the `paper` branch.

To run the code, you need to install the dependencies in `requirements.txt`. We recommend using a virtual environment to
avoid conflicts with other packages. You can install the dependencies by running:

```
pip install -r requirements.txt
```

To run the experiments, you can use the scripts in the `experiments/` folder. For example, to run the IPRO algorithm
with the default configuration, run:

```
python experiments/run_experiment.py
```

## Citation

If you use this code or the results in your research, please use the following BibTeX entry:

```
@misc{ropke2025divide,
    title={Divide and Conquer: Provably Unveiling the Pareto Front with Multi-Objective Reinforcement Learning}, 
    author={Willem Röpke and Mathieu Reymond and Patrick Mannion and Diederik M. Roijers and Ann Nowé and Roxana Rădulescu},
    year={2025},
    url={https://arxiv.org/abs/2402.07182},
}
```