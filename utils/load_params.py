import wandb


def load_parameters_from_wandb(run_id):
    """Load parameters from a wandb run."""
    api = wandb.Api(timeout=120)
    run = api.run(f'{run_id}')
    parameters = run.config

    # Remove unused parameters.
    parameters.pop('seed', None)
    parameters.pop('window_size', None)
    parameters.pop('track', None)
    parameters.pop('method', None)
    parameters.pop('max_steps', None)
    parameters.pop('max_iterations')
    parameters.pop('warm_start', None)
    parameters.pop('tolerance', None)
    parameters.pop('dimensions', None)
    return parameters
