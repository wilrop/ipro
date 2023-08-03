import wandb


class OuterLoop:
    def __init__(self,
                 oracle,
                 track=False,
                 exp_name=None,
                 wandb_project_name=None,
                 wandb_entity=None):
        self.oracle = oracle

        self.track = track
        self.exp_name = exp_name
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity

    def setup_wandb(self, config):
        """Setup wandb."""
        if self.track:
            config.update(self.oracle.config())
            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config=config,
                name=self.exp_name,
            )

            wandb.define_metric('iteration')
            wandb.define_metric('outer/dominated_hv', step_metric='iteration')
            wandb.define_metric('outer/discarded_hv', step_metric='iteration')
            wandb.define_metric('outer/coverage', step_metric='iteration')
            wandb.define_metric('outer/error', step_metric='iteration')

    def close_wandb(self):
        """Close wandb."""
        if self.track:
            wandb.finish()

    def log_iteration(self, iteration, dominated_hv, discarded_hv, coverage, error):
        """Log the iteration."""
        if self.track:
            wandb.log({
                'outer/dominated_hv': dominated_hv,
                'outer/discarded_hv': discarded_hv,
                'outer/coverage': coverage,
                'outer/error': error,
                'iteration': iteration
            })
