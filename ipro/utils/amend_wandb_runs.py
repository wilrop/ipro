import wandb


def amend_wandb_runs():
    """Amend the hypervolume of runs."""
    api = wandb.Api(timeout=120)
    runs = api.runs("wilrop/IPRO_runs")
    for run in runs:
        if 'outer/hypervolume' not in run.summary:
            try:
                run.summary['outer/hypervolume'] = run.history(keys=['outer/hypervolume']).iloc[-1]['outer/hypervolume']
            except IndexError:
                run.summary['outer/hypervolume'] = 0.0
            run.summary.update()
            print(f'❌ Amended {run.name}')
        else:
            print(f'✔️ Already amended {run.name}')


if __name__ == '__main__':
    amend_wandb_runs()
