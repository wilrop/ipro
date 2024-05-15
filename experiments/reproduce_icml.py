import argparse
from experiments.reproduce_experiment import reproduce_experiment


best_runs = {
    "deep-sea-treasure-concave-v0":
        {
            "SN-MO-PPO": [
                "wilrop/IPRO_ppo_grid/gwa8nvbe",
                "wilrop/IPRO_ppo_grid/21c7ycmc",
                "wilrop/IPRO_ppo_grid/hbu3ybfq",
                "wilrop/IPRO_ppo_grid/u6te97bq",
                "wilrop/IPRO_ppo_grid/w8qfs9w5"
            ],
            "SN-MO-DQN": [
                "wilrop/IPRO_runs/1ojc222h",
                "wilrop/IPRO_runs/avocph1w",
                "wilrop/IPRO_runs/dkny8i4v",
                "wilrop/IPRO_runs/1gjje4pj",
                "wilrop/IPRO_runs/2tq9otrp"],
            "SN-MO-A2C": [
                "wilrop/IPRO_a2c_grid/3481rvcj",
                "wilrop/IPRO_a2c_grid/21clruwk",
                "wilrop/IPRO_a2c_grid/3as3zg3g",
                "wilrop/IPRO_a2c_grid/2i8tyz7s",
                "wilrop/IPRO_a2c_grid/29z8i2e1"
            ]
        },
    "minecart-v0":
        {
            "SN-MO-PPO": [
                "wilrop/IPRO_runs/1w9vpxix",
                "wilrop/IPRO_runs/363i3xko",
                "wilrop/IPRO_runs/1bj4yyap",
                "wilrop/IPRO_runs/2gl8rzvi",
                "wilrop/IPRO_runs/1y8ragru"
            ],
            "SN-MO-DQN": [
                "wilrop/IPRO_runs/8055wxug",
                "wilrop/IPRO_runs/2cxy8hpl",
                "wilrop/IPRO_runs/zjtykldy",
                "wilrop/IPRO_runs/2z4i5byd",
                "wilrop/IPRO_runs/3b6iyi47"],
            "SN-MO-A2C": [
                "wilrop/IPRO_runs/38djofsu",
                "wilrop/IPRO_runs/13kslpir",
                "wilrop/IPRO_runs/2zhwcgpg",
                "wilrop/IPRO_runs/34ov8sxl",
                "wilrop/IPRO_runs/14490d6q"
            ]
        },
    "mo-reacher-v4":
        {
            "SN-MO-PPO": [
                "wilrop/IPRO_runs/210fl68y",
                "wilrop/IPRO_runs/fvisrcpq",
                "wilrop/IPRO_runs/13lwvxzx",
                "wilrop/IPRO_runs/36bxc0s6",
                "wilrop/IPRO_runs/24a8hucc"
            ],
            "SN-MO-DQN": [
                "wilrop/IPRO_runs/kz6xxbxi",
                "wilrop/IPRO_runs/5djz09xq",
                "wilrop/IPRO_runs/3rf3j8ej",
                "wilrop/IPRO_runs/1s4hsn7d",
                "wilrop/IPRO_runs/1rd810nx"
            ],
            "SN-MO-A2C": [
                "wilrop/IPRO_runs/ltkv0vv8",
                "wilrop/IPRO_runs/3f4yyb16",
                "wilrop/IPRO_runs/p94nf972",
                "wilrop/IPRO_runs/sen51jbt",
                "wilrop/IPRO_runs/tvv4zhbw"
            ]
        }
}


def reproduce_icml(u_dir, exp_id):
    """Reproduce an experiment given its ID."""
    i = 1
    for env_id in best_runs:
        for oracle in best_runs[env_id]:
            for seed, run_id in enumerate(best_runs[env_id][oracle]):
                if i == exp_id:
                    reproduce_experiment(oracle, env_id, seed, run_id, u_dir)
                    return
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce experiments from a YAML file.')
    parser.add_argument(
        '--u_dir',
        type=str,
        default='./utility_function/utility_fns',
        help='Path to directory containing utility functions.'
    )
    parser.add_argument('--exp_id', type=str, default=1)
    parser.add_argument('--exp_dir', type=str, default='./icml_configs')
    args = parser.parse_args()

    reproduce_icml(args.u_dir, args.exp_id)
