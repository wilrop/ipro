import os
import torch
import numpy as np

from environments.bounding_boxes import get_bounding_box
from utility_function.monotonic_utility import MonotonicUtility


def generate_utility_fns(min_vec, max_vec, num_utility_fns):
    utility_fns = [
        MonotonicUtility(
            torch.tensor(min_vec, dtype=torch.float32),
            torch.tensor(max_vec, dtype=torch.float32),
            frozen=True,
            scale_in=True,
            scale_out=True,
            max_weight=0.1,
            size_factor=1
        )
        for _ in range(num_utility_fns)
    ]
    return utility_fns


def save_utility_fns(utility_fns, u_dir='./utility_fns'):
    os.makedirs(u_dir, exist_ok=True)
    for i, utility_fn in enumerate(utility_fns):
        torch.save(utility_fn, f"{u_dir}/utility_fn_{i}.pt")


def load_utility_fns(u_dir='./utility_fns'):
    utility_fns = []
    num_utility_fns = len(os.listdir(u_dir))
    for i in range(num_utility_fns):
        utility_fn = torch.load(f"{u_dir}/utility_fn_{i}.pt")
        utility_fns.append(utility_fn)
    return utility_fns


def generate_utility_fns_per_environment(environments, num_utility_fns, top_u_dir='./utility_fns'):
    for env_id in environments:
        minimals, maximals, ref_point = get_bounding_box(env_id)
        nadir = np.min(minimals, axis=0)
        ideal = np.max(maximals, axis=0)
        u_fns = generate_utility_fns(nadir, ideal, num_utility_fns)
        u_dir = f"{top_u_dir}/{env_id}"
        save_utility_fns(u_fns, u_dir)
        print(f"Saved utility functions for {env_id}.")


if __name__ == '__main__':
    environments = ['deep-sea-treasure-concave-v0', 'mo-reacher-v4', 'minecart-v0']
    num_utility_fns = 100
    top_u_dir = './utility_fns'
    generate_utility_fns_per_environment(environments, num_utility_fns, top_u_dir)
