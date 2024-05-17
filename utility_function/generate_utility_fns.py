import os
import pickle
import numpy as np

from environments.bounding_boxes import get_bounding_box
from game_generators.functions.monotonic import increasing
from game_generators.functions.concave import concave


def generate_utility_fns(
        min_vec,
        max_vec,
        num_utility_fns,
        num_points=6,
        max_grad=5,
        fn_type='concave',
        seed=None
):
    dim = len(min_vec)
    if fn_type == 'concave':
        utility_fns = concave(
            dim,
            min_vec=min_vec,
            max_vec=max_vec,
            min_y=0,
            max_y=1,
            batch_size=num_utility_fns,
            batched=True,
            num_points=num_points,
            seed=seed
        )
    elif fn_type == 'increasing_cumsum':
        utility_fns = increasing(
            dim,
            max_grad=max_grad,
            method="cumsum",
            min_vec=min_vec,
            max_vec=max_vec,
            min_y=0,
            max_y=1,
            batch_size=num_utility_fns,
            batched=True,
            num_points=num_points,
            seed=seed
        )
    elif fn_type == 'increasing_max_add':
        utility_fns = increasing(
            dim,
            max_grad=max_grad,
            method="max_add",
            min_vec=min_vec,
            max_vec=max_vec,
            min_y=0,
            max_y=1,
            batch_size=num_utility_fns,
            batched=True,
            num_points=num_points,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown function type: {fn_type}")
    return utility_fns


def save_utility_fns(utility_fns, u_dir='./utility_fns'):
    os.makedirs(u_dir, exist_ok=True)
    for i, utility_fn in enumerate(utility_fns):
        with open(f"{u_dir}/utility_fn_{i}.pkl", 'wb') as f:
            pickle.dump(utility_fn, f)


def load_utility_fns(u_dir='./utility_fns'):
    utility_fns = []
    file_names = os.listdir(u_dir)
    for file_name in file_names:
        if file_name.endswith('.pkl'):
            with open(f"{u_dir}/{file_name}", 'rb') as f:
                utility_fn = pickle.load(f)
            utility_fns.append(utility_fn)
    return utility_fns


def save_utility_fns_per_environment(
        environments,
        num_utility_fns,
        fn_type='concave',
        num_points=6,
        max_grad=5,
        top_u_dir='./utility_fns',
        seed=None
):
    for env_id in environments:
        print(f'Generating utility functions for {env_id}.')
        minimals, maximals, ref_point = get_bounding_box(env_id)
        nadir = np.min(minimals, axis=0)
        ideal = np.max(maximals, axis=0)
        u_fns = generate_utility_fns(
            nadir,
            ideal,
            num_utility_fns,
            fn_type=fn_type,
            num_points=num_points,
            max_grad=max_grad,
            seed=seed
        )
        u_dir = os.path.join(top_u_dir, fn_type, env_id)
        save_utility_fns(u_fns, u_dir)
        print(f"Saved utility functions")
        print('---------------------------------')


if __name__ == '__main__':
    environments = ['deep-sea-treasure-concave-v0', 'mo-reacher-v4', 'minecart-v0']
    num_utility_fns = 1000
    top_u_dir = './utility_fns'
    fn_type = 'increasing_cumsum'
    num_points = 6
    max_grad = 5
    seed = 0
    save_utility_fns_per_environment(
        environments,
        num_utility_fns,
        fn_type=fn_type,
        top_u_dir=top_u_dir,
        seed=seed
    )
