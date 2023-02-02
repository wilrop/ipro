import numpy as np
from sortedcontainers import SortedKeyList

from pareto import pareto_dominates
from patch2d import Patch2d
from vis import plot_patches, create_gif


def outer_loop(problem, inner_loop, linear_solver, num_objectives=2, tolerance=1e-6, save_figs=False, log_dir=None):
    """Implement the outer loop for the algorithm.

    Args:
        problem (Problem): A problem instance.
        inner_loop (function): The inner loop function.
        linear_solver (function): The linear solver function for the extremities.
        num_objectives (int, optional): The number of objectives. Defaults to 2.
        tolerance (float, optional): The tolerance for outer loop to exit. Defaults to 1e-6.
        save_figs (bool, optional): Whether to save the figures. Defaults to False.
        log_dir: The directory to save the figures. Defaults to None.

    Returns:
        list: The Pareto front.
    """
    pf = []

    for obj in range(num_objectives):
        weights = np.zeros(num_objectives)
        weights[obj] = 1
        point = linear_solver(problem, weights)
        pf.append(point)

    if pareto_dominates(pf[0], pf[1]):
        pf = [pf[0], pf[0]]
    elif pareto_dominates(pf[1], pf[0]):
        pf = [pf[1], pf[1]]

    start_patch = Patch2d(*pf)
    patches_queue = SortedKeyList([start_patch], key=lambda x: x.area)
    step = 0

    while patches_queue:
        if save_figs:
            plot_patches(patches_queue, pf, problem, log_dir, f'patches_{step}')
        patch = patches_queue.pop()

        target = patch.get_intersection_point()
        pareto_optimal_vec = inner_loop(problem, target)
        if not patch.on_rectangle(pareto_optimal_vec):  # Check that a new Pareto optimal point was found.
            pf.append(pareto_optimal_vec)
            new_patch1, new_patch2 = patch.split(pareto_optimal_vec)

            if new_patch1.area > tolerance:
                patches_queue.add(new_patch1)

            if new_patch2.area > tolerance:
                patches_queue.add(new_patch2)

        step += 1

    if save_figs:
        plot_patches(patches_queue, pf, problem, log_dir, f'patches_{step}')
        create_gif(log_dir, 'algorithm')

    return pf
