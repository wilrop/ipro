from itertools import product, combinations, chain


def construct_set(dimensions):
    all_groups = []
    indices = list(range(dimensions))
    options = [0, 1]

    for i in range(2, dimensions + 1):
        static_indexes = list(combinations(indices, i))
        static_part = list(product(options, repeat=i))
        non_static_part = list(product(options, repeat=dimensions - i))
        for static_index in static_indexes:
            for static in static_part:
                group = []
                for non_static in non_static_part:
                    patch = [0] * dimensions
                    for index, value in zip(static_index, static):
                        patch[index] = value
                    non_static_index = 0
                    for index in indices:
                        if index not in static_index:
                            value = non_static[non_static_index]
                            non_static_index += 1
                            patch[index] = value
                    group.append(tuple(patch))
                all_groups.append(set(group))

    selected_groups = []
    dominated_patch = tuple([0] * dimensions)
    dominating_patch = tuple([1] * dimensions)
    for group in all_groups:
        if not (dominating_patch in group or dominated_patch in group):
            selected_groups.append(group)

    return selected_groups


def construct_universe(dimensions):
    patches = set(product([0, 1], repeat=dimensions))
    patches.remove((0,) * dimensions)
    patches.remove((1,) * dimensions)
    return patches


def brute_force_minal_cover(dimensions, universe, sets, overlap_allowed=False):
    prev_size = 0
    for subsets in chain.from_iterable(combinations(sets, r) for r in range(dimensions, len(sets) + 1)):
        if len(subsets) > prev_size:
            print(f'Starting size: {len(subsets)}')
            prev_size = len(subsets)
        cover = set().union(*subsets)
        if universe == cover:
            if overlap_allowed:
                return len(subsets), subsets
            else:
                if max(check_patches_overlap(universe, subsets)) == 1:
                    return len(subsets), subsets
    else:
        raise ValueError("No cover found")


def check_patches_overlap(universe, minimal_cover):
    memberships = []
    for patch in universe:
        membership = 0
        for group in minimal_cover:
            if patch in group:
                membership += 1
        memberships.append(membership)

    return memberships


dimensions = 5
universe = construct_universe(dimensions)
print(f'Universe size = {len(universe)} - Theoretical size = {2 ** dimensions - 2}')
groups = construct_set(dimensions)
print(f'Number of groups = {len(groups)}')
len_min_cover, minimal_cover = brute_force_minal_cover(dimensions, universe, groups)
print(f'Minimal cover size = {len_min_cover}')
print(f'Minimal cover = {minimal_cover}')
print(f'Membership counts = {check_patches_overlap(universe, minimal_cover)}')
