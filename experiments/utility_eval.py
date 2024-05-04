import torch


def generalised_maximum_utility_loss(front, reference_set, utility_fns):
    """Compute the maximum utility loss for a front and utility functions wrt a reference set."""
    utility_losses = []
    for utility_fn in utility_fns:
        front_utilities = utility_fn(reference_set)  # Compute the utility for the front
        approx_utilities = utility_fn(front)  # Compute the utility for the approximate front
        max_utility_loss = torch.max(front_utilities) - torch.max(approx_utilities)  # Compute the utility loss.
        utility_losses.append(max_utility_loss)
    return torch.max(torch.stack(utility_losses))


def generalised_expected_utility(front, utility_fns):
    """Compute the expected utility for the set of utility functions when taking vectors from the front."""
    utilities = [torch.max(utility_fn(front)) for utility_fn in utility_fns]
    return torch.mean(torch.stack(utilities))
