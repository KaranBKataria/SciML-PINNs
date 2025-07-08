from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch
    import numpy as np

def hessian_based_2D_loss_topology(
    eigenvectors: list['torch.Tensor'],
    loss: Callable,
    scalar_1: 'np.ndarray',
    scalar_2: 'np.ndarray',
    minimiser: 'torch.Tensor'
    ) -> 'np.ndarray':

    vec1 = eigenvectors[0].detach().numpy()
    vec2 = eigenvectors[1].detach().numpy()

    perturbed_parameters = \
        minimiser.detach().numpy() \
        + scalar_1 * vec1 \
        + scalar_2 * vec2

    losses = []

    for params in perturbed_parameters:
        losses.append(loss(params))

    return np.array(losses)

loss = torch.nn.MSELoss()
