"""
This script defines the utility functions which are used in the numerical
experiment notebooks.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

from typing import Callable

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from loss_functions import *


def set_global_seed(seed: int) -> None:
    """
    Set global seed across all random elements in a script.

    Parameters
    ----------
    seed : int
        The seed value
    
    Returns
    --------
    None
    """

    # Set seeds
    # random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior if CUDA used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def absolute_max_gradient(gradient_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute absolute maximum element of gradient vector.

    Parameters
    ----------
    gradient_vec : torch.Tensor
        Gradient vector of a given loss term with respect to network parameters.

    Returns 
    -------
    abs_max_element : torch.Tensor
        Absolute maximum element of gradient vector.
    """

    flatten = [grad.clone().detach().abs().reshape(-1) for grad in gradient_vec if grad is not None]

    abs_max_element = torch.cat(flatten).max()

    return abs_max_element


def mean_gradient(gradient_vec: torch.Tensor) -> torch.Tensor:
    """
    Compute absolute mean element of gradient vector.

    Parameters
    ----------
    gradient_vec : torch.Tensor
        Gradient vector of a given loss term with respect to network parameters.

    Returns 
    -------
    abs_mean_element : torch.Tensor
        Absolute mean element of gradient vector.
    """

    flatten = [grad.clone().detach().abs().reshape(-1) for grad in gradient_vec if grad is not None]

    abs_mean_element = torch.cat(flatten).mean()

    return abs_mean_element


def grad_norm(model: Callable, loss: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 (Euclidean) norm of the gradient vector.

    Parameters
    ----------
    loss : torch.Tensor
        Loss term to compute gradient norm of with respect to network parameters.
    model : Callable
        NN model (PINN or Vanilla NN).
        
    Returns 
    -------
    l2_norm_grad : torch.Tensor
        L2/Euclidean norm of the gradient vector.
    """

    loss = loss.clone()

    # Compute gradient vector with respect to model (network) parameters 
    grad_loss = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=False)
    flatten = [grad.clone().detach().reshape(-1) for grad in grad_loss if grad is not None]

    # Compute L2 norm
    l2_norm_grad = torch.cat(flatten).norm(2)

    return l2_norm_grad


def loss_grad(model: Callable, loss: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient vector of loss term with respect to model (network) parameters.

    Parameters
    ----------
    loss : torch.Tensor
        Loss term to compute gradient of with respect to network parameters.
    model : Callable
        NN model (PINN or Vanilla NN).
        
    Returns 
    -------
    gradient_vector : torch.Tensor
        Gradient vector of loss with respect to model weights.
    """

    loss = loss.clone()

    # Compute gradient vector with respect to model (network) parameters 
    grad_loss = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=False)
    flatten = [grad.clone().detach().reshape(-1) for grad in grad_loss if grad is not None]

    gradient_vector = torch.cat(flatten)

    return gradient_vector


if __name__ == "__main__":
    pass

########################################################################################
#                               DEPRECATED FUNCTIONS
########################################################################################

# def hessian_based_2D_loss_topology(
#     eigenvectors: list['torch.Tensor'],
#     loss: Callable,
#     scalar_1: 'np.ndarray',
#     scalar_2: 'np.ndarray',
#     minimiser: 'torch.Tensor'
#     ) -> 'np.ndarray':

#     vec1 = eigenvectors[0].detach().numpy()
#     vec2 = eigenvectors[1].detach().numpy()

#     perturbed_parameters = \
#         minimiser.detach().numpy() \
#         + scalar_1 * vec1 \
#         + scalar_2 * vec2

#     losses = []

#     for params in perturbed_parameters:
#         losses.append(loss(params))

#     return np.array(losses)


# def get_esd_plot(eigenvalues, weights):
#     plt.xticks(fontsize=12)
#     density, grids = density_generate(eigenvalues, weights)
#     plt.semilogy(grids, density + 1.0e-7)
#     plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
#     plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
#     plt.yticks(fontsize=12)
#     plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
#     plt.tight_layout()


# def density_generate(
#   eigenvalues: list[float],
#   weights: listp[float],
#   num_bins: int = 10000,
#   sigma_squared: float = 1e-5,
#   overhead: float = 0.01):

#     eigenvalues = np.array(eigenvalues)
#     weights = np.array(weights)

#     lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
#     lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

#     grids = np.linspace(lambda_min, lambda_max, num=num_bins)
#     sigma = sigma_squared * max(1, (lambda_max - lambda_min))

#     num_runs = eigenvalues.shape[0]
#     density_output = np.zeros((num_runs, num_bins))

#     for i in range(num_runs):
#         for j in range(num_bins):
#             x = grids[j]
#             tmp_result = gaussian(eigenvalues[i, :], x, sigma)
#             density_output[i, j] = np.sum(tmp_result * weights[i, :])
#     density = np.mean(density_output, axis=0)
#     normalization = np.sum(density) * (grids[1] - grids[0])
#     density = density / normalization
#     return density, grids


# def gaussian(x: float, x0: float, sigma_squared: float) -> float:
#     return np.exp(-(x0 - x)**2 /
#                   (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)


# def R3_algorithm(collocation_set: torch.Tensor, model: Callable, **kwargs) -> torch.Tensor:
#     assert collocation_set.requires_grad is True

#     residual_array = []

#     model.eval()

#     for t in collocation_set:
#         t_fresh = t.clone().detach().requires_grad_(True)  # New leaf node with grad

#         phase_angle_pred = model.forward(data=t_fresh)

#         angular_frequency_pred = torch.autograd.grad(
#             outputs=phase_angle_pred,
#             inputs=t_fresh,
#             grad_outputs=torch.ones_like(phase_angle_pred),
#             create_graph=True,
#             retain_graph=True
#         )[0]

#         angular_acceleration_pred = torch.autograd.grad(
#             outputs=angular_frequency_pred,
#             inputs=t_fresh,
#             grad_outputs=torch.ones_like(angular_frequency_pred),
#             create_graph=True,
#             retain_graph=True
#         )[0]

#         swing_inputs = SwingEquationInputs(
#             phase_angle=phase_angle_pred,
#             angular_frequency=angular_frequency_pred,
#             angular_acceleration=angular_acceleration_pred,
#             inertia=INERTIA,
#             damping=kwargs["DAMPING"],
#             mechanical_power=MECHANICAL_POWER,
#             voltage_magnitude=VOLTAGE,
#             voltages=VOLTAGES,
#             phase_angles=PHASE_ANGLES,
#             susceptances=SUSCEPTANCES,
#             controller_proportional=None,
#             controller_integral=None,
#         )

#         residual = physics_based_loss(swing_inputs=swing_inputs, include_controllers=False)
#         residual_array.append(torch.abs(residual).detach())
        
#     residual_tensor = torch.stack(residual_array)
#     threshold = torch.mean(residual_tensor)

#     # Retain points above threshold
#     retained = collocation_set[residual_tensor > threshold]
#     retained = retained.clone().detach().requires_grad_(True) 

#     # Resample new ones
#     n_to_sample = N_C - retained.shape[0]
#     new_samples = (T0 - FINALTIME) * torch.rand((n_to_sample, 1), device=collocation_set.device, requires_grad=True) + FINALTIME

#     collocation_next_epoch = torch.cat([retained, new_samples], dim=0)
#     collocation_next_epoch = collocation_next_epoch.clone().detach().requires_grad_(True)  # Final safety

#     print(collocation_set)
#     print(threshold)
#     print(residual_tensor)
#     print(retained)
#     print(collocation_next_epoch)

#     assert collocation_next_epoch.shape == (N_C, 1)
#     assert collocation_next_epoch.requires_grad is True
    
#     model.train()
#     return collocation_next_epoch


# def FFE(original_input: torch.Tensor, frequency_vector: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the Fourier Feature Embedding input.

#     Parameters
#     ----------
#     original_input : torch.Tensor
#         The original network input
#     frequency_vector : torch.Tensor
#         Frequency vector for pre-multiplication of the original input

#     Returns 
#     -------
#     FFE : torch.Tensor
#         The Fourier Feature Embedding (FFE) to get input into the network
#     """

#     m = frequency_vector.shape[0]

#     cos_feature = torch.cos(input=frequency_vector @ original_input.T)
#     sin_feature = torch.sin(input=frequency_vector @ original_input.T)

#     FFE = torch.cat((original_input.T, cos_feature, sin_feature), dim=0)

#     assert FFE.shape[0] == 2*m + 1
#     assert FFE.requires_grad is True

#     return FFE.T