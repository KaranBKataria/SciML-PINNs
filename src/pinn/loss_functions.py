"""
This script defines all the loss functions that will be used as
part of training the PINNs. This includes the data-driven loss term,
as well as the physics-based and initial condition (IC) regularisation
terms, made into a single physics-based term by modifying the PINN output.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

from typing import NamedTuple, Callable

import torch
import numpy as np


class SwingEquationInputs(NamedTuple):
    phase_angle: torch.Tensor
    angular_frequency: torch.Tensor
    angular_acceleration: torch.Tensor
    inertia: torch.Tensor
    damping: torch.Tensor
    mechanical_power: torch.Tensor
    voltage_magnitude: torch.Tensor
    voltages: torch.Tensor
    phase_angles: torch.Tensor
    susceptances: torch.Tensor
    controller_proportional: torch.Tensor
    controller_integral: torch.Tensor


# Define all loss functions
def l2_error(pred: torch.Tensor, ground_truth: torch.Tensor, dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the per-sample L2 absolute and relative test errors.

    Parameters
    ----------
    pred : torch.Tensor
        PINN prediction for the phase angle (PINN) and angular frequency (AD).
    ground_truth : torch.Tensor
        Ground truth vector from the training set.

    Returns
    -------
    l2_abs : torch.Tensor
        Loss for a single training sample.
    l2_rel : torch.Tensor
    """
    # Compute the square of the l2 norm between the prediction and the ground
    # truth, obtaining a single error
    l2_abs: float = torch.norm(input=pred - ground_truth, dim=dim)
    l2_rel = l2_abs / torch.norm(input=ground_truth, dim=dim)
    
    return torch.mean(l2_abs), torch.mean(l2_rel)


def physics_based_loss(
    swing_inputs: SwingEquationInputs, include_controllers: bool = False
) -> float:
    """
    Computes ODE residual regularisation term for a single training example.

    Parameters
    ----------
    swing_inputs : NamedTuple
        NamedTuple of ODE parameters, solution and deriatives
    include_controllers : bool
        Boolean arguement to determine whether or not controllers should be included

    Returns
    -------
    error: The physics-based loss for a single training example
    """

    # Compute the total electrical power output generator k supplies to the grid
    total_electrical_output = 0
    for v, delta, B in zip(
        swing_inputs.voltages, swing_inputs.phase_angles, swing_inputs.susceptances
    ):
        total_electrical_output += (
            B
            * swing_inputs.voltage_magnitude
            * v
            * torch.sin(swing_inputs.phase_angle - delta)
        )

    # Compute the cost based on whether or not PI controllers are accounted for
    if include_controllers:
        cost: float = (
            (swing_inputs.inertia * swing_inputs.angular_acceleration)
            + (swing_inputs.damping - swing_inputs.controller_proportional)
            * swing_inputs.angular_frequency
            + total_electrical_output
            - (swing_inputs.controller_integral * swing_inputs.phase_angle)
        )

        # No **2 to prevent invoking expensive microcode in each iteration of the training
        error: float = torch.mean(input=cost * cost)
        return error

    else:
        cost: float = (
            (swing_inputs.inertia * swing_inputs.angular_acceleration)
            + (swing_inputs.damping * swing_inputs.angular_frequency)
            + total_electrical_output
            - swing_inputs.mechanical_power
        )

        # No **2 to prevent invoking expensive microcode in each iteration of the training
        error: float = torch.mean(input=cost * cost)
        return error


def IC_based_loss(model: Callable, initial_state: torch.Tensor, device: str) -> float:
    """
    Computes IC loss/regularisation term for pre-specified ICs
    for a single training example.

    Parameters
    ----------
    model : Callable
        The PINN
    initial_states : torch.Tensor
        Pre-specified initial state of the system
    device : str
        Device to move the tensors to (CPU or GPU)

    Returns
    -------
    error : float
        IC-based loss for a single training example
    """

    # Compute the square of the l2 norm between the PINN prediction at t=0 and the
    # initial state

    t0 = torch.tensor(data=[[0.0]], requires_grad=True).to(device=device)
    initial_phase_angle = model.forward(t0, initial_state)

    initial_angular_frequency = torch.autograd.grad(
        outputs=initial_phase_angle,
        inputs=t0,
        grad_outputs=torch.ones_like(initial_phase_angle),
        create_graph=True,
    )[0]

    initial_pred = torch.cat(
        tensors=[initial_phase_angle, initial_angular_frequency], dim=1
    )

    error: float = torch.norm(input=initial_pred - initial_state) ** 2

    return error


def total_loss(
    swing_inputs: SwingEquationInputs,
    physics_weight: float,
    IC_weight: float,
    model: Callable,
    initial_state: torch.Tensor,
    device: str,
    include_controllers: bool = False,
) -> float:
    """
    This function computes the total loss for a single training example, which is composed of the
    data-driven loss, the physics-based regularisation term and the initial condition regularisation
    term.

    Parameters
    ----------
    swing_inputs : NamedTuple
        NamedTuple of ODE parameters, solution and deriatives
    model : Callable
        The PINN
    initial_state : torch.Tensor
        Pre-specified initial state of the system
    device : str
        Device to move the tensors to (CPU or GPU)
    physics_weight : float
        The regularisation weight on the physics-based regularisation term
    IC_weight : float
        The regularisation weight on the initial conditions regularisation term
    include_controllers : bool
        Boolean arguement to determine whether or not controllers should be included

    Returns
    -------
    total_loss : float
        Total loss for a single training example
    """
    assert physics_weight >= 0 or IC_weight >= 0, "Regularisation penalty weights must be non-negative"

    # Obtain the loss values for each component
    # data_loss: float = data_loss_mse(pred, ground_truth)
    physics_loss: float = physics_based_loss(
        swing_inputs=swing_inputs,
        include_controllers=include_controllers,
    )

    IC_loss: float = IC_based_loss(
        model=model, initial_state=initial_state, device=device
    )

    # Aggregate the components and scale by the regularisation weights
    total_loss: float = (physics_weight * physics_loss) + (IC_weight * IC_loss)

    return total_loss

def loss_closure(
    swing_inputs: SwingEquationInputs,
    physics_weight: float,
    IC_weight: float,
    model: Callable,
    initial_state: torch.Tensor,
    device: str,
    include_controllers: bool = False,
    ) -> float:
    
    def loss_function(inputs: torch.Tensor, targets: torch.Tensor) -> float:
        return total_loss(
            swing_inputs=swing_inputs,
            physics_weight=physics_weight,
            IC_weight=IC_weight,
            model=model,
            initial_state=initial_state,
            device=device,
            include_controllers=include_controllers,
            )
    
    return loss_function