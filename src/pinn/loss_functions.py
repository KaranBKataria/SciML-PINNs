"""
This script defines all the loss functions that will be used as
part of training the PINNs. This includes the data-driven loss term,
as well as the physics-based and initial condition (IC) regularisation
terms, made into a single physics-based term by modifying the PINN output.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

# Import in all required packages and functions
import torch
import numpy as np
from tqdm import tqdm


# Define all loss functions
def data_loss_mse(pred: np.array, ground_truth: np.array) -> float:
    """
    This function computes the data-driven loss for a single training sample.

        Inputs:
                pred: The (modified) prediction of the PINN
                ground_truth: The ground truth vector from the training set

        Outputs:
                error: The loss for a single training sample
    """

    # Compute the square of the l2 norm between the prediction and the ground
    # truth, obtaining a single error
    error: float = np.linalg.norm(x=pred - ground_truth) ** 2

    return error


def physics_based_loss(
    phase_angle: torch.tensor,
    angular_frequency: torch.tensor,
    angular_acceleration: torch.tensor,
    inertia: torch.tensor,
    damping: torch.tensor,
    mechanical_power: torch.tensor,
    voltage_magnitude: torch.tensor,
    voltages: torch.tensor,
    phase_angles: torch.tensor,
    susceptances: torch.tensor,
    include_controllers: bool = False,
    **kwargs,
) -> float:
    """
    This function computes the physics-based regularisation term for a single training example.

        Inputs:
                phase_angle: The (modified) phase angle prediction of the PINN
                angular_frequency: The gradient of the (modified) phase angle prediction of the PINN
                angular_acceleration: The second-order time derivative of the phase angle - obtained
                      via automatic differenciation
                include_controllers: Boolean arguement to determine whether or not controllers should be included
                **kwargs: The controller coefficients (integral and proportional)

        Outputs:
                error: The physics-based loss for a single training example
    """

    # Compute the total electrical power output generator k supplies to the grid
    total_electrical_output = 0
    for v, delta, B in zip(voltages, phase_angles, susceptances):
        total_electrical_output += (
            B * voltage_magnitude * v * torch.sin(phase_angle - delta)
        )

    # Compute the cost based on whether or not PI controllers are accounted for
    if include_controllers:
        controller_proportional = kwargs["controller_proportional"]
        controller_integral = kwargs["integral"]

        cost: float = (
            (inertia * angular_acceleration)
            + (damping - controller_proportional) * angular_frequency
            + total_electrical_output
            - (controller_integral * phase_angle)
        )

        # No **2 to prevent invoking expensive microcode in each iteration of the training
        error: float = torch.mean(input=cost * cost)
        return error

    else:
        cost: float = (
            (inertia * angular_acceleration)
            + (damping * angular_frequency)
            + total_electrical_output
            - mechanical_power
        )

        # No **2 to prevent invoking expensive microcode in each iteration of the training
        error: float = torch.mean(input=cost * cost)
        return error


def IC_based_loss(model, initial_state: torch.tensor, device: str) -> float:
    """
    This function defines the IC regularisation term to ensure the training
    proceedure enforces the satisfaction of the pre-specified ICs.

        Inputs:
                initial_pred: The PINN prediction at t = 0
                initial_states: The pre-specified initial state of the system
                device: The device to move the tensors to (CPU or GPU)

        Output:
                error: The IC-based loss for a single training example
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
    phase_angle: torch.tensor,
    angular_frequency: torch.tensor,
    angular_acceleration: torch.tensor,
    inertia: torch.tensor,
    damping: torch.tensor,
    mechanical_power: torch.tensor,
    voltage_magnitude: torch.tensor,
    voltages: torch.tensor,
    phase_angles: torch.tensor,
    susceptances: torch.tensor,
    physics_weight: float,
    IC_weight: float,
    model,
    initial_state: torch.tensor,
    device: str,
    include_controllers: bool = False,
    **kwargs,
) -> float:
    """
    This function computes the total loss for a single training example, which is composed of the
    data-driven loss, the physics-based regularisation term and the initial condition regularisation
    term.

        Inputs:
                pred: The prediction of the PINN
                initial_pred: The PINN prediction at t = 0
                ground_truth: The ground truth vector from the training set
                initial_state: The pre-specified initial state of the system
                angular_acceleration: The second-order time derivative of the phase angle - obtained
                    via automatic differenciation
                physics_weight: The regularisation weight on the physics-based regularisation term
                IC_weight: The regularisation weight on the initial conditions regularisation term
                include_controllers: Boolean arguement to determine whether or not controllers should be included
                **kwargs: The controller coefficients (integral and proportional)

        Output:
                total_loss: The total loss for a single training example
    """
    assert physics_weight >= 0 or IC_weight >= 0, (
        "Regularisation penalty weights must be non-negative"
    )

    # Obtain the loss values for each component
    # data_loss: float = data_loss_mse(pred, ground_truth)
    physics_loss: float = physics_based_loss(
        phase_angle=phase_angle,
        angular_frequency=angular_frequency,
        angular_acceleration=angular_acceleration,
        inertia=inertia,
        damping=damping,
        mechanical_power=mechanical_power,
        voltage_magnitude=voltage_magnitude,
        voltages=voltages,
        phase_angles=phase_angles,
        susceptances=susceptances,
        include_controllers=include_controllers,
        **kwargs,
    )

    IC_loss: float = IC_based_loss(model=model, initial_state=initial_state, device=device)

    # Aggregate the components and scale by the regularisation weights
    total_loss: float = (physics_weight * physics_loss) + (IC_weight * IC_loss)

    return total_loss
