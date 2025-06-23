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

def physics_based_loss(pred: np.array, angular_acceleration: float, include_controllers: bool = False, **kwargs) -> float:
    """
    This function computes the physics-based regularisation term for a single training example.

        Inputs:
                pred: The (modified) prediction of the PINN
                angular_acceleration: The second-order time derivative of the phase angle - obtained
                      via automatic differenciation
                include_controllers: Boolean arguement to determine whether or not controllers should be included
                **kwargs: The controller coefficients (integral and proportional)

        Outputs:
                error: The physics-based loss for a single training example
    """

    # Extract the phase angle and angular frequency PINN predictions
    phase_angle = pred[0]
    angular_frequency = pred[1]

    # Compute the total electrical power output generator k supplies to the grid
    total_electrical_output = 0
    for v, delta, B in zip(voltages, phase_angles, susceptances):
        total_electrical_output += B * voltage_magnitude * v * np.sin(phase_angle - delta)
    
    # Compute the cost based on whether or not PI controllers are accounted for
    if include_controllers:
        controller_proportional = kwargs['controller_proportional']
        controller_integral = kwargs['controller_intergral']

        cost: float = (inertia * angular_acceleration) + (damping - controller_proportional)\
                * angular_frequency + total_electrical_output - (controller_integral * phase_angle) 

    else:
        cost: float = (inertia * angular_acceleration) + (damping * angular_frequency) + total_electrical_output - mechanical_power 

    # No **2 to prevent invoking expensive microcode in each iteration of the training
    error: float = cost * cost

    return error

def modified_PINN_pred(time: float, initial_state: np.array, pred: np.array) -> np.array:
    """
    This function returns the modified PINN prediction such that both the
    physics-based loss and the IC terms are satisfied simultaneously.

        Inputs:
                time: The input time into the PINN
                initial_states: The pre-specified initial state of the system
                pred: The (modified) prediction of the PINN

        Output:
                pred_modified: The modified PINN prediction
    """

    # Define the modified prediction 
    pred_modified = initial_state + (time * pred)

    return pred_modified

def IC_based_loss(initial_pred: np.array, initial_state: np.array) -> float:
    """
    This function defines the IC regularisation term to ensure the training
    proceedure enforces the satisfaction of the pre-specified ICs.

        Inputs:
                initial_pred: The PINN prediction at t = 0
                initial_states: The pre-specified initial state of the system

        Output:
                error: The IC-based loss for a single training example
    """
    
    # Compute the square of the l2 norm between the PINN prediction at t=0 and the
    # initial state
    error: float = np.linalg.norm(x=initial_pred - initial_state) ** 2

    return error

def total_loss(
        pred: np.array, initial_pred: np.array, ground_truth: np.array, initial_state: np.array,
        angular_acceleration: float, physics_weight: float,
        #IC_weight: float,
        include_controllers: bool = False, **kwargs) -> float:
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
    assert physics_weight < 0 or IC_weight < 0, "Regularisation penalty weights must be non-negative"

    # Obtain the loss values for each component
    data_loss: float = data_loss_mse(pred, ground_truth)
    physics_loss: float = physics_based_loss(pred, angular_acceleration, include_controllers, kwargs)
    #IC_loss: float = IC_based_loss(initial_pred, initial_state)

    # Aggregate the components and scale by the regularisation weights
    total_loss: float = data_loss + (physics_weight * physics_loss) #+ (IC_weight * IC_loss)

    return total_loss
