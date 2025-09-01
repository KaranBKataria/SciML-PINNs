import sys

# Add ODE_numerical_solver module to PATH variable
sys.path.insert(
    1,
    "/Users/karankataria/Library/CloudStorage/OneDrive-UniversityofWarwick/dissertation_code/src/data"
)

import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import qmc
from torchinfo import summary
from utils import set_global_seed
from pinn_architecture import PINN
from loss_functions import *
from global_constants import *
from ODE_numerical_solver import swing_ODEs_solver


def physics_based_loss(model: Callable, input: torch.Tensor) -> float:
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

    phase_angle_pred = model.forward(data=input)

    angular_frequency_pred = torch.autograd.grad(
        outputs=phase_angle_pred,
        inputs=input,
        grad_outputs=torch.ones_like(phase_angle_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    angular_acceleration_pred = torch.autograd.grad(
        outputs=angular_frequency_pred,
        inputs=input,
        grad_outputs=torch.ones_like(angular_frequency_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute the total electrical power output generator k supplies to the grid
    total_electrical_output = 0
    for v, delta, B in zip(
        VOLTAGES, PHASE_ANGLES, SUSCEPTANCES
    ):
        total_electrical_output += (
            B
            * VOLTAGE
            * v
            * torch.sin(phase_angle_pred - delta)
        )

    # # Compute the cost based on whether or not PI controllers are accounted for
    # if include_controllers:
    #     cost: float = (
    #         (swing_inputs.inertia * swing_inputs.angular_acceleration)
    #         + (swing_inputs.damping - swing_inputs.controller_proportional)
    #         * swing_inputs.angular_frequency
    #         + total_electrical_output
    #         - (swing_inputs.controller_integral * swing_inputs.phase_angle)
    #     )

    #     # No **2 to prevent invoking expensive microcode in each iteration of the training
    #     error: float = torch.mean(input=cost * cost)
    #     return error

    cost: float = (
        (INERTIA * angular_acceleration_pred)
        + (DAMPING * angular_frequency_pred)
        + total_electrical_output
        - MECHANICAL_POWER
    )

    # No **2 to prevent invoking expensive microcode in each iteration of the training
    error: float = torch.mean(input=cost ** 2)
    return error


EPOCHS = 5_000
EPOCHS_ADAM = 4900

N_C: int = 1_000  
LEARNING_RATE: float = 0.01
PATIENCE: int = 10
SCHEDULER_FACTOR: float = 0.9
ACTIVATION: str = "tanh"
HISTORY: int = 100
LEARNING_RATE_LBFGS: float = 1.0

TOTAL_NUM_PARAMS: int = summary(model=PINN(activation=ACTIVATION).to(device="cpu")).total_params

SEED: int = 0
set_global_seed(SEED)

# DAMPING: torch.Tensor = torch.tensor(data=[[0.0015]])
INERTIA: torch.Tensor = torch.tensor(data=[[0.25]])
MECHANICAL_POWER: torch.Tensor = torch.tensor(data=[[0.13]])
IC_WEIGHT: torch.Tensor = torch.tensor(data=[[1.0]])
PHYSICS_WEIGHT: torch.Tensor = torch.tensor(data=[[1.0]])
TIMESTEP_FLOAT = 0.1
INITIAL_STATE: torch.tensor = torch.tensor(
    data=np.array([0.1, 0.1]), dtype=torch.float32
).to(device="cpu")

LHC = qmc.LatinHypercube(d=1, seed=SEED)
collocation_points = LHC.random(n=N_C)
collocation_points = qmc.scale(
    collocation_points, T0, FINALTIME
).flatten()  # Scale from a unit interval [0,1] (default) to [t0,T]

collocation_points: torch.tensor = torch.tensor(
    data=collocation_points[:, None].astype(np.float32), requires_grad=True
).to(device="cpu")

print(torch.sum(collocation_points))

PARAM_LIST = [0.00015, 0.0015, 0.015, 0.15, 1.5]

# solution, noisy_solution, numerical_times = swing_ODEs_solver(
#     initial_time=T0,
#     initial_state=INITIAL_STATE.detach().numpy(),
#     final_time=FINALTIME,
#     timestep=TIMESTEP_FLOAT,
#     inertia=INERTIA.item(),
#     damping=DAMPING.item(),
#     mechanical_power=MECHANICAL_POWER.item(),
#     voltage_magnitude=VOLTAGE.item(),
#     include_controllers=False,
#     voltages=np.array([VOLTAGES.item()]),
#     phase_angles=np.array([PHASE_ANGLES.item()]),
#     susceptances=np.array([SUSCEPTANCES.item()]),
#     file_name="test_run",
#     save_output_to_file=False,
#     controller_proportional=0.05,
#     controller_integral=0.1
# )

# times_tensor = torch.tensor(numerical_times[:, None].astype(np.float32), requires_grad=True).to(device="cpu")

pinn_models = []
gradient_norm_per_param = []

for param in PARAM_LIST:

    set_global_seed(SEED)

    DAMPING = torch.tensor(data=[[param]])

    # Define PINN, optimiser and learning rate scheduler
    pinn = PINN(activation=ACTIVATION).to(device="cpu")

    total_sum = sum(p.sum().item() for p in pinn.parameters())
    print(f"Sum of all weights: {total_sum:.6f}")

    # Instantiate the Adam optimiser and learning rate scheduler
    optimiser_adam = torch.optim.Adam(params=pinn.parameters(), lr=LEARNING_RATE)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimiser_adam, patience=PATIENCE, factor=SCHEDULER_FACTOR
    )

    optimiser_lbfgs = torch.optim.LBFGS(
            params=pinn.parameters(),
            lr=LEARNING_RATE_LBFGS,
            history_size=HISTORY,
            line_search_fn="strong_wolfe",
            max_iter=100000,
            max_eval=100000,
            tolerance_change= np.finfo(float).eps
        )

    def closure():
        optimiser_lbfgs.zero_grad()
        residual_loss = physics_based_loss(model=pinn, input=collocation_points)
        ic_loss = IC_based_loss(model=pinn, initial_state=INITIAL_STATE, device="cpu")
        loss = PHYSICS_WEIGHT*residual_loss + IC_WEIGHT*ic_loss
        loss.backward()
        # total_norm = 0
        # for w_n_b in pinn.parameters():
        #     if w_n_b.grad is not None:
        #         param_norm = w_n_b.grad.detach().data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # gradient_norm.append(total_norm)
        return loss
    
    gradient_norm = []

    for epoch in (range(1, EPOCHS+1)):

        residual_loss = physics_based_loss(model=pinn, input=collocation_points)

        ic_loss = IC_based_loss(model=pinn, initial_state=INITIAL_STATE, device="cpu")

        loss = PHYSICS_WEIGHT*residual_loss + IC_WEIGHT*ic_loss

        # Backpropogate using reverse/backward-mode AD
        if epoch <= EPOCHS_ADAM:
            optimiser_adam.zero_grad()
            loss.backward()

            total_norm = 0
            total_num_params = 0
            for w_n_b in pinn.parameters():
                if w_n_b.grad is not None:
                    total_num_params += w_n_b.flatten().shape[0]
                    # print(f"This is w_n_b.grad.detach().data: {w_n_b.grad.detach()}")
                    param_norm = w_n_b.grad.detach().norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norm.append(total_norm)

            assert total_num_params == TOTAL_NUM_PARAMS

            optimiser_adam.step()
            lr_scheduler.step(metrics=loss)

            if epoch % 1_000 == 0:
                print(f"param={param:.2e}\t\tEpoch: {epoch}\t\tTraining loss: {loss.item()}")
        else:
            optimiser_lbfgs.step(closure=closure)

            if epoch % 1_000 == 0:
                print(f"param={param:.2e}\t\tEpoch: {epoch}\t\tRunning L-BFGS")

    pinn_models.append(pinn)
    gradient_norm_per_param.append(gradient_norm)

# fig, ax = plt.subplots(2, len(PARAM_LIST))

# for idx, (model, param) in enumerate(zip(pinn_models, PARAM_LIST)):

#     DAMPING = torch.tensor(data=[[param]])

#     solution, _, numerical_times = swing_ODEs_solver(
#     initial_time=T0,
#     initial_state=INITIAL_STATE.detach().numpy(),
#     final_time=FINALTIME,
#     timestep=TIMESTEP_FLOAT,
#     inertia=INERTIA.item(),
#     damping=DAMPING.item(),
#     mechanical_power=MECHANICAL_POWER.item(),
#     voltage_magnitude=VOLTAGE.item(),
#     include_controllers=False,
#     voltages=np.array([VOLTAGES.item()]),
#     phase_angles=np.array([PHASE_ANGLES.item()]),
#     susceptances=np.array([SUSCEPTANCES.item()]),
#     file_name="test_run",
#     save_output_to_file=False,
#     controller_proportional=0.05,
#     controller_integral=0.1
#     )

#     times_tensor = torch.tensor(numerical_times[:, None].astype(np.float32), requires_grad=True).to(device="cpu")

#     pinn = model
#     pinn.eval()
#     pred = pinn.forward(data=times_tensor)
#     pred_dot = torch.autograd.grad(
#             outputs=pred,
#             inputs=times_tensor,
#             grad_outputs=torch.ones_like(pred),
#             create_graph=False,
#             retain_graph=False
#     )[0]

#     ax[0, idx].plot(numerical_times, pred.detach().numpy(), label="PINN")
#     ax[1, idx].plot(numerical_times, pred_dot.detach().numpy(), label="PINN")

#     ax[0, idx].plot(numerical_times, solution[0,:], label="True Dynamics")
#     ax[1, idx].plot(numerical_times, solution[1,:], label="True Dynamics")

#     ax[0, idx].legend(loc="best")
#     ax[1, idx].legend(loc="best")


fig, ax = plt.subplots()

for grad_norm in gradient_norm_per_param:
    ax.semilogx(range(1, EPOCHS_ADAM+1), grad_norm)

plt.show()