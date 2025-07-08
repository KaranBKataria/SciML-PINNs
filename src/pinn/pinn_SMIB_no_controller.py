"""
This script defines the architecture of, trains, evaluates and outputs
the performance of physics-informed neural networks to obtain
solutions for the transient dynamics of a synchronous generator for every
numerical solution obtained. That is, it will be trained to find the solution
for the swing equation. The solutions will be benchmarked against the true
dynamics obtained from the RK45 numerical solutions.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

from pathlib import Path
from os import listdir
from re import findall

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import qmc
from tqdm import tqdm
import scienceplots

from loss_functions import *
from pinn_architecture import PINN


# Config matplotlib
plt.style.use("science")
plt.rcParams["text.usetex"] = False

########################################################################################
# Define constants
########################################################################################

# Move tensors and models to GPU
DEVICE: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print the training loss every specified number of epochs
PRINT_TRAINING_LOSS_EVERY_EPOCH: int = 100

# Define swing equation constants
MECHANICAL_POWER = torch.tensor(0.13)
VOLTAGE = torch.tensor(1.0)
VOLTAGES = torch.tensor([1.0])
SUSCEPTANCES = torch.tensor([0.2])
PHASE_ANGLES = torch.tensor([0.0])

# Define the parameters for the ODE numerical solution
INITIAL_STATE: torch.tensor = torch.tensor(
    data=np.array([0.1, 0.1]), dtype=torch.float64
).to(device=DEVICE)

TIMESTEP: torch.Tensor = torch.tensor(0.1)
T0: float = 0.0
FINALTIME: float = 20.0

# Boolean constant for whether or not PI controllers included
CONTROLLERS: bool = False

# PINN Hyperparameter constants
LEARNING_RATE: float = 0.01
SCHEDULER_STEP_SIZE: int = 200
SCHEDULER_FACTOR: float = 0.9
EPOCHS: int = 5_000
N_C: int = 5_000    # Number of collocation points

# PINN soft regularisation weights in the loss function
PHYSICS_WEIGHT: float = 1.0
IC_WEIGHT: float = 1.0

ACTIVATION: str = "gelu"

# Define directory constants
ROOT: Path = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive-UniversityofWarwick"
    / "dissertation_code"
)

# Go to the correct directory depending on whether or not PI controllers were used
if CONTROLLERS:
    PATH: Path = ROOT / "data" / "numerical_solutions" / "controllers"
else:
    PATH: Path = ROOT / "data" / "numerical_solutions" / "no_controllers"

#List of numerical solution .npz files
FILE_NAMES: list[str] = listdir(path=PATH)

########################################################################################
# Define the set of collocation points in the temporal domain of the swing equation
########################################################################################

# Obtain samples via LHS of size N_C
LHC = qmc.LatinHypercube(d=1)
collocation_points = LHC.random(n=N_C)
collocation_points = qmc.scale(
    collocation_points, T0, FINALTIME
).flatten()  # Scale from a unit interval [0,1] (default) to [t0,T]

collocation_points: torch.tensor = torch.tensor(
    data=collocation_points[:, None].astype(np.float32), requires_grad=True
).to(device=DEVICE)

########################################################################################
# Loop through each numerical solution, train and evaluate the performance of the train-
# -ed PINN against the times of the numerical solution. Compare the PINN vs RK45 solutions
# and output the PINN models and visualisations to disk.
########################################################################################

for file_index, FILE in enumerate(FILE_NAMES):
    print(
        f"{'-' * 10:^30}File number: {file_index + 1}/{len(FILE_NAMES)}{'-' * 10:^30}"
    )
    # file_name: str = f'inertia_{INERTIA.item()}_damping_{DAMPING.item()}'#_power_{mechanical_power}'
    data = np.load(PATH / FILE)

    phase_angle_numerical = data["phase_angle"]
    angular_frequency_numerical = data["angular_freq"]

    phase_angle_noisy = data["phase_angle_noisy"]
    angular_frequency_noisy = data["angular_freq_noisy"]

    times = data["times"]

    # Define swing equation constants
    INERTIA_DAMPING = findall(pattern="0.[0-9]+", string=FILE)
    INERTIA = torch.tensor(data=float(INERTIA_DAMPING[0]))
    DAMPING = torch.tensor(data=float(INERTIA_DAMPING[1]))

    # Define PINN, optimiser and learning rate scheduler
    pinn = PINN(activation=ACTIVATION).to(device=DEVICE)
    optimiser = torch.optim.Adam(params=pinn.parameters(), lr=LEARNING_RATE)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimiser, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_FACTOR
    )

    # Define array to collect training loss every epoch
    training_loss = []

    # Loop through each epoch (full-batch training)
    for epoch in tqdm(range(EPOCHS)):

        # Obtain PINN predictions and it's time derivatives
        phase_angle_pred = pinn.forward(
            data=collocation_points, initial_çstate=INITIAL_STATE
        )

        angular_frequency_pred = torch.autograd.grad(
            outputs=phase_angle_pred,
            inputs=collocation_points,
            grad_outputs=torch.ones_like(phase_angle_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        angular_acceleration_pred = torch.autograd.grad(
            outputs=angular_frequency_pred,
            inputs=collocation_points,
            grad_outputs=torch.ones_like(angular_frequency_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        # loss = physics_based_loss(
        #     phase_angle=phase_angle_pred,
        #     angular_frequency=angular_frequency_pred,
        #     angular_acceleration=angular_acceleration_pred,
        #     inertia=INERTIA,
        #     damping=DAMPING,
        #     mechanical_power=MECHANICAL_POWER,
        #     voltage_magnitude=VOLTAGE,
        #     voltages=VOLTAGES,
        #     phase_angles=PHASE_ANGLES,
        #     susceptances=SUSCEPTANCES,
        #     include_controllers=CONTROLLERS
        # )

        swing_inputs = SwingEquationInputs(
            phase_angle=phase_angle_pred,
            angular_frequency=angular_frequency_pred,
            angular_acceleration=angular_acceleration_pred,
            inertia=INERTIA,
            damping=DAMPING,
            mechanical_power=MECHANICAL_POWER,
            voltage_magnitude=VOLTAGE,
            voltages=VOLTAGES,
            phase_angles=PHASE_ANGLES,
            susceptances=SUSCEPTANCES,
            controller_proportional=None,
            controller_integral=None,
        )

        loss = total_loss(
            swing_inputs=swing_inputs,
            physics_weight=PHYSICS_WEIGHT,
            IC_weight=IC_WEIGHT,
            model=pinn,
            initial_state=INITIAL_STATE,
            device=DEVICE,
            include_controllers=CONTROLLERS,
        )

        # Backpropogate using reverse/backward-mode AD 
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        lr_scheduler.step()
        training_loss.append(loss.item())

        if epoch % PRINT_TRAINING_LOSS_EVERY_EPOCH == 0:
            print(f"Training loss: {loss}")

    MODEL_NAME: str = f"pinn_inertia_{INERTIA_DAMPING[0]}_damping_{INERTIA_DAMPING[1]}_power.pth"  # _{mechanical_power}.pth'

    # Evaluate the trained PINN
    pinn.eval()
    evaluation_points = torch.tensor(
        data=times, dtype=torch.float32, requires_grad=True
    ).to(device=DEVICE)[:, None]

    phase_angle_eval = pinn(data=evaluation_points, initial_state=INITIAL_STATE)

    angular_frequency_eval = torch.autograd.grad(
        outputs=phase_angle_eval,
        inputs=evaluation_points,
        grad_outputs=torch.ones_like(phase_angle_eval),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Bring back data into CPU memory from GPU memory and type cast into np arrays
    phase_angle_eval = phase_angle_eval.cpu().detach().numpy()
    angular_frequency_eval = angular_frequency_eval.cpu().detach().numpy()
    evaluation_points = evaluation_points.cpu().detach().numpy()

    # Plot the PINN prediction vs RK45 solution, and the training loss per epoch
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].plot(evaluation_points, phase_angle_eval, color="steelblue", label="PINN")
    axes[0].plot(
        times, phase_angle_numerical, color="red", linestyle="--", label="RK45"
    )
    axes[0].set_xlabel("Time (s)", fontsize=14)
    axes[0].set_ylabel("Phase angle $\delta$ (rad)", fontsize=14)
    axes[0].legend(fontsize=13)

    axes[1].plot(
        evaluation_points, angular_frequency_eval, color="steelblue", label="PINN"
    )
    axes[1].plot(
        times, angular_frequency_numerical, color="red", linestyle="--", label="RK45"
    )
    axes[1].set_xlabel("Time (s)", fontsize=14)
    axes[1].set_ylabel("Angular frequency $\dot{\delta}$ (rad/s)", fontsize=14)
    axes[1].legend(fontsize=13)

    axes[2].semilogy(range(EPOCHS), training_loss, color="steelblue")
    axes[2].set_xlabel("Epochs", fontsize=14)
    axes[2].set_ylabel(
        "Physics-based loss $\mathcal{L}_{\mathrm{physics}}$", fontsize=14
    )

    fig.tight_layout(pad=2.0)

    # Save the trained PINN model and the visualisation to risk, depending on whether
    # or not PI controllers are included
    if CONTROLLERS:
        torch.save(obj=pinn, f=ROOT / "models" / "pinn" / "controllers" / MODEL_NAME)

        plt.savefig(
            ROOT
            / "data"
            / "visualisations"
            / "PINN_solutions"
            / "controllers"
            / (FILE.replace(".npz", ".pdf")),
            format="pdf",
            bbox_inches="tight",
        )

    else:
        torch.save(obj=pinn, f=ROOT / "models" / "pinn" / "no_controllers" / MODEL_NAME)

        plt.savefig(
            ROOT
            / "data"
            / "visualisations"
            / "PINN_solutions"
            / "no_controllers"
            / (FILE.replace(".npz", ".pdf")),
            format="pdf",
            bbox_inches="tight",
        )
