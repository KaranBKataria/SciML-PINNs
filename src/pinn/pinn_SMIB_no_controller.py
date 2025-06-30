"""
This script defines the architecture of, trains and evaluates
the performance of a physics-informed neural network to obtain
a solution for the transient dynamics of a synchronous generator.
That is, it will be trained to find the solution for the swing
equation. The solutions will be benchmarked against the true dynamics
obtained from the RK45 numerical solutions.

The PINN defined in this script is for the SMIB system with no
PI controller.

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


plt.style.use('science')

# Move tensors and models to GPU (MPS not CUDA for M4 chip)
# DEVICE: str = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
DEVICE: str = 'cpu'

# Define swing equation constants
MECHANICAL_POWER = torch.tensor(0.13)
VOLTAGE = torch.tensor(1.0)
VOLTAGES = torch.tensor([1.0])
SUSCEPTANCES = torch.tensor([0.2])
PHASE_ANGLES = torch.tensor([0.0])

INITIAL_STATE: torch.tensor = torch.tensor(data=np.array([0.1, 0.1]), dtype=torch.float64).to(device=DEVICE)
TIMESTEP: torch.tensor = torch.tensor(0.1)
T0: float = 0.0
FINALTIME: float = 20.0

# Boolean constant for whether or not PI controllers included
CONTROLLERS: bool = False

# PINN Hyperparameter constants
LEARNING_RATE: float = 0.01
SCHEDULER_STEP_SIZE: int = 200
SCHEDULER_FACTOR: float = 0.9
EPOCHS: int = 5_000
N_C: int = 5_000

PHYSICS_WEIGHT: float = 1.0
IC_WEIGHT: float = 1.0

ACTIVATION: str = 'gelu'

# Obtain samples via LHS of size N_C
LHC = qmc.LatinHypercube(d=1)
collocation_points = LHC.random(n=N_C)
collocation_points = qmc.scale(collocation_points, T0, FINALTIME).flatten() # Scale from a unit interval [0,1] (default) to [t0,T]

collocation_points: torch.tensor = torch.tensor(data=collocation_points[:, None].astype(np.float32), requires_grad=True).to(device=DEVICE)

# Define directory constants
ROOT: str = Path.home() / 'Library' / 'CloudStorage' / 'OneDrive-UniversityofWarwick'/ 'dissertation_code'
PATH: str = ROOT / 'data' / 'numerical_solutions'

FILE_NAMES: list[str] = listdir(path=PATH)

for file_index, FILE in enumerate(FILE_NAMES):

    print(f"{'-' * 10:^30}File number: {file_index+1}{'-' * 10:^30}")
    # file_name: str = f'inertia_{INERTIA.item()}_damping_{DAMPING.item()}'#_power_{mechanical_power}'
    data = np.load(PATH / FILE)

    phase_angle_numerical = data['phase_angle']
    angular_frequency_numerical = data['angular_freq']

    phase_angle_noisy = data['phase_angle_noisy']
    angular_frequency_noisy = data['angular_freq_noisy']

    times = data['times']

    # Define swing equation constants
    INERTIA_DAMPING = findall(pattern='0.[0-9]+', string=FILE)
    INERTIA = torch.tensor(data=float(INERTIA_DAMPING[0]))    
    DAMPING = torch.tensor(data=float(INERTIA_DAMPING[1]))

    # Define PINN, optimiser and learning rate scheduler
    pinn = PINN(activation=ACTIVATION).to(device=DEVICE)
    optimiser = torch.optim.Adam(params=pinn.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_FACTOR)

    # Define array to collect training loss every X epochs
    training_loss = []
    PRINT_TRAINING_LOSS_EVERY_EPOCH: int = 100

    for epoch in tqdm(range(EPOCHS)):
        
        phase_angle_pred = pinn.forward(data=collocation_points, initial_state=INITIAL_STATE)

        angular_frequency_pred = torch.autograd.grad(
            outputs=phase_angle_pred,
            inputs=collocation_points,
            grad_outputs=torch.ones_like(phase_angle_pred),
            create_graph=True,
            retain_graph=True
        )[0]

        angular_acceleration_pred = torch.autograd.grad(
            outputs=angular_frequency_pred,
            inputs=collocation_points,
            grad_outputs=torch.ones_like(angular_frequency_pred),
            create_graph=True,
            retain_graph=True
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

        loss = total_loss(
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
            physics_weight=PHYSICS_WEIGHT,
            IC_weight=IC_WEIGHT,
            model=pinn,
            initial_state=INITIAL_STATE,
            device=DEVICE,
            include_controllers=CONTROLLERS
        )

        optimiser.zero_grad()

        loss.backward()
        optimiser.step()
        lr_scheduler.step()
        training_loss.append(loss.item())

        if epoch % PRINT_TRAINING_LOSS_EVERY_EPOCH == 0:
            print(f'Training loss: {loss}')


    model_name: str = f'pinn_inertia_{INERTIA_DAMPING[0]}_damping_{INERTIA_DAMPING[1]}_power.pth'#_{mechanical_power}.pth'

    # Evaluate the trained PINN
    pinn.eval()
    evaluation_points = torch.tensor(data=times, dtype=torch.float32, requires_grad=True).to(device=DEVICE)[:, None]

    phase_angle_eval = pinn(data=evaluation_points, initial_state=INITIAL_STATE)

    angular_frequency_eval = torch.autograd.grad(
        outputs=phase_angle_eval,
        inputs=evaluation_points,
        grad_outputs=torch.ones_like(phase_angle_eval),
        create_graph=True,
        retain_graph=True
    )[0]

    phase_angle_eval = phase_angle_eval.cpu().detach().numpy()
    angular_frequency_eval = angular_frequency_eval.cpu().detach().numpy()
    evaluation_points = evaluation_points.cpu().detach().numpy()

    # testing_RMSE_phase_angle = np.sqrt(np.linalg.norm(x=phase_angle_eval - phase_angle_numerical) ** 2)
    # testing_RMSE_angular_frequency = np.sqrt(np.linalg.norm(x=angular_frequency_eval - angular_frequency_numerical) ** 2)

    # print(f'PINN v. RK45 RMSE (phase angle): {testing_RMSE_phase_angle}')
    # print(f'PINN v. RK45 RMSE (angular frequency): {testing_RMSE_angular_frequency}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].plot(evaluation_points, phase_angle_eval, color='steelblue', label='PINN')
    axes[0].plot(times, phase_angle_numerical, color='red', linestyle='--', label='RK45')
    axes[0].set_xlabel('Time (s)', fontsize=14)
    axes[0].set_ylabel('Phase angle $\delta$ (rad)', fontsize=14)
    axes[0].legend(fontsize=13)

    axes[1].plot(evaluation_points, angular_frequency_eval, color='steelblue', label='PINN')
    axes[1].plot(times, angular_frequency_numerical, color='red', linestyle='--', label='RK45')
    axes[1].set_xlabel('Time (s)', fontsize=14)
    axes[1].set_ylabel('Angular frequency $\dot{\delta}$ (rad/s)', fontsize=14)
    axes[1].legend(fontsize=13)

    axes[2].semilogy(range(EPOCHS), training_loss, color='steelblue')
    axes[2].set_xlabel('Epochs', fontsize=14)
    axes[2].set_ylabel('Physics-based loss $\mathcal{L}_{\mathrm{physics}}$', fontsize=14)

    fig.tight_layout(pad=2.0)
  
    if CONTROLLERS:
        torch.save(obj=pinn, f=ROOT / 'models' / 'pinn' / 'controllers' / model_name)
        plt.savefig(ROOT / 'data' / 'visualisations' / 'PINN_solutions' / 'controllers' / (FILE.replace('.npz', '.pdf')), format="pdf", bbox_inches="tight")
    else:
        torch.save(obj=pinn, f=ROOT / 'models' / 'pinn' / 'no_controllers' / model_name)
        plt.savefig(ROOT / 'data' / 'visualisations' / 'PINN_solutions' / 'no_controllers' / (FILE.replace('.npz', '.pdf')), format="pdf", bbox_inches="tight")

