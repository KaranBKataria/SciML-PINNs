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

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import qmc
from tqdm import tqdm
import scienceplots
from pathlib import Path

from loss_functions import *
from pinn_architecture import PINN

plt.style.use('science')

# Move tensors and models to GPU (MPS not CUDA for M4 chip)
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
DEVICE: str = 'cpu'

# Define directory constants
ROOT: str = Path.home() / 'Library' / 'CloudStorage' / 'OneDrive-UniversityofWarwick'/ 'dissertation_code'
PATH: str = ROOT / 'data' / 'numerical_solutions'

# Define swing equation constants
INERTIA = torch.tensor(0.1, dtype=torch.float64)
DAMPING = torch.tensor(0.09, dtype=torch.float64)
MECHANICAL_POWER = torch.tensor(0.13)
VOLTAGE = torch.tensor(1.0)
VOLTAGES = torch.tensor([1.0])
SUSCEPTANCES = torch.tensor([0.2])
PHASE_ANGLES = torch.tensor([0.0])

INITIAL_STATE: torch.tensor = torch.tensor(data=np.array([0.1, 0.1]), dtype=torch.float32).to(device=DEVICE)
TIMESTEP = torch.tensor(0.1)
T0 = 0
FINALTIME = 20.0

# Boolean constant for whether or not PI controllers included
CONTROLLERS: bool = False

file = f'inertia_{INERTIA.item()}_damping_{DAMPING.item()}.npz' #_power_{mechanical_power}.npz'
data = np.load(PATH / file)

phase_angle_numerical = data['phase_angle']
angular_frequency_numerical = data['angular_freq']

phase_angle_noisy = data['phase_angle_noisy']
angular_frequency_noisy = data['angular_freq_noisy']

times = data['times']

# PINN Hyperparameter constants
LEARNING_RATE: float = 0.01
SCHEDULER_STEP_SIZE: int = 500
SCHEDULER_FACTOR: float = 0.9

EPOCHS: int = 8_000
N_C: int = 10_000

PHYSICS_WEIGHT: float = 1.0
IC_WEIGHT: float = 1.0

# Obtain samples via LHS of size N_C
LHC = qmc.LatinHypercube(d=1)
collocation_points = LHC.random(n=N_C)
collocation_points = qmc.scale(collocation_points, T0, FINALTIME).flatten() # Scale from a unit interval [0,1] (default) to [t0,T]
collocation_points = torch.tensor(data=collocation_points[:, None].astype(np.float32), requires_grad=True).to(device=DEVICE)

pinn = PINN().to(device=DEVICE)
optimiser = torch.optim.Adam(params=pinn.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_FACTOR)

training_loss = []

for epoch in tqdm(range(EPOCHS)):
    
    phase_angle_pred = pinn.forward(data=collocation_points)

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

    # physics_loss = physics_based_loss(
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
        include_controllers=CONTROLLERS
    )

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    lr_scheduler.step()
    training_loss.append(loss.item())

    if epoch % 100 == 0:
        print(f'Training loss: {loss}')

evaluation_points = torch.tensor(data=times, dtype=torch.float32, requires_grad=True)[:, None]
phase_angle_eval = pinn(data=evaluation_points)

angular_frequency_eval = torch.autograd.grad(
    outputs=phase_angle_eval,
    inputs=evaluation_points,
    grad_outputs=torch.ones_like(phase_angle_eval),
    create_graph=True,
    retain_graph=True
)[0]

fig, axes = plt.subplots(1, 3)

axes[0].plot(evaluation_points.detach().numpy(), phase_angle_eval.detach().numpy(), color='steelblue', label='PINN')
axes[0].plot(times, phase_angle_numerical, color='red', linestyle='--', label='RK45')
axes[0].set_xlabel('Time (s)', fontsize=13)
axes[0].set_ylabel('Phase angle $\delta$ (rad)', fontsize=13)
axes[0].legend(fontsize=12)

axes[1].plot(evaluation_points.detach().numpy(), angular_frequency_eval.detach().numpy(), color='steelblue', label='PINN')
axes[1].plot(times, angular_frequency_numerical, color='red', linestyle='--', label='RK45')
axes[1].set_xlabel('Time (s)', fontsize=13)
axes[1].set_ylabel('Angular frequency $\dot{\delta}$ (rad/s)', fontsize=13)
axes[1].legend(fontsize=12)

axes[2].plot(range(EPOCHS), training_loss)
axes[2].set_xlabel('Epochs', fontsize=13)
axes[2].set_ylabel('Physics-based loss $\mathcal{L}_{\mathrm{physics}}$', fontsize=13)

plt.show()

model_name: str = f'pinn_inertia_{INERTIA.item()}_damping_{DAMPING.item()}_power.pth'#_{mechanical_power}.pth'

if CONTROLLERS:
    torch.save(obj=pinn, f=ROOT / 'models' / 'pinn' / 'controllers' / model_name)
else:
    torch.save(obj=pinn, f=ROOT / 'models' / 'pinn' / 'no_controllers' / model_name)