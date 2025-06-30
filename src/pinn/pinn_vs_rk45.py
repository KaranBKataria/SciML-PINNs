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
from tqdm import tqdm
import scienceplots

from pinn_architecture import PINN


# Config matplotlib and define plot constants
plt.style.use('science')
plt.rcParams['text.usetex'] = True

CMAP: str = 'plasma'
LEVELS: int = 30

INITIAL_STATE: torch.tensor = torch.tensor(data=np.array([0.1, 0.1]), dtype=torch.float64)

# Define directory constants
ROOT: str = Path.home() / 'Library' / 'CloudStorage' / 'OneDrive-UniversityofWarwick'/ 'dissertation_code'
PATH_MODEL: str = ROOT / 'models' / 'pinn' / 'no_controllers'
PATH_RK45: str = ROOT / 'data' / 'numerical_solutions'

PINN_MODELS: list[str] = [file.name for file in PATH_MODEL.glob('*.pth')]#listdir(path=PATH_MODEL)
NUMERICAL_FILE_NAMES: list[str] = listdir(path=PATH_RK45)
print(PINN_MODELS)
print(NUMERICAL_FILE_NAMES)

# PINN Hyperparameter constants
ACTIVATION: str = 'gelu'

HYPERPARAMS = np.load(file=ROOT / 'data' / 'hyperparameter_grid.npy')

RMSE_phase_angle: list[float] = []
RMSE_angular_frequency: list[float] = []

INERTIA: np.array = np.unique(HYPERPARAMS[:,1])
DAMPING: np.array = np.unique(HYPERPARAMS[:,0])

for file_index, (numerical_file, model) in enumerate(zip(sorted(NUMERICAL_FILE_NAMES), sorted(PINN_MODELS))):

    # hyperparams = findall(pattern='0.[0-9]+', string=model)
    # inertia.append(hyperparams[0])
    # damping.append(hyperparams[1])

    # Evaluate the trained PINN
    pinn = torch.load(f=PATH_MODEL / model,
        map_location=torch.device('cpu'),
        weights_only=False)
    pinn.eval()

    data = np.load(PATH_RK45 / numerical_file)

    phase_angle_numerical = data['phase_angle']
    angular_frequency_numerical = data['angular_freq']

    phase_angle_noisy = data['phase_angle_noisy']
    angular_frequency_noisy = data['angular_freq_noisy']

    times = data['times']

    evaluation_points = torch.tensor(data=times, dtype=torch.float32, requires_grad=True)[:, None]

    phase_angle_eval = pinn(data=evaluation_points, initial_state=INITIAL_STATE)

    angular_frequency_eval = torch.autograd.grad(
        outputs=phase_angle_eval,
        inputs=evaluation_points,
        grad_outputs=torch.ones_like(phase_angle_eval),
        create_graph=True,
        retain_graph=True
    )[0]

    phase_angle_eval = phase_angle_eval.detach().numpy()
    angular_frequency_eval = angular_frequency_eval.detach().numpy()
    evaluation_points = evaluation_points.detach().numpy()

    testing_RMSE_phase_angle = np.sqrt(np.linalg.norm(x=phase_angle_eval - phase_angle_numerical) ** 2)
    RMSE_phase_angle.append(testing_RMSE_phase_angle)

    testing_RMSE_angular_frequency = np.sqrt(np.linalg.norm(x=angular_frequency_eval - angular_frequency_numerical) ** 2)
    RMSE_angular_frequency.append(testing_RMSE_angular_frequency)

X, Y = np.meshgrid(INERTIA, DAMPING)

fig, ax = plt.subplots(1, 2)
ax[0].contourf(X, Y, np.array(RMSE_phase_angle).reshape(6, 6), levels=LEVELS)
ax[0].set_ylabel('Damping $d$', fontsize=13)
ax[0].set_xlabel('Inertia $m$', fontsize=13)
ax[0].set_title('Phase Angle $\delta$', fontsize=13)

ax[1].contourf(X, Y, np.array(RMSE_angular_frequency).reshape(6, 6), levels=LEVELS)
ax[1].set_xlabel('Inertia $m$', fontsize=13)
ax[1].set_title('Angular Frequency $\dot{\delta}$', fontsize=13)

cbar1 = fig.colorbar(ax[0].contourf(X, Y, np.array(RMSE_phase_angle).reshape(6, 6), levels=LEVELS, cmap=CMAP), ax=ax[0])
cbar2 = fig.colorbar(ax[1].contourf(X, Y, np.array(RMSE_angular_frequency).reshape(6, 6), levels=LEVELS, cmap=CMAP), ax=ax[1])

# cbar1.set_label(label='RMSE', rotation=270, labelpad=18, fontsize=13)
cbar2.set_label(label='RMSE', rotation=270, labelpad=18, fontsize=13)

plt.show()