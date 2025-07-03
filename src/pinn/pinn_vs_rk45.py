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

import matplotlib.pyplot as plt
import numpy as np
import torch
import scienceplots


# Config matplotlib and define plot constants
plt.style.use("science")
plt.rcParams["text.usetex"] = True

CMAP: str = "plasma"
LEVELS: int = 30
ACTIVATION: str = "tanh"

INITIAL_STATE: torch.tensor = torch.tensor(
    data=np.array([0.1, 0.1]), dtype=torch.float64
)

# Define directory constants
ROOT: str = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive-UniversityofWarwick"
    / "dissertation_code"
)
PATH_MODEL: str = ROOT / "models" / "pinn" / "no_controllers" / ACTIVATION
PATH_RK45: str = ROOT / "data" / "numerical_solutions"

PINN_MODELS: list[str] = [
    file.name for file in PATH_MODEL.glob("*.pth")
]  # listdir(path=PATH_MODEL)

NUMERICAL_FILE_NAMES: list[str] = listdir(path=PATH_RK45)

# PINN Hyperparameter constants
HYPERPARAMS = np.load(file=ROOT / "data" / "hyperparameter_grid.npy")

RMSE_phase_angle: list[float] = []
RMSE_angular_frequency: list[float] = []

INERTIA: np.array = np.unique(HYPERPARAMS[:, 1])
DAMPING: np.array = np.unique(HYPERPARAMS[:, 0])

for file_index, (numerical_file, model) in enumerate(
    zip(sorted(NUMERICAL_FILE_NAMES), sorted(PINN_MODELS))
):
    # Evaluate the trained PINN
    pinn = torch.load(
        f=PATH_MODEL / model, map_location=torch.device("cpu"), weights_only=False
    )
    pinn.eval()

    data = np.load(PATH_RK45 / numerical_file)

    phase_angle_numerical = data["phase_angle"]
    angular_frequency_numerical = data["angular_freq"]

    # phase_angle_noisy = data["phase_angle_noisy"]
    # angular_frequency_noisy = data["angular_freq_noisy"]

    times = data["times"]

    evaluation_points = torch.tensor(
        data=times, dtype=torch.float32, requires_grad=True
    )[:, None]

    phase_angle_eval = pinn(data=evaluation_points, initial_state=INITIAL_STATE)

    angular_frequency_eval = torch.autograd.grad(
        outputs=phase_angle_eval,
        inputs=evaluation_points,
        grad_outputs=torch.ones_like(phase_angle_eval),
        create_graph=True,
        retain_graph=True,
    )[0]

    phase_angle_eval = phase_angle_eval.detach().numpy().flatten()
    angular_frequency_eval = angular_frequency_eval.detach().numpy().flatten()
    evaluation_points = evaluation_points.detach().numpy().flatten()

    testing_RMSE_phase_angle = np.sqrt(
        np.mean((phase_angle_eval - phase_angle_numerical) ** 2)
    )
    RMSE_phase_angle.append(testing_RMSE_phase_angle)

    testing_RMSE_angular_frequency = np.sqrt(
        np.mean((angular_frequency_eval - angular_frequency_numerical) ** 2)
    )
    RMSE_angular_frequency.append(testing_RMSE_angular_frequency)

    print(f"Model: {model}")
    print(f"Phase angle RMSE: {testing_RMSE_phase_angle}")

RMSE_phase_angle = (
    np.array(RMSE_phase_angle).reshape(DAMPING.shape[0], INERTIA.shape[0]).T
)
RMSE_angular_frequency = (
    np.array(RMSE_angular_frequency).reshape(DAMPING.shape[0], INERTIA.shape[0]).T
)

# Define shared colormap limits
vmin = min(RMSE_phase_angle.min(), RMSE_angular_frequency.min())
vmax = max(RMSE_phase_angle.max(), RMSE_angular_frequency.max())

# Define tick labels
INERTIA_LABELS = [str(round(num, 2)) for num in INERTIA]
DAMPING_LABELS = [str(round(num, 2)) for num in DAMPING]

fig, ax = plt.subplots(1, 2, sharey=True)

heat1 = ax[0].imshow(
    RMSE_phase_angle,
    origin="lower",
    cmap=CMAP,
    extent=[INERTIA.min(), INERTIA.max(), DAMPING.min(), DAMPING.max()],
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
)
ax[0].set_ylabel("Damping $d$", fontsize=13)
ax[0].set_xlabel("Inertia $m$", fontsize=13)
ax[0].set_title("Phase Angle $\delta$", fontsize=13)
ax[0].set_xticks(INERTIA, labels=INERTIA_LABELS)
ax[0].set_yticks(DAMPING, labels=DAMPING_LABELS)

heat2 = ax[1].imshow(
    RMSE_angular_frequency,
    origin="lower",
    cmap=CMAP,
    extent=[INERTIA.min(), INERTIA.max(), DAMPING.min(), DAMPING.max()],
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
)
ax[1].set_xlabel("Inertia $m$", fontsize=13)
ax[1].set_title("Angular Frequency $\dot{\delta}$", fontsize=13)
ax[1].set_xticks(INERTIA, labels=INERTIA_LABELS)

plt.suptitle(f"Activation Function: {ACTIVATION.upper()}", fontsize=13)

cbar = fig.colorbar(heat2, ax=ax.ravel().tolist())
cbar.set_label(label="RMSE", rotation=270, labelpad=18, fontsize=13)

plt.show()
