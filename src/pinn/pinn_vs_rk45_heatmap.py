"""
This script evaluates the test root mean square error (RMSE) of a PINN model and the
corresponding RK45 ground truth trajectory for each combination of the ODE parameters.
This is visualised as a heatmap.

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

########################################################################################
# Define constants
########################################################################################

# Define the activation function to be used in the PINN
ACTIVATION: str = "tanh"

INITIAL_STATE: torch.Tensor = torch.tensor(
    data=np.array([0.1, 0.1]), dtype=torch.float64
)

# Boolean constant for whether or not PI controllers included
CONTROLLERS: bool = False

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
    PATH_MODEL: Path = ROOT / "models" / "pinn" / "controllers" / ACTIVATION
    PATH_RK45: Path = ROOT / "data" / "numerical_solutions" / "controllers"
else:
    PATH_MODEL: Path = ROOT / "models" / "pinn" / "no_controllers" / ACTIVATION
    PATH_RK45: Path = ROOT / "data" / "numerical_solutions" / "no_controllers"

# Extract the PINN model names and the list of numerical .npz files
PINN_MODELS: list[str] = [file.name for file in PATH_MODEL.glob("*.pth")]  # listdir(path=PATH_MODEL)
NUMERICAL_FILE_NAMES: list[str] = listdir(path=PATH_RK45)

# Extract PINN Hyperparameter constants
HYPERPARAMS = np.load(file=ROOT / "data" / "hyperparameter_grid.npy")
INERTIA: np.array = np.unique(HYPERPARAMS[:, 1])
DAMPING: np.array = np.unique(HYPERPARAMS[:, 0])

# Define tick labels
INERTIA_LABELS: list[str] = [str(round(num, 2)) for num in INERTIA]
DAMPING_LABELS: list[str] = [str(round(num, 2)) for num in DAMPING]

########################################################################################
# Loop through each numerical solution and associated PINN model and computing the
# test RMSE. Repeat for all parameter combinations and produce a heatmap of the test
# RMSE for both the phase angle and angular frequency.
########################################################################################

# Define empty lists to collect the RMSE of the PINN vs RK45 solutions for each
# parameter combination
RMSE_phase_angle: list[float] = []
RMSE_angular_frequency: list[float] = []

# Loop through each numerical solution and associated PINN model
for file_index, (numerical_file, model) in enumerate(
    zip(sorted(NUMERICAL_FILE_NAMES), sorted(PINN_MODELS))
    ):

    # Evaluate the trained PINN
    pinn = torch.load(
        f=PATH_MODEL / model, map_location=torch.device("cpu"), weights_only=False
    )
    pinn.eval()

    # Extract the numerical solution arrays
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

    # Bring back data into CPU memory from GPU memory and type cast into np arrays
    phase_angle_eval = phase_angle_eval.detach().numpy().flatten()
    angular_frequency_eval = angular_frequency_eval.detach().numpy().flatten()
    evaluation_points = evaluation_points.detach().numpy().flatten()

    # Compute test RMSE for both phase angle and angular frequency respectively
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

########################################################################################
# Plot the heatmaps showcasing the test RMSE of the PINN vs RK45 solution for each
# ODE parameter combination.
########################################################################################

# Type cast lists into numpy arrays
RMSE_phase_angle = (
    np.array(RMSE_phase_angle).reshape(DAMPING.shape[0], INERTIA.shape[0]).T
)
RMSE_angular_frequency = (
    np.array(RMSE_angular_frequency).reshape(DAMPING.shape[0], INERTIA.shape[0]).T
)

# Define shared colormap limits
vmin = min(RMSE_phase_angle.min(), RMSE_angular_frequency.min())
vmax = max(RMSE_phase_angle.max(), RMSE_angular_frequency.max())

# Plot a heatmap of the test RMSE for each parameter combination of the PINN vs RK45
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
