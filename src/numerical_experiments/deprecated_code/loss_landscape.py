import matplotlib.pyplot as plt
import numpy as np
import torch
import scienceplots
from tqdm import tqdm
from scipy.stats import qmc
from torchinfo import summary
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from pinn_architecture import PINN
from utils import set_global_seed
from loss_functions import *
from global_constants import *

# Define and fix seed
SEED: int = 0
set_global_seed(SEED)

# Config matplotlib and define plot constants
plt.style.use("science")
plt.rcParams["text.usetex"] = True

CMAP: str = "plasma"  # Color map for the visualisations
LEVELS: int = 30

# Move tensors and models to GPU
DEVICE: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the training loss every specified number of epochs
PRINT_TRAINING_LOSS_EVERY_EPOCH: int = 100

DAMPING: torch.Tensor = torch.tensor(data=[[0.3]])
INERTIA: torch.Tensor = torch.tensor(data=[[0.1]])

# Define the parameters for the ODE numerical solution
INITIAL_STATE: torch.tensor = torch.tensor(
    data=np.array([0.1, 0.1]), dtype=torch.float64
).to(device=DEVICE)

# Boolean constant for whether or not PI controllers included
CONTROLLERS: bool = False

# PINN Hyperparameter constants
LEARNING_RATE: float = 0.01
SCHEDULER_STEP_SIZE: int = 200
PATIENCE: int = 100
SCHEDULER_FACTOR: float = 0.9
HISTORY: int = 50
EPOCHS: int = 5_000
N_C: int = 5_000  # Number of collocation points

# PINN soft regularisation weights in the loss function
PHYSICS_WEIGHT: float = 1.0
IC_WEIGHT: float = 1.0

ACTIVATION: str = "tanh"

# Obtain samples via LHS of size N_C
LHC = qmc.LatinHypercube(d=1)
collocation_points = LHC.random(n=N_C)
collocation_points = qmc.scale(
    collocation_points, T0, FINALTIME
).flatten()  # Scale from a unit interval [0,1] (default) to [t0,T]

collocation_points: torch.tensor = torch.tensor(
    data=collocation_points[:, None].astype(np.float32), requires_grad=True
).to(device=DEVICE)

# Define PINN, optimiser and learning rate scheduler
pinn = PINN(activation=ACTIVATION).to(device=DEVICE)

# optimiser = torch.optim.LBFGS(
#     params=pinn.parameters(),
#     lr=LEARNING_RATE,
#     history_size=HISTORY,
#     line_search_fn="strong_wolfe",
# )

optimiser = torch.optim.Adam(params=pinn.parameters(), lr=LEARNING_RATE)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimiser, patience=PATIENCE, factor=SCHEDULER_FACTOR
)

# Define array to collect training loss every epoch
training_loss = []

for epoch in tqdm(range(EPOCHS)):
    # Obtain PINN predictions and it's time derivatives
    phase_angle_pred = pinn.forward(
        data=collocation_points, initial_state=INITIAL_STATE
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
    lr_scheduler.step(metrics=loss)
    training_loss.append(loss.item())

    if epoch % PRINT_TRAINING_LOSS_EVERY_EPOCH == 0:
        print(f"Training loss: {loss}")

# Define mesh
alpha_1: torch.Tensor = torch.arange(start=-21, end=22, step=1)
alpha_2: torch.Tensor = torch.arange(start=-21, end=22, step=1)
ALPHA_1, ALPHA_2 = np.meshgrid(alpha_1.numpy(), alpha_2.numpy())

# Sample two dimension compatible random direction vectors
TOTAL_NUM_PARAMS: int = summary(model=pinn).total_params

# Sample Gaussian random direction vectors as in Li et al. (2018)
direction_vec_1: torch.Tensor = torch.normal(
    mean=torch.zeros(TOTAL_NUM_PARAMS), std=torch.ones(TOTAL_NUM_PARAMS)
)

# Make into a unit vector
direction_vec_1 = direction_vec_1 / torch.norm(input=direction_vec_1)

direction_vec_2: torch.Tensor = torch.normal(
    mean=torch.zeros(TOTAL_NUM_PARAMS), std=torch.ones(TOTAL_NUM_PARAMS)
)

# Use the Gram-Schmidt process to convert linearly independant vectors
# into orthonormal vectors
direction_vec_2 = (
    direction_vec_2 - torch.dot(direction_vec_2, direction_vec_1) * direction_vec_1
)

direction_vec_2 = direction_vec_2 / torch.norm(input=direction_vec_2)

# Test orthogonality
assert torch.dot(direction_vec_1, direction_vec_2) < 1e-8

MINIMISER = parameters_to_vector(pinn.parameters())

loss_landscape = []

pinn.eval()

for i in alpha_1:
    for j in alpha_2:
        perturbation = MINIMISER + i * direction_vec_1 + j * direction_vec_2

        vector_to_parameters(vec=perturbation, parameters=pinn.parameters())

        phase_angle_pred = pinn.forward(
            data=collocation_points, initial_state=INITIAL_STATE
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

        loss_perturb = total_loss(
            swing_inputs=swing_inputs,
            physics_weight=PHYSICS_WEIGHT,
            IC_weight=IC_WEIGHT,
            model=pinn,
            initial_state=INITIAL_STATE,
            device=DEVICE,
            include_controllers=CONTROLLERS,
        )

        loss_landscape.append(loss_perturb.detach().numpy())

        # Reset PINN learnt parameters back to the minimiser
        vector_to_parameters(vec=MINIMISER, parameters=pinn.parameters())

loss_landscape = np.array(loss_landscape).reshape(alpha_1.shape[0], alpha_2.shape[0])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surface = ax.plot_surface(
    ALPHA_1, ALPHA_2, np.log(loss_landscape), cmap=CMAP, linewidth=0, antialiased=True
)

# ax.set_xlabel('$\alpha_1$')
# ax.set_ylabel('$\alpha_2$')
# ax.set_zlabel('$\RMSE (log scale)$')

plt.show()
