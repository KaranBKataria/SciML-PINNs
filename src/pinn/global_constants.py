import torch

# Define swing equation constants
MECHANICAL_POWER: torch.Tensor = torch.tensor(0.13)
VOLTAGE: torch.Tensor = torch.tensor(1.0)
VOLTAGES: torch.Tensor = torch.tensor([1.0])
SUSCEPTANCES: torch.Tensor = torch.tensor([0.2])
PHASE_ANGLES: torch.Tensor = torch.tensor([0.0])

# Define the parameters for the ODE numerical solution
TIMESTEP: torch.Tensor = torch.tensor(0.1)
T0: float = 0.0
FINALTIME: float = 20.0
