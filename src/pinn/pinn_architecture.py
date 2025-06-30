"""
This script defines the PINN architecture to be
imported into other scripts.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

# Import in the required libraries and functions
import torch


class PINN(torch.nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # Define PINN architecture
        self.PINN = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.GELU(),
            torch.nn.Linear(10, 10),
            torch.nn.GELU(),
            torch.nn.Linear(10, 10),
            torch.nn.GELU(),
            torch.nn.Linear(10, 10),
            torch.nn.GELU(),
            torch.nn.Linear(10, 10),
            torch.nn.GELU(),
            torch.nn.Linear(10, 10),
            torch.nn.GELU(),
            torch.nn.Linear(10, 10),
            torch.nn.GELU(),
            torch.nn.Linear(10, 10),
            # torch.nn.GELU(),
            torch.nn.Linear(10, 1, bias=False),
        )

        # Run Xavier weight initialisation and zero-bias
        self.weight_initialiser()

    # Define forward propogation function
    def forward(self, data: torch.tensor, initial_state: torch.tensor) -> torch.tensor:
        initial_phase_angle = initial_state[0]
        initial_angular_frequency = initial_state[1]

        phase_angle_pred = self.PINN(data)

        # trial_solution = initial_phase_angle + (initial_angular_frequency * data) + (phase_angle_pred * data * data)

        return phase_angle_pred

    # For each hidden layer, use Xavier weight initialisation and zero the biases
    def weight_initialiser(self):
        # Loop through each fully-connected hidden layer
        for module in self.PINN:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(tensor=module.weight)

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
