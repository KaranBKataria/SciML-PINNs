"""
This script defines the PINN architecture to be
imported into other scripts.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

import torch


class PINN(torch.nn.Module):
    """
    Class to define Physics-Informed Neural Network (PINN).
    """
    def __init__(self, activation: str):
        super(PINN, self).__init__()

        self.activation = activation
        act_func = None

        if activation.lower() == "gelu":
            act_func = torch.nn.GELU
        elif activation.lower() == "tanh":
            act_func = torch.nn.Tanh
        else:
            act_func = torch.nn.Sigmoid

        # Define PINN architecture
        self.PINN = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            act_func(),
            torch.nn.Linear(10, 10),
            act_func(),
            torch.nn.Linear(10, 10),
            act_func(),
            torch.nn.Linear(10, 10),
            act_func(),
            torch.nn.Linear(10, 10),
            act_func(),
            torch.nn.Linear(10, 10),
            act_func(),
            torch.nn.Linear(10, 10),
            act_func(),
            torch.nn.Linear(10, 10),
            # act_func,
            torch.nn.Linear(10, 1, bias=False),
        )

        # Run Xavier or He weight initialisation and zero-bias
        self.weight_initialiser()

    # Define forward propogation function
    def forward(self, data: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        # initial_phase_angle = initial_state[0]
        # initial_angular_frequency = initial_state[1]

        phase_angle_pred = self.PINN(data)

        # trial_solution = initial_phase_angle + (initial_angular_frequency * data) + (phase_angle_pred * data * data)

        return phase_angle_pred

    # For each hidden layer, use Xavier or He weight initialisation and zero the biases
    def weight_initialiser(self):
        """
        Weight initialisation at every linear layer of the PINN.
        """
        # Loop through each fully-connected hidden layer
        for module in self.PINN:
            if isinstance(module, torch.nn.Linear):
                if self.activation.lower() == "gelu":
                    torch.nn.init.kaiming_normal_(
                        tensor=module.weight,
                        nonlinearity="relu"
                    )
                else:
                    torch.nn.init.xavier_normal_(tensor=module.weight)

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
