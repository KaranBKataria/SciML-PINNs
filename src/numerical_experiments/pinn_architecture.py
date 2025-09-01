"""
This script defines the network architecture to be used for all 
numerical exeperiments, enhancement strategies, etc.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

import torch


class PINN(torch.nn.Module):
    """
    Class to define network architecture (PINNs and vanilla NNs)
    """

    def __init__(self, activation: str, hidden_units: int = 10, no_of_inputs: int = 1):
        """
        Call constructor for PINN class.

        Parameters
        ----------
        activation : str
            Activation function to be used.
        hidden_units : int
            Number of hidden units across all hidden layers
        no_of_inputs : int
            (Default 1) The number of inputs into the network
        """
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
            torch.nn.Linear(no_of_inputs, hidden_units),
            act_func(),
            torch.nn.Linear(hidden_units, hidden_units),
            act_func(),
            torch.nn.Linear(hidden_units, hidden_units),
            act_func(),
            torch.nn.Linear(hidden_units, hidden_units),
            act_func(),
            torch.nn.Linear(hidden_units, hidden_units),
            act_func(),
            torch.nn.Linear(hidden_units, hidden_units),
            act_func(),
            torch.nn.Linear(hidden_units, hidden_units),
            act_func(),
            torch.nn.Linear(hidden_units, hidden_units),
            # act_func,
            torch.nn.Linear(hidden_units, 1, bias=False),
        )

        # Run Xavier or He weight initialisation and zero-bias
        self.weight_initialiser()
        
    # Define forward propogation function
    def forward(self, data: torch.Tensor, initial_state: torch.Tensor=None, lagaris: bool = False) -> torch.Tensor:
        """
        Evaluate network given an input via forward propagation.

        Parameters
        ----------
        data : torch.Tensor
            Input into the network
        initial_state : torch.Tensor
            (Default None) Initial conditions
        lagaris : bool
            (Default False) Boolean expression to use ICs as hard constraint (Lagaris et al.)

        Returns
        -------
        trial solution | phase_angle_pred : torch.Tensor
            Network prediction
        """
        
        if lagaris:
            initial_phase_angle = initial_state[0]
            initial_angular_frequency = initial_state[1]

            trial_solution = initial_phase_angle + (initial_angular_frequency * data) + (phase_angle_pred * data * data)
            return trial_solution
        
        else:
            phase_angle_pred = self.PINN(data)

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


if __name__ == '__main__':
    pass