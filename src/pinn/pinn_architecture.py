# Import in the required libraries and functions
import torch

from loss_functions import modified_PINN_pred

class PINN(torch.nn.Module):

    def __init__(self):

        super(PINN, self).__init__()

        self.PINN = torch.nn.Sequential(
            torch.nn.Linear(1, 10),

            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),

            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),

            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),

            torch.nn.Tanh(),
            torch.nn.Linear(10, 10),

            torch.nn.Linear(10, 1, bias=False)
        )
        
    def forward(self, data: torch.tensor, initial_state: torch.tensor) -> tuple[torch.tensor, torch.tensor]:

        phase_angle_pred = self.PINN(data)

        return phase_angle_pred