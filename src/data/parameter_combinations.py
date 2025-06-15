"""
This script computes the Cartesian product (obtaining a grid)
between different parameter values of the swing equation for a SMIB system.
This data is then output as a .npy file to be used to create different
simulations by numerically solving the system of ODEs.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

# Import in the required packages and functions
from itertools import product
from pathlib import Path

import numpy as np

# Define the global constants 
save_to_file: bool = False

# Define absolute path to save .npz files in
PATH: str = Path.home() / 'Library' / 'CloudStorage' / 'OneDrive-UniversityofWarwick'/ 'dissertation_code' / 'data'

# Define grids of hyperparameter values
damping_coefs = np.arange(start=0.05, stop=0.15+0.05, step=0.05)
inertia_coefs = np.arange(start=0.1, stop=0.3+0.1, step=0.1)
mechanical_power_coefs = np.arange(start=0.05, stop=0.2+0.05, step=0.05)

# Obtain the grid of hyperparameters by performing a Cartesian product over the hyperparameter sets
hyperparameter_combinations = np.array(list(product(damping_coefs, inertia_coefs, mechanical_power_coefs)))

if __name__ == '__main__':

    """
    If save_to_file is set to True, save the hyperparameter grid as a NumPy
    array file to the desired location.
    """

    if save_to_file:
        np.save(
            file=PATH / 'hyperparameter_grid',
            arr=hyperparameter_combinations
        )