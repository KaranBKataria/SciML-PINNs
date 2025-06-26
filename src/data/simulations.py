"""
This script contains the main simulation code; it iterates over
all hyperparameter/swing equation coefficient combinations defined
in the 'hyperparameter_grid.npy' file (produced by the
'hyperparameter_combinations.py' script) and outputs the numerical ODE 
solutions as .npz files as well as the corresponding transient dynamic
graphs of the phase angle and angular frequencies in the .png format.

The hyperparameters in the grid include: damping coefficient and inertia
coefficient.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

# Import in all required libraries and functions
import numpy as np
from tqdm import tqdm

from ODE_numerical_solver import *

# Load in the grid of hyperparameters
hyperparameter_grid = np.load(file=PATH / 'hyperparameter_grid.npy')

# Set style of plots to scientific
plt.style.use('science')

# Set up the input arguements (these follow the paper 'Physics-Informed Neural Networks for Power Systems')
TIMESTEP = 0.1
FINAL_TIME = 20.0
VOLTAGE = 1.0
MECHANICAL_POWER = 0.13 # Choosen inbetween [0.08, 0.18] explored in the paper

VOLTAGES = np.array([1.0])
SUSCEPTANCES = np.array([0.2])
PHASE_ANGLES = np.array([0.0])

# Set up the initial conditions (these follow the paper 'Physics-Informed Neural Networks for Power Systems')
INITIAL_TIME = 0
INITIAL_STATE = np.array([0.1, 0.1])

# Boolean constant for whether or not PI controllers included
CONTROLLERS: bool = False

# Set the covariance matrix to be isotropic with choosen standard deviation
SIGMA = np.array([0.02, 0.02])

# Loop through each of the hyperparameter / ODE coefficient combinations and save the numerical solution data
for iter, (damping, inertia) in tqdm(enumerate(hyperparameter_grid)):

    print(f'\t Outputting combination {iter+1}...')

    # Define the output file name (.npz)
    file_name: str = 'inertia_' + str(round(inertia, 2)) +\
        '_damping_' + str(round(damping, 2))

    # Obtain numerical solutions (true and noisy) along with times
    solution, solution_noisy, times = swing_ODEs_solver(
        initial_time=INITIAL_TIME,
        initial_state=INITIAL_STATE,
        final_time=FINAL_TIME,
        timestep=TIMESTEP,
        sigma=SIGMA,
        inertia=inertia,
        damping=damping,
        mechanical_power=MECHANICAL_POWER,
        voltage_magnitude=VOLTAGE,
        include_controllers=CONTROLLERS,
        voltages=VOLTAGES,
        phase_angles=PHASE_ANGLES,
        susceptances=SUSCEPTANCES,
        file_name=file_name,
        save_output_to_file=True,
#       controller_proportional=0.05,
#       controller_integral=0.1
    )

    # Plot the transient dynamics of the phase angle and angular velocity
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].plot(times, solution[0,:], linestyle='-.', color='red', label='True dynamics')
    ax[0].scatter(times, solution_noisy[0,:], marker='.', color='steelblue', label='Noisy data')
    ax[0].legend(fontsize=12)
    ax[0].grid()
    ax[0].set_xlabel('Time (s)', fontsize=13)
    ax[0].set_ylabel('Phase Angle $\delta$ (rad)', fontsize=13)

    ax[1].plot(times, solution[1,:], linestyle='-.', color='red', label='True dynamics')
    ax[1].scatter(times, solution_noisy[1,:], marker='.', color='steelblue', label='Noisy data')
    ax[1].legend(fontsize=12)
    ax[1].grid()
    ax[1].set_xlabel('Time (s)', fontsize=13)
    ax[1].set_ylabel('Angular Frequency $\dot{\delta}$ (rad/s)', fontsize=13) 

    plt.suptitle(
        f'Inertia: {round(inertia, 2)}; Damping: {round(damping, 2)}', #; Mechanical Power: {round(mechanical_power, 2)}
        fontsize=15
    )
    
    # Update the file name to reflect .png format
    file_name = file_name + '.png'

    # Save transient dynamic visualisations to the desired location
    if CONTROLLERS:
        plt.savefig(PATH / 'visualisations' / 'numerical_solutions' / 'controllers' / file_name, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(PATH / 'visualisations' / 'numerical_solutions' / 'no_controllers' / file_name, dpi=300, bbox_inches='tight')