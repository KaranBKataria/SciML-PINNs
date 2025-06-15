"""
This script contains the main simulation code; it iterates over
all hyperparameter/swing equation coefficient combinations defined
in the 'hyperparameter_grid.npy' file (produced by the
'hyperparameter_combinations.py' script) and outputs the numerical ODE 
solutions as .npz files as well as the corresponding transient dynamic
graphs of the phase angle and angular frequencies in the .png format.

The hyperparameters in the grid include: damping coefficient, inertia coefficient
and the mechanical power input.

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
timestep = 0.1
final_time = 20.0
voltage = 1.0

voltages = np.array([1.0])
susceptances = np.array([0.2])
phase_angles = np.array([0.0])

# Set up the initial conditions (these follow the paper 'Physics-Informed Neural Networks for Power Systems')
initial_time = 0
initial_state = np.array([0.1, 0.1])

# Set the covariance matrix to be isotropic 
sigma = np.array([0.02, 0.02])

# Loop through each of the hyperparameter / ODE coefficient combinations and save the numerical solution data
for iter, (damping, inertia, mechanical_power) in tqdm(enumerate(hyperparameter_grid)):

    print(f'Outputting combination {iter+1}...')

    # Define the output file name (.npz)
    file_name: str =\
        'inertia_' + str(round(inertia, 2)) +\
        '_damping_' + str(round(damping, 2)) +\
        '_power_' + str(round(mechanical_power, 2))

    # Obtain numerical solutions (true and noisy) along with times
    solution, solution_noisy, times = swing_ODEs_solver(
        initial_time=initial_time,
        initial_state=initial_state,
        final_time=final_time,
        timestep=timestep,
        sigma=sigma,
        inertia=inertia,
        damping=damping,
        mechanical_power=mechanical_power,
        voltage_magnitude=voltage,
        include_controllers=False,
        voltages=voltages,
        phase_angles=phase_angles,
        susceptances=susceptances,
        file_name=file_name,
        save_output_to_file=True,
#       controller_proportional=0.05,
#       controller_integral=0.1
    )

    # Plot the transient dynamics of the phase angle and angular velocity
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].plot(times, solution[0,:], linestyle='-.', color='red', label='True dynamics')
    ax[0].plot(times, solution_noisy[0,:], linestyle=':', color='blue', label='Noisy dynamics')
    ax[0].legend(fontsize=15)
    ax[0].grid()
    ax[0].set_xlabel('Time (s)', fontsize=15)
    ax[0].set_ylabel('Phase Angle $\delta$ (rad)', fontsize=15)

    ax[1].plot(times, solution[1,:], linestyle='-.', color='red', label='True dynamics')
    ax[1].plot(times, solution_noisy[1,:], linestyle=':', color='blue', label='Noisy dynamics')
    ax[1].legend(fontsize=15)
    ax[1].grid()
    ax[1].set_xlabel('Time (s)', fontsize=15)
    ax[1].set_ylabel('Angular Frequency $\dot{\delta}$ (rad/s)', fontsize=15) 

    plt.suptitle(
        f'Inertia: {round(inertia, 2)}; Damping: {round(damping, 2)}; Mechanical Power: {round(mechanical_power, 2)}',
        fontsize=15
    )
    
    # Update the file name to reflect .png format
    file_name = file_name + '.png'

    # Save transient dynamic visualisations to the desired location
    plt.savefig(PATH / 'visualisations' / 'numerical_solutions' / file_name, dpi=300, bbox_inches='tight')