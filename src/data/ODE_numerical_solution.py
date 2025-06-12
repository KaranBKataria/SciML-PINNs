""" 
This script numerically solves the system of first-order ODEs using Runge-Kutta 4,5.
This enables the generation of synthetic data, which can optionally be output to a
specified location in the form of NumPy archive files (.npz). The script solves an
initial value problem (IVP) using SciPy.

Author: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

# Import in the required packages and functions
from pathlib import Path

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import scienceplots    # Used to make plots more scientific

# Import in the swing equation converted into a system of first-order ODEs
from swing_equation import swing_equation

# Define absolute path to save .npz files in
PATH: str = Path.home() / 'Library' / 'CloudStorage' / 'OneDrive-UniversityofWarwick'/ 'dissertation_code' / 'data'

def swing_ODEs_solver(
    initial_time: float, initial_state: np.array, final_time: float, timestep: float,
    sigma: np.array, inertia: float, damping: float, mechanical_power: float, voltage_magnitude: float,
    include_controllers: bool, voltages: np.array, phase_angles: np.array,
    susceptances: np.array, file_name: str, save_output_to_file: bool = False, **kwargs
    ) -> tuple[np.array, np.array, np.array]:

    """
    This function outputs the numerical solution and evaluation times of the system of first-
    order ODEs, provided the initial conditions and discretisation interval.

        Inputs:
                initial_time: The initial time of the ODEs
                initial_state: The initial state of the ODEs
                final_time: The final time of the numerical solution
                timestep: The timestep between numerical solutions
                sigma: A covariance vector to produce a diagonal covariance matrix for Gaussian noise
                inertia:
                damping:

                file_name: The name of the output file (.npz)
                save_output_to_file: A boolean expression to determine whether or not to save data

        Outputs:
                solutions: The numerical solutions at each time step of the system of 1st order ODEs
                times: The evaluation times the numerical solutions were evaluated at
    """
   
    # Define local variables for consistency with solve_ivp documentation
    t0 = initial_time
    y0 = initial_state
    tN = final_time

    # Define a lambda function of the RHS of the ODE system in the format required by solve_ivp i.e.
    # f(t, y) where y is the state of the ODE system
    func = lambda time, state_vec: swing_equation(
        inertia=inertia, damping=damping, state=state_vec, mechanical_power=mechanical_power,
        voltage_magnitude=voltage_magnitude, include_controllers=include_controllers, voltages=voltages,
        phase_angles=phase_angles, susceptances=susceptances, **kwargs
        )

    # Define the mesh the numerical solver will provide solutions to the system of ODEs at 
    discretisation = np.arange(start=t0, stop=tN+timestep, step=timestep)

    # Obtain the numerical solution object from the solve_ivp function
    numerical_solution = solve_ivp(fun=func, t_span=(t0, tN), y0=y0, method='RK45', t_eval=discretisation)

    # Extract the solutions (states) and times (t) the numerical solutions are provided at
    solution = numerical_solution.y
    times = numerical_solution.t

    # Add Gaussian noise to obtain noisy synthetic data to train the PINNs on
    num_states = solution.shape[0] 
    num_evals = solution.shape[1]

    gaussian_noise = np.random.multivariate_normal(mean=np.zeros(shape=num_states), cov=sigma*sigma*np.eye(N=num_states), size=num_evals)
    solution_noise = solution + gaussian_noise.T

    # If True, save output .npz file at the specified absolute path
    if save_output_to_file:
        np.savez(
            file=PATH / file_name,
            phase_angle=solution[0,:],
            angular_freq=solution[1,:],
            phase_angle_noisy=solution_noise[0,:],
            angular_freq_noisy=solution_noise[1,:],
            times=times
        )

    return (solution, solution_noise, times)

# Test the functionality below
if __name__ == '__main__':
    """
    This section defines an arbitrary test case to test the numerical solver functionality.
    """
    plt.style.use('science')

    # Set up the input arguements (these follow the paper 'Physics-Informed Neural Networks for Power Systems')
    timestep = 0.1
    initial_time = 0
    final_time = 20.0

    initial_state = np.array([0.1, 0.1])

    inertia = 0.2
    damping = 0.1
    mechanical_power = 0.1
    voltage = 1.0
    voltages = np.array([1.0])
    susceptances = np.array([0.2])
    phase_angles = np.array([0.0])

    sigma = np.array([0.02, 0.02])

    # Obtain numerical solutions (true and noisy) along with times
    solution, solution_noisy, times = swing_ODEs_solver(
        initial_time=initial_time, initial_state=initial_state, final_time=final_time, timestep=timestep, sigma=sigma,
        inertia=inertia, damping=damping, mechanical_power=mechanical_power, voltage_magnitude=voltage, include_controllers=False,
        voltages=voltages, phase_angles=phase_angles, susceptances=susceptances, file_name='test_run', save_output_to_file=False
    )

    # Plot the transient dynamics of the phase angle and angular velocity
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(times, solution[0,:], linestyle='-.', color='red', label='True dynamics')
    ax[0].plot(times, solution_noisy[0,:], linestyle=':', color='blue', label='Noisy dynamics')
    ax[0].legend(fontsize=12)

    ax[1].plot(times, solution[1,:], linestyle='-.', color='red', label='True dynamics')
    ax[1].plot(times, solution_noisy[1,:], linestyle=':', color='blue', label='Noisy dynamics')
    ax[1].axhline(0, color='gray', linestyle='--')
    ax[1].legend(fontsize=12)

    ax[0].set_xlabel('Time (s)', fontsize=12)
    ax[0].set_ylabel('Phase Angle $\delta$ (rad)', fontsize=12)

    ax[1].set_xlabel('Time (s)', fontsize=12)
    ax[1].set_ylabel('Angular Frequency $\dot{\delta}$ (rad/s)', fontsize=12)

    plt.show()
