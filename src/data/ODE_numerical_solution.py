"""
This script numerically solves the system of first-order ODEs
using Runge-Kutta 4,5. This enables the generation of synthetic
data.
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
    inertia: float, damping: float, mechanical_power: float,
    voltage_magnitude: float, include_controllers: bool, voltages: np.array, phase_angles: np.array,
    susceptances: np.array, file_name: str, save_output_to_file: bool = False, **kwargs) -> tuple[np.array, np.array]:
   
    t0 = initial_time
    y0 = initial_state
    tN = final_time

    func = lambda time, state_vec: swing_equation(
        inertia=inertia, damping=damping, state=state_vec, mechanical_power=mechanical_power,
        voltage_magnitude=voltage_magnitude, include_controllers=include_controllers, voltages=voltages,
        phase_angles=phase_angles, susceptances=susceptances, **kwargs
        )

    discretisation = np.arange(start=t0, stop=tN+timestep, step=timestep)

    numerical_solution = solve_ivp(fun=func, t_span=(t0, tN), y0=y0, method='RK45', t_eval=discretisation)

    solution = numerical_solution.y
    times = numerical_solution.t

    if save_output_to_file:
        np.savez(file=PATH / file_name, phase_angle=solution[0,:], angular_freq=solution[1,:], times=times)

    return (solution, times)

if __name__ == '__main__':
    """
    This section defines an arbitrary test case to test the numerical solver
    functionality.
    """
    plt.style.use('science')

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

    solution, times = swing_ODEs_solver(
        initial_time=initial_time, initial_state=initial_state, final_time=final_time, timestep=timestep,
        inertia=inertia, damping=damping, mechanical_power=mechanical_power, voltage_magnitude=voltage, include_controllers=False,
        voltages=voltages, phase_angles=phase_angles, susceptances=susceptances, file_name='test_run', save_output_to_file=False
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(times, solution[0,:], linestyle='-.', color='red')

    ax[1].plot(times, solution[1,:], linestyle='-.', color='red')
    ax[1].axhline(0, color='gray', linestyle='--')

    ax[0].set_xlabel('Time (s)', fontsize=12)
    ax[0].set_ylabel('Phase Angle $\delta$ (rad)', fontsize=12)

    ax[1].set_xlabel('Time (s)', fontsize=12)
    ax[1].set_ylabel('Angular Frequency $\dot{\delta}$ (rad/s)', fontsize=12)

    plt.show()