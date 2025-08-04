"""
This script numerically solves the system of first-order ODEs using Runge-Kutta 4,5.
This enables the generation of synthetic data, which can optionally be output to a
specified location in the form of NumPy archive files (.npz). The script solves an
initial value problem (IVP) using SciPy.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

# Import in the required packages and functions
from pathlib import Path

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # Used to make plots more scientific

# Import in the swing equation converted into a system of first-order ODEs
from swing_equation import swing_equation

# Define absolute path to save .npz files in
PATH: Path = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive-UniversityofWarwick"
    / "dissertation_code"
    / "data"
)


def swing_ODEs_solver(
    initial_time: float,
    initial_state: np.ndarray,
    final_time: float,
    timestep: float,
    inertia: float,
    damping: float,
    mechanical_power: float,
    voltage_magnitude: float,
    include_controllers: bool,
    voltages: np.ndarray,
    phase_angles: np.ndarray,
    susceptances: np.ndarray,
    file_name: str,
    save_output_to_file: bool = False,
    noise: float = 0.01,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function outputs the numerical solution, the noisy dynamics and evaluation
    times of the system of first-order ODEs, provided the initial conditions and
    discretisation interval.

    NB: The noisy dynamics are obtained by perturbing the true dynamics with bivariate
    Gaussian noise with a zero-vector mean and a diagonal (by default isotropic) covariance
    matrix of dimensions 2x2 over a real field.

        Inputs:
                initial_time: The initial time of the ODEs
                initial_state: The initial state of the ODEs
                final_time: The final time of the numerical solution
                timestep: The timestep between numerical solutions
                inertia: The inertia coefficient for the given generator
                damping: The damping coefficient for the given generator
                mechanical_power: The mechanical power being supplied to the generator (p.u.)
                voltage_magnitude: The voltage being supplied by the generator (p.u.)
                include_controllers: Boolean arguement to determine whether or not controllers should be included
                voltages: The voltages of the other generators in the power system (p.u.)
                phase_angles: The phase angles of the other generators in the power system (rad)
                susceptances: The susceptance values between other generators and the given generator (p.u.)
                file_name: The name of the output file (.npz)
                save_output_to_file: A boolean expression to determine whether or not to save data
                **kwargs: The controller coefficients (integral and proportional)

        Outputs:
                solutions: The numerical solutions at each time step of the system of 1st order ODEs
                solutions_noise: The noisy (Gaussian noise) dynamics of the numerical solutions
                times: The evaluation times the numerical solutions were evaluated at
    """

    # Define local variables for consistency with solve_ivp documentation
    t0 = initial_time
    y0 = initial_state
    tN = final_time

    # Define a lambda function of the RHS of the ODE system in the format required by solve_ivp i.e.
    # f(t, y) where y is the state of the ODE system
    func = lambda time, state_vec: swing_equation(
        inertia=inertia,
        damping=damping,
        state=state_vec,
        mechanical_power=mechanical_power,
        voltage_magnitude=voltage_magnitude,
        include_controllers=include_controllers,
        voltages=voltages,
        phase_angles=phase_angles,
        susceptances=susceptances,
        **kwargs,
    )

    # Define the mesh the numerical solver will provide solutions to the system of ODEs at
    discretisation = np.arange(start=t0, stop=tN + timestep, step=timestep)

    # Obtain the numerical solution object from the solve_ivp function
    numerical_solution = solve_ivp(
        fun=func, t_span=(t0, tN), y0=y0, method="RK45", t_eval=discretisation
    )

    # Extract the solutions (states) and times (t) the numerical solutions are provided at
    solution = numerical_solution.y
    times = numerical_solution.t

    # Add Gaussian noise to obtain noisy synthetic data to train the PINNs on
    num_states = solution.shape[0]
    num_evals = solution.shape[1]

    gaussian_noise = np.random.randn(num_states, num_evals)

    assert gaussian_noise.shape == solution.shape

    # Append uncorrelated noise% Gaussian noise as in Raissi et al. 2019
    solution_noise = solution + noise * np.std(solution) * gaussian_noise

    # If True, save output .npz file at the specified absolute path
    if save_output_to_file:
        np.savez(
            file=PATH / "numerical_solutions" / file_name,
            phase_angle=solution[0, :],
            angular_freq=solution[1, :],
            phase_angle_noisy=solution_noise[0, :],
            angular_freq_noisy=solution_noise[1, :],
            times=times,
        )

    return (solution, solution_noise, times)


# Test the functionality below
if __name__ == "__main__":
    """
    This section defines an arbitrary test case to test the numerical solver functionality.
    """

    np.random.seed(10)

    plt.style.use("science")

    TRAIN_TEST_SPLIT = 0.3

    # Set up the input arguements (these follow the paper 'Physics-Informed Neural Networks for Power Systems')
    timestep = 0.1
    final_time = 20.0
    inertia = 0.25
    damping = 0.15
    mechanical_power = 0.13
    voltage = 1.0

    voltages = np.array([1.0])
    susceptances = np.array([0.2])
    phase_angles = np.array([0.0])

    # Set up the initial conditions (these follow the paper 'Physics-Informed Neural Networks for Power Systems')
    initial_state = np.array([0.1, 0.1])
    initial_time = 0

    # Obtain numerical solutions (true and noisy) along with times
    solution, solution_noisy, times = swing_ODEs_solver(
        initial_time=initial_time,
        initial_state=initial_state,
        final_time=final_time,
        timestep=timestep,
        inertia=inertia,
        damping=damping,
        mechanical_power=mechanical_power,
        voltage_magnitude=voltage,
        include_controllers=False,
        voltages=voltages,
        phase_angles=phase_angles,
        susceptances=susceptances,
        file_name="test_run",
        save_output_to_file=False,
        controller_proportional=0.05,
        controller_integral=0.1,
    )

    # Define number of total data points from numerical solution, N
    N: int = int((final_time - initial_time)/(timestep) + 1)

    N_D: int = int(np.ceil(len(times)*TRAIN_TEST_SPLIT))

    rand_index = np.random.choice(np.arange(1, N, 1), replace=False, size=N_D)
    rand_index = np.append(rand_index, 0)

    training_data = np.array([solution_noisy[:, idx] for idx in rand_index])
    training_times = np.array([times[idx] for idx in rand_index])

    # Plot the transient dynamics of the phase angle and angular velocity
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(
        times, solution[0, :], linestyle="--", color="black", label="True dynamics"
    )
    ax[0].scatter(
        times, solution_noisy[0, :], color="blue", label="Noisy dynamics", marker='.', alpha=0.5, s=45
    )

    ax[0].scatter(
        training_times, training_data[:, 0], color='red', label="Training data", marker='x', s=70
    )

    ax[0].legend(fontsize=15)
    ax[0].grid()
    ax[0].set_xlabel("Time (s)", fontsize=15)
    ax[0].set_ylabel("Phase Angle $\delta$ (rad)", fontsize=15)

    ax[1].plot(
        times, solution[1, :], linestyle="--", color="black", label="True dynamics"
    )
    ax[1].scatter(
        times, solution_noisy[1, :], color="blue", label="Noisy dynamics", marker='.', alpha=0.5, s=45
    )

    ax[1].scatter(
        training_times, training_data[:, 1], color='red', label="Training data", marker='x', s=70
    )

    ax[1].legend(loc="best", fontsize=15)
    ax[1].grid()
    ax[1].set_xlabel("Time (s)", fontsize=15)
    ax[1].set_ylabel("Angular Frequency $\dot{\delta}$ (rad/s)", fontsize=15)

    plt.suptitle("$N_{\mathcal{D}}=$" + f" {N_D}", fontsize=17)

    PATH_TO_IM_DIR: str = "/Users/karankataria/Library/CloudStorage/OneDrive-UniversityofWarwick"\
                        f"/dissertation_code/data/visualisations/experiments/"

    # plt.savefig(fname=PATH_TO_IM_DIR+"data_generation_example_seed_10.pdf", format="pdf", bbox_inches="tight")

    plt.show()