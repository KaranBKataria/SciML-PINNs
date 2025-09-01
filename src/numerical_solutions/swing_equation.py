"""
This file defines the swing equations (inclusive and exclusive of controllers)
to be used in the ODE_numerical_solution.py file for numerically solving the ODEs
and obtaining synthetic data.

NB: Use of controllers is now deprecated; this was NOT used in the report.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

# Import in the required functions
import numpy as np


def swing_equation(
    inertia: float,
    damping: float,
    state: np.ndarray,
    mechanical_power: float,
    voltage_magnitude: float,
    include_controllers: bool,
    voltages: np.ndarray,
    phase_angles: np.ndarray,
    susceptances: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Defines the swing equation for a given generator, including
    or exluding controllers, and returns a system of first order ODEs to be
    passed into a numerical solver.

    Parameters
    ----------
    inertia : float
        The inertia coefficient for the given generator
    damping : float
        The damping coefficient for the given generator
    state : np.ndarray 
        The phase angle (rad) and angular frequency (rad/s) for the given generator
    mechanical_power : float
        The mechanical power being supplied to the generator (p.u.)
    voltage_magnitude : float
        The voltage being supplied by the generator (p.u.)
    include_controllers : bool
        Boolean arguement to determine whether or not controllers should be included
    voltages : np.ndarray
        The voltages of the other generators in the power system (p.u.)
    phase_angles : np.ndarray
        The phase angles of the other generators in the power system (rad)
    susceptances : np.ndarray
        The susceptance values between other generators and the given generator (p.u.)
    **kwargs : 
        The controller coefficients (integral and proportional)

    Returns 
    -------
     : np.ndarray
        The time-derivatives of the first and second dummy variables
    """

    # Extract the phase angle and angular frequencies from the state vector
    phase_angle = state[0]
    angular_frequency = state[1]

    # Compute the total electrical power output generator k supplies to the grid
    total_electrical_output = 0
    for v, delta, B in zip(voltages, phase_angles, susceptances):
        total_electrical_output += (
            B * voltage_magnitude * v * np.sin(phase_angle - delta)
        )

    # Define dummy variables for converting second-order ODE to system of first order ODEs
    eta_1 = phase_angle
    eta_2 = angular_frequency

    eta_1_dot = eta_2

    # If controllers should be included, return the system of 1st order ODEs with controller coefficients
    if include_controllers:
        controller_proportional = kwargs["controller_proportional"]
        controller_integral = kwargs["controller_integral"]

        eta_2_dot = (1 / inertia) * (
            controller_integral * phase_angle
            - total_electrical_output
            - (damping - controller_proportional) * angular_frequency
        )

        # Return a system of first-order ODEs
        return np.array([eta_1_dot, eta_2_dot])

    # Else return the system of 1st order ODEs without any controller coefficients
    else:
        eta_2_dot = (1 / inertia) * (
            mechanical_power - total_electrical_output - damping * angular_frequency
        )

        # Return a system of first-order ODEs
        return np.array([eta_1_dot, eta_2_dot])
