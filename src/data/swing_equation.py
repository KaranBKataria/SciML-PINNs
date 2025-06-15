"""
This file defines the swing equations (inclusive and exclusive of controllers)
to be used in the ODE_numerical_solution.py file for numerically solving the ODEs
and obtaining synthetic data.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

# Import in the required functions 
import numpy as np 

def swing_equation(
        inertia: float, damping: float, state: np.array,
        mechanical_power: float, voltage_magnitude: float, include_controllers: bool,
        voltages: np.array, phase_angles: np.array, susceptances: np.array, **kwargs) -> np.array:
    """
    This function defines the swing equation for a given generator, including
    and exluding controllers, and returns a system of first order ODEs to be
    passed into a numerical solver (e.g. RK45).

        Inputs:
                inertia: The inertia coefficient for the given generator
                damping: The damping coefficient for the given generator
                state: The phase angle (rad) and angular frequency (rad/s) for the given generator
                mechanical_power: The mechanical power being supplied to the generator (p.u.)
                voltage_magnitude: The voltage being supplied by the generator (p.u.)
                include_controllers: Boolean arguement to determine whether or not controllers should be included
                voltages: The voltages of the other generators in the power system (p.u.)
                phase_angles: The phase angles of the other generators in the power system (rad)
                susceptances: The susceptance values between other generators and the given generator (p.u.)
                **kwargs: The controller coefficients (integral and proportional)

        Outputs:
                eta_1_dot: The time-derivative of the first dummy variable (equal to the angular frequency)
                eta_2_dot: The time-derivative of the second dummy variable (equal to the angular acceleration)
    """

    # Extract the phase angle and angular frequencies from the state vector
    phase_angle = state[0]
    angular_frequency = state[1] 
    
    # Compute the total electrical power output generator k supplies to the grid
    total_electrical_output = 0
    for v, delta, B in zip(voltages, phase_angles, susceptances):
        total_electrical_output += B * voltage_magnitude * v * np.sin(phase_angle - delta)

    # Define dummy variables for converting second-order ODE to system of first order ODEs
    eta_1 = phase_angle
    eta_2 = angular_frequency
    
    eta_1_dot = eta_2
    
    # If controllers should be included, return the system of 1st order ODEs with controller coefficients
    if include_controllers:
        controller_proportional = kwargs['controller_proportional']
        controller_integral = kwargs['controller_integral']
        
        eta_2_dot = (1 / inertia) * (controller_integral * phase_angle - total_electrical_output - (damping - controller_proportional) * angular_frequency)

        # Return a system of first-order ODEs
        return np.array([eta_1_dot, eta_2_dot])

    # Else return the system of 1st order ODEs without any controller coefficients
    else:
        eta_2_dot = (1 / inertia) * (mechanical_power - total_electrical_output - damping * angular_frequency) 

        # Return a system of first-order ODEs
        return np.array([eta_1_dot, eta_2_dot])