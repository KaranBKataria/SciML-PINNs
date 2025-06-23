"""
This script defines the architecture of, trains and evaluates
the performance of a physics-informed neural network to obtain
a solution for the transient dynamics of a synchronous generator.
That is, it will be trained to find the solution for the swing
equation. The solutions will be benchmarked against the true dynamics
obtained from the RK45 numerical solutions.

The PINN defined in this script is for the SMIB system with no
PI controller.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayana
"""

