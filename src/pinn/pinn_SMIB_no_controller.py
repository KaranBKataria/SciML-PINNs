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

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc 

if __name__ == '__main__':

    rand_samples = np.random.uniform(0, 20, size=20)

    LHC = qmc.LatinHypercube(d=1)

    samples = LHC.random(n=20)
    samples = qmc.scale(samples, 0, 20)

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(samples, np.zeros(20))
    ax[1].scatter(rand_samples, np.zeros(20), color='red')

    plt.show()
