"""
This script produces the plots for the comparison between Latin Hypercube
Sampling (LHS) and vanilla uniform random sampling in obtain an uniform
distribution of collocation points densly sampling the temporal domain
of the swing equation.

Note: LHS in 1D is equivalent to stratified uniform random sampling.

Dev: Karan Kataria
Work: MSc thesis
Supervisor: Dr. Subhash Lakshminarayan
"""

# Import in the required libraries and functions
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc # Import in the Quasi-Monte Carlo class 
import scienceplots

# Set random seet for reproducibility
np.random.seed(2)

# Set matplotlib style for professional, scientific plots
plt.style.use('science')

# Define axes object
fig, ax = plt.subplots(4, 2, figsize=(6, 10), constrained_layout=True, sharex=True, sharey=True)

# Loop through different sample sizes for the collocation points and compare
# LHS and vanilla uniform random sampling of the temporal domain [0, 20]
for n_sample, index in zip(range(10, 30, 5), range(0, 4)):

    # Obtain vanilla uniform random samples of size n_sample
    rand_samples = np.random.uniform(0, 20, size=n_sample)

    # Obtain samples via LHS of size n_sample
    LHC = qmc.LatinHypercube(d=1)
    samples = LHC.random(n=n_sample)
    samples = qmc.scale(samples, 0, 20) # Scale from a unit interval [0,1] (default) to [0,20]
 
    # Plot the LHS samples
    ax[index][0].scatter(samples, np.zeros(n_sample), label='$N_{\mathcal{C}}$' + f'={n} (LHS)') 
    ax[index][0].legend(fontsize=10)

    # Plot the uniform random samples
    ax[index][1].scatter(rand_samples, np.zeros(n_sample), color='red', label='$N_{\mathcal{C}}$' + f'={n} (random)') 
    ax[index][1].legend(fontsize=10)

    # Only add x-axis label on the last row of subplots
    if index == 3:
        ax[index][0].set_xlabel('Temporal domain, $\mathcal{T}$', fontsize=10)
        ax[index][1].set_xlabel('Temporal domain, $\mathcal{T}$', fontsize=10)

# Ensure tight orientation of subplots
plt.subplots_adjust()
plt.show()
