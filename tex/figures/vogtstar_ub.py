# -*- coding: utf-8 -*-
"""
Solve the VOGTSTAR problem.

In this case, we know the rest frame spectrum, but we don't
know the map coefficients or the baseline flux. The problem
can be linearized to solve for the coefficients, and then
refined with the non-linear solver.

"""
import paparazzi as pp
import numpy as np
from utils.vogtstar_plot import plot_results

np.random.seed(13)

# Let's plot the high SNR case for the paper
high_snr = True

# High or low SNR?
if high_snr:
    # At high SNR, we need to do a bit of refinement
    # with the non-linear solver.
    ferr = 1e-4
    niter = 80
    lr = 1e-4
else:
    # At low SNR, a single run of the bi-linear solver
    # gets us to the optimum!
    ferr = 1e-3
    niter = 0
    lr = None

# Generate data
dop = pp.Doppler(ydeg=15)
dop.generate_data(ferr=ferr)

# Reset all coefficients
dop.vT = dop.vT_true
dop.u = None

# Solve!
loss, cho_u, cho_vT = dop.solve(vT=dop.vT, niter=niter, lr=lr)

# Plot the results
plot_results(dop, name="vogtstar_ub", loss=loss, cho_u=cho_u, cho_vT=cho_vT)
