# -*- coding: utf-8 -*-
"""
Solve the SPOT problem.

In this case, we know the rest frame spectrum, but we don't
know the map coefficients or the baseline flux. The problem
can be linearized to solve for the coefficients, and then
refined with the non-linear solver.

"""
import paparazzi as pp
import numpy as np
from utils.spot import plot_results

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
    T = 100.0
    dlogT = -0.25
else:
    # At low SNR, a single run of the bi-linear solver
    # gets us to the optimum!
    ferr = 1e-3
    niter = 0
    lr = None
    T = 1.0
    dlogT = -0.25

# Generate data
dop = pp.Doppler(ydeg=15, u=[0.5, 0.25])
dop.generate_data(ferr=ferr)

# Reset all coefficients
dop.s = dop.s_true
dop.y1 = None

# Solve!
loss, cho_y1, cho_s = dop.solve(s=dop.s, niter=niter, lr=lr, T=T, dlogT=dlogT)

# Plot the results
plot_results(dop, name="spot_y1b", loss=loss, cho_y1=cho_y1, cho_s=cho_s)
