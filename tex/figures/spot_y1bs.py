# -*- coding: utf-8 -*-
"""
Solve the SPOT problem.

In this case, we know nothing: we're going to learn
both the map and the spectrum. We're not giving the
algorithm an initial guess, either: the spectrum is
learned by deconvolving the data, and the initial
guess for the map is computed via the linearized
problem.

"""
import paparazzi as pp
import numpy as np
from utils.spot import plot_results

np.random.seed(13)

# Let's plot the high SNR case for the paper
high_snr = True

# High or low SNR?
if high_snr:
    # We rely heavily on tempering here. Once we get
    # a good initial guess via the bilinear solver,
    # we run the non-linear solver with a slow learning
    # rate.
    ferr = 1e-4
    T = 5000.0
    niter = 3000
    lr = 1.5e-4
    dlogT = -0.25

    # DEBUG
    dlogT = -0.025

else:
    # This case is easier; just a little tempering for
    # good measure, followed by a fast non-linear
    # refinement.
    ferr = 1e-3
    T = 10.0
    niter = 250
    lr = 2e-3
    dlogT = -0.25

# Generate data
dop = pp.Doppler(ydeg=15, u=[0.5, 0.25])
dop.generate_data(ferr=ferr)

# Reset all coefficients
dop.y1 = None
dop.s = None

# Solve!
loss, cho_y1, cho_s = dop.solve(niter=niter, lr=lr, T=T, dlogT=dlogT)

# Plot the results
plot_results(dop, name="spot_y1bs", loss=loss, cho_y1=cho_y1, cho_s=cho_s)
