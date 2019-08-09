# -*- coding: utf-8 -*-
"""
Solve the VOGTSTAR problem.

In this case, we know the spectrum and the baseline
perfectly. The problem is linear in the map, so solving
the Doppler problem is easy!

"""
import paparazzi as pp
import numpy as np
from utils.vogtstar_plot import plot_results

np.random.seed(13)

# Let's plot the high SNR case for the paper
high_snr = True

# High or low SNR?
if high_snr:
    ferr = 1e-4
else:
    ferr = 1e-3

# Generate data at high SNR
dop = pp.Doppler(ydeg=15)
dop.generate_data(ferr=ferr)

# Compute the true baseline (assumed to be known exactly)
dop.u = dop.u_true
baseline = dop.baseline()

# Reset all coefficients
dop.vT = dop.vT_true
dop.u = None

# Solve!
loss, cho_u, cho_vT = dop.solve(vT=dop.vT, baseline=baseline)

# Plot the results
plot_results(dop, name="vogtstar_u", loss=loss, cho_u=cho_u, cho_vT=cho_vT)
