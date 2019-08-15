# -*- coding: utf-8 -*-
"""
Solve the SPOT problem.

In this case, we know the spectrum and the baseline
perfectly. The problem is linear in the map, so solving
the Doppler problem is easy!

"""
import paparazzi as pp
import numpy as np
from utils.spot import plot_results

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
dop.y1 = dop.y1_true
baseline = dop.baseline()

# Reset all coefficients
dop.s = dop.s_true
dop.y1 = None

# Solve!
loss, cho_y1, cho_s = dop.solve(s=dop.s, baseline=baseline)

# Plot the results
plot_results(dop, name="spot_y1", loss=loss, cho_y1=cho_y1, cho_s=cho_s)
