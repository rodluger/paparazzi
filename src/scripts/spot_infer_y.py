# -*- coding: utf-8 -*-
"""
Setup for the SPOT problem.

"""
from utils import patch_theano
from utils.generate import generate_data
from utils.plot import plot_timeseries, plot_maps
import starry
import numpy as np
import os
import paths


# Generate the synthetic dataset
data = generate_data()
y_true = data["truths"]["y"]
spectrum_true = data["truths"]["spectrum"]
theta = data["data"]["theta"]
flux = data["data"]["flux0"]
flux_err = data["data"]["flux0_err"]

# Instantiate the map
map = starry.DopplerMap(lazy=False, **data["kwargs"])
map.spectrum = data["truths"]["spectrum"]
for n in range(map.udeg):
    map[1 + n] = data["props"]["u"][n]

# Solve for the Ylm coeffs
soln = map.solve(
    flux,
    theta=theta,
    normalized=False,
    fix_spectrum=True,
    flux_err=flux_err,
    quiet=os.getenv("CI", "false") == "true",
)

# Get the inferred map
y_inferred = map.y

# Compute the Ylm expansion of the posterior standard deviation field
P = map.sht_matrix(inverse=True)
Q = map.sht_matrix()
L = np.tril(soln["cho_ycov"])
W = P @ L
y_uncert = Q @ np.sqrt(np.diag(W @ W.T))

# Plot the maps
fig = plot_maps(y_true, y_inferred, y_uncert)
fig.savefig(
    paths.figures / "spot_infer_y_maps.pdf", bbox_inches="tight", dpi=150
)

# Plot the timeseries
fig = plot_timeseries(data, y_inferred, spectrum_true, normalized=False)
fig.savefig(
    paths.figures / "spot_infer_y_timeseries.pdf", bbox_inches="tight", dpi=300
)
