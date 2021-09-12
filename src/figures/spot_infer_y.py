# -*- coding: utf-8 -*-
"""
Setup for the SPOT problem.

"""
from utils.generate import generate_data
from utils.plot import plot_timeseries, plot_maps

# Generate the synthetic dataset
data = generate_data()
map = data["map"]
y = data["y"]
theta = data["theta"]
flux0 = data["flux0"]
flux_err = data["flux0_err"]

# Solve
soln = map.solve(
    flux0, theta=theta, normalized=False, fix_spectrum=True, flux_err=flux_err
)

# Plot the maps
fig = plot_maps(map, y, soln["cho_ycov"])
fig.savefig("spot_infer_y_maps.pdf", bbox_inches="tight", dpi=300)

# Plot the timeseries
fig = plot_timeseries(map, theta, flux0, normalized=False)
fig.savefig("spot_infer_y_timeseries.pdf", bbox_inches="tight", dpi=300)
