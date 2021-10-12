# -*- coding: utf-8 -*-
"""
Setup for the SPOT problem.

"""
from utils.generate import generate_data
from utils.plot import plot_timeseries, plot_maps, plot_spectra
import starry
import numpy as np
import os


# Generate the synthetic dataset
data = generate_data(flux_err=2e-3)
y_true = data["truths"]["y"]
spectrum_true = data["truths"]["spectrum"].reshape(-1)
theta = data["data"]["theta"]
flux = data["data"]["flux"]
flux_err = data["data"]["flux_err"]

# Instantiate the map
map = starry.DopplerMap(lazy=False, **data["kwargs"])
for n in range(map.udeg):
    map[1 + n] = data["props"]["u"][n]

# Solve for the Ylm coeffs and the spectrum
soln = map.solve(
    flux,
    theta=theta,
    normalized=True,
    flux_err=flux_err,
    spectral_lambda=1e4,
    spectral_cov=2e-2,
    spatial_cov=2e-4,
    quiet=os.getenv("CI", "false") == "true",
)

# Get the inferred map and spectrum
y_inferred = map.y
wav0 = map.wav0
wav = map.wav
spectrum_inferred = map.spectrum.reshape(-1)

# Compute the Ylm expansion of the posterior standard deviation field
P = map.sht_matrix(inverse=True)
Q = map.sht_matrix()
L = np.tril(soln["cho_ycov"])
W = P @ L
y_uncert = Q @ np.sqrt(np.diag(W @ W.T))

# Get the spectrum guess &  uncertainty
M = np.array(map._S0i2eTr.todense())  # converts from wav0_ grid to wav0 grid
spectrum_guess = (soln["spectrum_guess"] @ M).reshape(-1)
L = np.tril(soln["cho_scov"])
spectrum_uncert = (np.sqrt(np.diag(L @ L.T)) @ M).reshape(-1)

# Plot the maps
fig = plot_maps(y_true, y_inferred, y_uncert)
fig.savefig("spot_infer_ybs_low_snr_maps.pdf", bbox_inches="tight", dpi=150)

# Plot the spectra
fig = plot_spectra(
    wav,
    wav0,
    spectrum_true,
    spectrum_guess,
    spectrum_inferred,
    spectrum_uncert,
)
fig.savefig("spot_infer_ybs_low_snr_spectra.pdf", bbox_inches="tight", dpi=300)

# Plot the timeseries
fig = plot_timeseries(
    data, y_inferred, spectrum_inferred, normalized=True, overlap=5
)
fig.savefig(
    "spot_infer_ybs_low_snr_timeseries.pdf", bbox_inches="tight", dpi=300
)
