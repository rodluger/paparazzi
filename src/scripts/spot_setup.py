# -*- coding: utf-8 -*-
"""
Setup for the SPOT problem.

"""
from utils import patch_theano
from utils.generate import generate_data
import numpy as np
import matplotlib.pyplot as plt
import starry
import paths

# Instantiate the Doppler map
data = generate_data()
map = starry.DopplerMap(lazy=False, **data["kwargs"])
map[:, :] = data["truths"]["y"]
map.spectrum = data["truths"]["spectrum"]
for n in range(map.udeg):
    map[1 + n] = data["props"]["u"][n]
theta = data["data"]["theta"]

# Set up the plot
fig = plt.figure(figsize=(11, 9))
fig.subplots_adjust(hspace=0.75)
ax = [
    plt.subplot2grid((33, 8), (0, 0), rowspan=12, colspan=5),
    plt.subplot2grid((33, 8), (0, 5), rowspan=12, colspan=3),
]
ax_ortho = [
    plt.subplot2grid((33, 8), (14, n), rowspan=4, colspan=1) for n in range(8)
]
ax_ortho += [
    plt.subplot2grid((33, 8), (18, n), rowspan=4, colspan=1) for n in range(8)
]
ax_data = plt.subplot2grid((33, 8), (23, 0), rowspan=10, colspan=8)

# Show the mollweide image
map.show(ax=ax[0], projection="moll")

# Show the ortho images
for n in range(16):
    map.show(ax=ax_ortho[n], theta=theta[n])
    ax_ortho[n].annotate(
        r"$%d^\circ$" % theta[n],
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(4, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6,
        color="k",
    )
    ax_ortho[n].set_rasterization_zorder(100)

# Points where we'll evaluate the spectrum. One corresponds to
# a place within the spot; the other is outside the spot.
lats = np.array([10.0, 22.0])
lons = np.array([-15.0, -55.0])
sz = 0.08
intensities = map.intensity(lat=lats, lon=lons)
spectra = map.spectrum.reshape(-1, 1).dot(intensities.reshape(1, -1)).T
spectra /= np.max(spectra)
c = [plt.get_cmap("plasma")(0.8), plt.get_cmap("plasma")(0.2)]
letters = ["A", "B"]
n = 0
for lon, lat in zip(lons, lats):
    t = lat * np.pi / 180
    for k in range(50):
        t -= (2 * t + np.sin(2 * t) - np.pi * np.sin(lat * np.pi / 180)) / (
            2 + 2 * np.cos(2 * t)
        )
    xpt = np.sqrt(2) / 90 * lon * np.cos(t)
    ypt = np.sqrt(2) * np.sin(t)
    ax[0].plot(
        [xpt - sz, xpt - sz], [ypt - sz, ypt + sz], "-", color="w", zorder=101
    )
    ax[0].plot(
        [xpt + sz, xpt + sz], [ypt - sz, ypt + sz], "-", color="w", zorder=101
    )
    ax[0].plot(
        [xpt - sz, xpt + sz], [ypt - sz, ypt - sz], "-", color="w", zorder=101
    )
    ax[0].plot(
        [xpt - sz, xpt + sz], [ypt + sz, ypt + sz], "-", color="w", zorder=101
    )
    ax[0].annotate(
        "%s" % letters[n],
        xy=(xpt + sz, ypt + sz),
        xycoords="data",
        xytext=(2, -2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10,
        color="w",
    )
    ax[1].plot(map.wav0, spectra[n], "-", color=c[n])
    ax[1].annotate(
        "%s" % letters[n],
        xy=(map.wav0[0], intensities[n] / np.max(intensities)),
        xycoords="data",
        xytext=(4, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10,
        color=c[n],
    )
    n += 1

# Tweak
ax[1].set_ylabel(r"intensity", fontsize=10)
ax[1].set_xlabel(r"$\lambda$ [nm]", fontsize=10)
ax[1].set_aspect(0.4)
for tick in ax[1].xaxis.get_major_ticks() + ax[1].yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
ax[1].margins(0, 0.2)

# Compute the spectra
flux = map.flux(theta=theta)

# Plot the data
ax_data.plot(map.wav0, spectra[0])
ax_data_twin = ax_data.twinx()
label = "observed"
for n in range(len(flux)):
    ax_data_twin.plot(
        map.wav,
        flux[n] / flux[n].max(),
        color="C1",
        alpha=0.3,
        label=label,
    )
    label = None
ax_data.axvspan(map.wav0[0], map.wav[0], color="k", alpha=0.3)
ax_data.axvspan(map.wav[-1], map.wav0[-1], color="k", alpha=0.3)

# Tweak
ax_data.set_ylabel(r"rest frame spectrum", fontsize=10)
ax_data_twin.set_ylabel(r"observed spectrum", fontsize=10)
ax_data.set_xlabel(r"$\lambda$ [nm]", fontsize=12)
for axis in [ax_data, ax_data_twin]:
    for tick in axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
        tick.label2.set_fontsize(10)
ax_data.margins(0, None)
ax_data.set_ylim(0.05, 1.05)
ax_data_twin.set_ylim(0.700, 1.0155)

# We're done!
fig.savefig(paths.figures / "spot_setup.pdf", bbox_inches="tight", dpi=300)
