# -*- coding: utf-8 -*-
"""
Plots the basis of `g` kernels.

"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import starry


# Get the `kT` functions at high res
ydeg = 10
map = starry.DopplerMap(ydeg)
vsini = map.ops.vsini_max
x = map.ops.get_x(vsini)
rT = map.ops.get_rT(x)
kT = map.ops.get_kT0(rT)

# Set up the plot
fig, ax = plt.subplots(
    ydeg + 1, 2 * ydeg + 1, figsize=(16, 10), sharex=True, sharey=True
)
fig.subplots_adjust(hspace=0)
for axis in ax.flatten():
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])

# Loop over the orders and degrees
n = 0
for i, l in enumerate(range(ydeg + 1)):
    for j, m in enumerate(range(-l, l + 1)):
        j += ydeg - l
        ax[i, j].plot(kT[n])
        n += 1

# Labels
for j, m in enumerate(range(-ydeg, ydeg + 1)):
    ax[-1, j].set_xlabel("%d" % m, fontsize=14, alpha=0.5)
for i, l in enumerate(range(ydeg + 1)):
    ax[i, ydeg - l].set_ylabel(
        "%d" % l, fontsize=14, rotation=45, labelpad=20, alpha=0.5
    )

plt.annotate(
    r"$\ell$",
    xy=(0.18, 0.55),
    xycoords="figure fraction",
    rotation=45,
    fontsize=22,
    alpha=0.5,
)
plt.annotate(
    r"$m$",
    xy=(0.415, 0.02),
    ha="center",
    xycoords="figure fraction",
    fontsize=20,
    alpha=0.5,
    clip_on=False,
)

# Save
fig.savefig("kT.pdf", bbox_inches="tight")
