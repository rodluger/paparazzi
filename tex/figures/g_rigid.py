# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils import RigidRotationSolver


# Instantiate the solver
ydeg = 10
solver = RigidRotationSolver(ydeg)
npts = 1000
wsini_c = 2.0e-6
maxD = 0.5 * np.log((1 + wsini_c) / (1 - wsini_c))
D = np.linspace(-maxD, maxD, npts)
g = solver.g(D, wsini_c)

# Set up the plot
fig, ax = plt.subplots(ydeg + 1, 2 * ydeg + 1, figsize=(16, 10), 
                       sharex=True, sharey=True)
fig.subplots_adjust(hspace=0)
for axis in ax.flatten():
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])

# Loop over the orders and degrees
n = 0
for i, l in enumerate(range(ydeg + 1)):
    for j, m in enumerate(range(-l, l + 1)):
        j += ydeg - l
        ax[i, j].plot(g[:, n])
        n += 1

# Labels
for j, m in enumerate(range(-ydeg, ydeg + 1)):
    ax[-1, j].set_xlabel("%d" % m, fontsize=14, fontweight="bold", alpha=0.5)
for i, l in enumerate(range(ydeg + 1)):
    ax[i, ydeg - l].set_ylabel("%d" % l, fontsize=14, fontweight="bold",
                               rotation=45, labelpad=20, alpha=0.5)

# Save
fig.savefig("g_rigid.pdf", bbox_inches="tight")