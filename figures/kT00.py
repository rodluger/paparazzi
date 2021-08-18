# -*- coding: utf-8 -*-
"""
Plots the first term of the convolution kernel.

"""
import matplotlib.pyplot as plt
import starry

# Get the first term in the kernel
map = starry.DopplerMap(1)
vsini = map.ops.vsini_max
x = map.ops.get_x(vsini)
rT = map.ops.get_rT(x)
kT = map.ops.get_kT0(rT)
kT00 = kT[0]

fig, ax = plt.subplots(1, figsize=(0.5, 0.25))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.plot(kT00)
ax.axis("off")
ax.margins(0.2, 0.2)
fig.savefig("kT00.pdf", dpi=500)
