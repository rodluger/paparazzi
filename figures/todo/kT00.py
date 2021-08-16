# -*- coding: utf-8 -*-
"""
Plots the first term of the convolution kernel.

"""
import matplotlib.pyplot as plt
import paparazzi as pp

dop = pp.Doppler(ydeg=1)
dop.generate_data(R=1e6, nlam=99, y1=[0, 0, 0])
kT00 = dop.kT()[0]

fig, ax = plt.subplots(1, figsize=(0.5, 0.25))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.plot(kT00)
ax.axis("off")
ax.margins(0.2, 0.2)
fig.savefig("kT00.pdf", dpi=500)
