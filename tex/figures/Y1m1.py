# -*- coding: utf-8 -*-
"""
Plots the ``Y_{1,-1}`` spherical harmonic.

"""
import matplotlib.pyplot as plt
import starry

map = starry.Map(ydeg=1, lazy=False)
map[1, -1] = 1
img = map.render(res=300)[0]
fig, ax = plt.subplots(1, figsize=(0.25, 0.25))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.imshow(img, origin="lower", cmap="plasma", extent=(-1, 1, -1, 1))
ax.axis("off")
eps = 0.01
ax.set_xlim(-1 - eps, 1 + eps)
ax.set_ylim(-1 - eps, 1 + eps)
fig.savefig("Y1m1.pdf", dpi=500)
