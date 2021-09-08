# -*- coding: utf-8 -*-
"""
Limb darkening operator illustration.

"""
import starry
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as tt


# Compute the limb-darkening matrix, `L`
ydeg = 1
udeg = 1
u1 = 1
L = (
    np.array(
        [
            [1 - u1, 0, u1 / np.sqrt(3), 0],
            [0, 1 - u1, 0, 0],
            [u1 / np.sqrt(3), 0, 1 - u1, 0],
            [0, 0, 0, 1 - u1],
            [0, 0, 0, 0],
            [0, u1 / np.sqrt(5), 0, 0],
            [0, 0, (2 * u1) / np.sqrt(15), 0],
            [0, 0, 0, u1 / np.sqrt(5)],
            [0, 0, 0, 0],
        ]
    )
    / (1 - u1 / 3)
)

# Go under the hood with `starry` to manually render
# the limb-darkened map. This is super hacky, sorry.
res = 300
map = starry.Map(ydeg + udeg)
render = (
    lambda y: map.ops.render(
        res,
        0,
        tt.as_tensor_variable([0.0, 0.0]).astype(tt.config.floatX),
        map._inc,
        map._obl,
        tt.as_tensor_variable(y).astype(tt.config.floatX),
        map._u,
        map._f,
    )[0]
    .eval()
    .reshape(res, res)
)

fig, ax = plt.subplots((ydeg + 1) ** 2, 2, figsize=(8, 8))
fig.subplots_adjust(wspace=0.75)
for axis in ax.flatten():
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])

# Plot the transformation of each Ylm
xc = np.linspace(-1, 1, 3000)
yc = np.sqrt(1 - xc ** 2)
n = 0
for l in range(ydeg + 1):
    for m in range(-l, l + 1):

        y = np.zeros((ydeg + udeg + 1) ** 2)
        y[n] = 1.0
        yprime = L.dot(y[: (ydeg + 1) ** 2])

        ax[n, 0].imshow(
            render(y), origin="lower", cmap="plasma", extent=(-1, 1, -1, 1)
        )
        ax[n, 1].imshow(
            render(yprime),
            origin="lower",
            cmap="plasma",
            extent=(-1, 1, -1, 1),
        )

        for k in [0, 1]:
            ax[n, k].plot(xc, yc, "k-")
            ax[n, k].plot(xc, -yc, "k-")

            ax[n, k].set_xlim(-1.02, 1.02)
            ax[n, k].set_ylim(-1.02, 1.02)

        ax[n, 0].set_ylabel(
            r"$Y_{{%d},{%d}}$" % (l, m), rotation=0, fontsize=20
        )
        ax[n, 0].yaxis.set_label_coords(-0.35, 0.4)

        n += 1

# Label the results
ax[0, 1].set_ylabel(r"$\frac{\sqrt{3}}{2}Y_{1, 0}$", rotation=0, fontsize=20)
ax[0, 1].yaxis.set_label_coords(1.5, 0.3)
ax[1, 1].set_ylabel(r"$\frac{3}{2\sqrt{5}}Y_{2, -1}$", rotation=0, fontsize=20)
ax[1, 1].yaxis.set_label_coords(1.6, 0.3)
ax[2, 1].set_ylabel(
    r"$\frac{\sqrt{3}}{2}Y_{0, 0} + \frac{\sqrt{3}}{\sqrt{5}}Y_{2, 0}$",
    rotation=0,
    fontsize=20,
)
ax[2, 1].yaxis.set_label_coords(2.0, 0.3)
ax[3, 1].set_ylabel(r"$\frac{3}{2\sqrt{5}}Y_{2, 1}$", rotation=0, fontsize=20)
ax[3, 1].yaxis.set_label_coords(1.6, 0.3)

# Transformation arrow
eps = -0.09
plt.annotate(
    r"$\mathbf{L}(u_1 = 1)$",
    xy=(0.54 + eps, 0.45),
    xycoords="figure fraction",
    ha="center",
    va="center",
    fontsize=18,
)
plt.annotate(
    r"",
    xy=(0.65 + eps, 0.4),
    xycoords="figure fraction",
    xytext=(0.425 + eps, 0.4),
    textcoords="figure fraction",
    arrowprops=dict(facecolor="black"),
)

fig.savefig("ld.pdf", bbox_inches="tight", dpi=300)
