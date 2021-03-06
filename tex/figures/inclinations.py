"""
Investigate how our recovered map varies with the
stellar inclination (assumed to be known exactly).

"""
import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt


# Instantiate
res = 300
dop = pp.Doppler(ydeg=15, u=[0.5, 0.25])

# Velocity is computed such that v * sin(40 deg) = 40 km / s
# so that we can compare directly to Vogt et al. (1987)
v = 40.0 / np.sin(40 * np.pi / 180)

# Inclinations
incs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Compute the normalization for the image
dop.generate_data(ferr=0)
dop.y1 = dop.y1_true
img_true = dop.render(projection="rect", res=res).reshape(res, res)
vmax = np.max(img_true)
img = [None for inc in incs]

# Loop
for i, inc in enumerate(incs):

    # Generate data
    dop.inc = inc
    dop.vsini = v * np.sin(inc * np.pi / 180.0)
    np.random.seed(13)
    dop.generate_data(ferr=1e-4)

    # Assume we don't know the baseline
    baseline = None
    T = 100.0

    # Reset all coefficients
    dop.s = dop.s_true
    dop.y1 = None

    # Solve!
    dop.solve(s=dop.s, T=T, niter=50, lr=1e-4, baseline=baseline)

    # Render the inferred map
    img[i] = dop.render(projection="rect", res=res).reshape(res, res)

# Plot
fig, ax = plt.subplots(3, 3, figsize=(15, 8))
ax = ax.flatten()
latlines = np.linspace(-90, 90, 7)[1:-1]
lonlines = np.linspace(-180, 180, 13)
for i in range(len(img)):
    axis = ax[i]
    axis.imshow(
        img[i] / vmax,
        origin="lower",
        extent=(-180, 180, -90, 90),
        cmap="plasma",
        vmin=0,
        vmax=1,
    )
    for lat in latlines:
        axis.axhline(lat, color="k", lw=0.5, alpha=0.5, zorder=100)
    for lon in lonlines:
        axis.axvline(lon, color="k", lw=0.5, alpha=0.5, zorder=100)
    axis.set_xticks(lonlines)
    axis.set_yticks(latlines)
    for tick in axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    axis.annotate(
        r"$%2d^\circ$" % incs[i],
        xy=(0, 0),
        xytext=(7, 7),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=18,
        color="k",
        zorder=101,
    )
fig.savefig("inclinations.pdf", bbox_inches="tight")
