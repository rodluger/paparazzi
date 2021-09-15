"""
Investigate how our recovered map varies with the
stellar inclination (assumed to be known exactly).

"""
from utils.generate import generate_data
import starry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Settings
ydeg = 15  # TODO: 20
smoothing = 0

# Array of inclinations
incs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
vsini = 40000.0  # m/s

# Plot the true map
fig, ax = plt.subplots(4, 3, figsize=(15, 10))
ax[0, 0].set_visible(False)
ax[0, 2].set_visible(False)
map = starry.Map(ydeg=ydeg)
map.load("starspot", smoothing=smoothing)
map.show(ax=ax[0, 1], projection="moll")
ax[0, 1].annotate(
    r"true",
    xy=(0, 1),
    xytext=(7, 7),
    clip_on=False,
    xycoords="axes fraction",
    textcoords="offset points",
    ha="left",
    va="top",
    fontsize=14,
    color="k",
    zorder=101,
)

# Solve & plot
ax = ax[1:].flatten()
map = None
for i, inc in enumerate(incs):

    # Maintain constant vsini
    veq = vsini / np.sin(inc * np.pi / 180)

    # Generate the data
    data = generate_data(
        inc=inc,
        veq=veq,
        image="starspot",
        flux_err=1e-4,
        ydeg=ydeg,
        smoothing=smoothing,
    )
    theta = data["data"]["theta"]
    flux = data["data"]["flux"]
    flux_err = data["data"]["flux_err"]

    # Instantiate the map
    if map is None:
        map = starry.DopplerMap(lazy=False, **data["kwargs"])
        map.spectrum = data["truths"]["spectrum"]
        for n in range(map.udeg):
            map[1 + n] = data["props"]["u"][n]
    else:
        map.inc = inc
        map.veq = veq

    # Solve
    soln = map.solve(
        flux,
        theta=theta,
        normalized=True,
        fix_spectrum=True,
        flux_err=flux_err,
        spatial_cov=3e-5,
    )

    # Visualize
    map.show(ax=ax[i], projection="moll")
    ax[i].annotate(
        r"$%2d^\circ$" % inc,
        xy=(0, 1),
        xytext=(7, 7),
        clip_on=False,
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=14,
        color="k",
        zorder=101,
    )
fig.savefig("inclinations.pdf", bbox_inches="tight")
