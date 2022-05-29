"""
Investigate how our recovered map varies with the
signal to noise ratio of the data.

"""
from utils import patch_theano
from utils.generate import generate_data
import starry
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import paths

# Settings
ydeg = 15
smoothing = 0.075

# Compute model at infinite SNR
data = generate_data(flux_err=0, ydeg=ydeg, smoothing=smoothing)
theta = data["data"]["theta"]
flux = data["data"]["flux"]

# Typical line depth
signal = 0.2

# Compute the pointwise uncertainty for a given SNR
snrs = np.array(
    [10.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
)
spatial_cov = np.minimum(1e-4, 1e-4 * (1000 / snrs) ** 2)
flux_errs = signal / snrs

# Plot the true map
fig, ax = plt.subplots(4, 3, figsize=(15, 10))
ax[0, 0].set_visible(False)
ax[0, 2].set_visible(False)
map = starry.Map(ydeg=ydeg)
map.load("spot", smoothing=smoothing)
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
ax[0, 1].set_rasterization_zorder(0)

# Instantiate the map we'll use for inference
map = starry.DopplerMap(lazy=False, **data["kwargs"])
map.spectrum = data["truths"]["spectrum"]
for n in range(map.udeg):
    map[1 + n] = data["props"]["u"][n]

# Solve & plot
ax = ax[1:].flatten()
for i, flux_err in enumerate(flux_errs):

    soln = map.solve(
        flux + flux_err * np.random.randn(*flux.shape),
        theta=theta,
        normalized=True,
        fix_spectrum=True,
        flux_err=flux_err,
        spatial_cov=spatial_cov[i],
    )

    map.show(ax=ax[i], projection="moll", norm=Normalize(vmin=0, vmax=0.5))
    ax[i].annotate(
        r"$\mathrm{SNR} = %.0f$" % snrs[i],
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
    ax[i].set_rasterization_zorder(0)
fig.savefig(paths.figures / "snr.pdf", bbox_inches="tight", dpi=100)
