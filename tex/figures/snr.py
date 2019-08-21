"""
Investigate how our recovered map varies with the
signal to noise ratio of the data.

"""
import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt


# Instantiate
res = 300
dop = pp.Doppler(ydeg=15, u=[0.5, 0.25])

# Compute model at infinite SNR to get the "signal"
# This is the standard deviation in each wavelength
# bin across all epochs.
dop.generate_data(ferr=0)
F_true = np.array(dop.F)
signal = np.mean(np.std(F_true, axis=0))

# Compute the pointwise uncertainty for a given SNR
# NOTE: Vogt et al. (1987) have ~35 resolution elements
# on the Fe line, and they compute the SNR per pixel,
# where each pixel is half a resolution element. We
# have 201 pixels in our model, so we need to *increase*
# the uncertainty on our data by a factor of sqrt(201 / 70)
# to get comparable results. This is a pretty tiny effect.
snrs = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
ferrs = signal / snrs
ferrs *= np.sqrt(F_true.shape[1] / 70.0)

# Compute the normalization for the image
dop.y1 = dop.y1_true
img_true = dop.render(projection="rect", res=res).reshape(res, res)
vmax = np.max(img_true)
img = [None for ferr in ferrs]

for i, ferr in enumerate(ferrs):

    # Generate data
    np.random.seed(13)
    dop.generate_data(ferr=ferr)

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
        vmax=1.0,
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
        r"$\mathrm{SNR} = %.0f$" % snrs[i],
        xy=(0, 0),
        xytext=(7, 7),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=18,
        color="w",
        zorder=101,
    )
fig.savefig("snr.pdf", bbox_inches="tight")
