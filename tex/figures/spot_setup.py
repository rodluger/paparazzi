# -*- coding: utf-8 -*-
"""
Setup for the SPOT problem.

"""
import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt

# Generate the data
np.random.seed(13)
dop = pp.Doppler(ydeg=15, u=[0.5, 0.25])
dop.generate_data(ferr=1e-4)

# Render the map
dop.s = dop.s_true
dop.y1 = dop.y1_true
map = dop._map
res = 300
img = map.render(projection="rect", res=res).eval()[0]
img_ortho = map.render(theta=dop.theta, projection="ortho", res=res).eval()

# Set up the plot
fig = plt.figure(figsize=(11, 6))
fig.subplots_adjust(hspace=0.75)
ax = [
    plt.subplot2grid((11, 8), (0, 0), rowspan=6, colspan=5),
    plt.subplot2grid((11, 8), (0, 5), rowspan=6, colspan=3),
]
ax_ortho = [
    plt.subplot2grid((11, 8), (7, n), rowspan=2, colspan=1) for n in range(8)
]
ax_ortho += [
    plt.subplot2grid((11, 8), (9, n), rowspan=2, colspan=1) for n in range(8)
]

# Show the rect image
ax[0].imshow(img, origin="lower", extent=(-180, 180, -90, 90), cmap="plasma")
latlines = np.linspace(-90, 90, 7)[1:-1]
lonlines = np.linspace(-180, 180, 13)
for lat in latlines:
    ax[0].axhline(lat, color="k", lw=0.5, alpha=0.5, zorder=100)
for lon in lonlines:
    ax[0].axvline(lon, color="k", lw=0.5, alpha=0.5, zorder=100)
ax[0].set_xticks(lonlines)
ax[0].set_yticks(latlines)
for tick in ax[0].xaxis.get_major_ticks() + ax[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
ax[0].set_xlabel("Longitude [deg]", fontsize=10)
ax[0].set_ylabel("Latitude [deg]", fontsize=10)

# Show the ortho images
vmin = np.nanmin(img_ortho)
vmax = np.nanmax(img_ortho)
for n in range(16):
    ax_ortho[n].imshow(
        img_ortho[n], origin="lower", cmap="plasma", vmin=vmin, vmax=vmax
    )
    ax_ortho[n].axis("off")
    ax_ortho[n].annotate(
        r"$%d^\circ$" % dop.theta[n],
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(4, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6,
        color="k",
    )

# Points where we'll evaluate the spectrum. One corresponds to
# a place within the spot; the other is outside the spot.
lats = np.array([10.0, 25.0])
lons = np.array([-15.0, -60.0])
sz = 5
intensities = map.intensity(lat=lats, lon=lons).eval()
spectra = dop.s.reshape(-1, 1).dot(intensities.reshape(1, -1)).T
spectra /= np.max(spectra)
c = [plt.get_cmap("plasma")(i) for i in intensities]
letters = ["A", "B"]
n = 0
for lat, lon in zip(lats, lons):
    ax[0].plot(
        [lon - sz, lon - sz], [lat - sz, lat + sz], "-", color="w", zorder=101
    )
    ax[0].plot(
        [lon + sz, lon + sz], [lat - sz, lat + sz], "-", color="w", zorder=101
    )
    ax[0].plot(
        [lon - sz, lon + sz], [lat - sz, lat - sz], "-", color="w", zorder=101
    )
    ax[0].plot(
        [lon - sz, lon + sz], [lat + sz, lat + sz], "-", color="w", zorder=101
    )
    ax[0].annotate(
        "%s" % letters[n],
        xy=(lon + sz, lat + sz),
        xycoords="data",
        xytext=(2, -2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10,
        color="w",
    )
    ax[1].plot(dop.lnlam_padded, spectra[n], "-", color=c[n])
    ax[1].annotate(
        "%s" % letters[n],
        xy=(dop.lnlam_padded[0], intensities[n] / np.max(intensities)),
        xycoords="data",
        xytext=(4, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10,
        color=c[n],
    )
    n += 1

# Tweak & save
ax[1].set_ylabel(r"intensity", fontsize=10)
ax[1].set_xlabel(r"$\ln\left(\lambda/\ln_0\right)$", fontsize=10)
ax[1].set_aspect(7e-4)
for tick in ax[1].xaxis.get_major_ticks() + ax[1].yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
ax[1].margins(0, 0.2)
ax[1].set_xticks([-0.0003, 0.0, 0.0003])
fig.savefig("spot_setup.pdf", bbox_inches="tight", dpi=400)
