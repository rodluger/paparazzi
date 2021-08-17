# -*- coding: utf-8 -*-
"""
Setup for the SPOT problem.

"""
import numpy as np
import matplotlib.pyplot as plt
import starry

# Generate the data
np.random.seed(13)


# Instantiate the Doppler map
ydeg = 15
u = [0.5, 0.25]
udeg = len(u)
inc = 40.0
veq = 60000
nt = 16
map = starry.DopplerMap(ydeg=ydeg, udeg=udeg, nt=nt, inc=inc, veq=veq, lazy=False)
map[1:] = u
map.load(map="spot")
map.spectrum = (
    1.0
    - 0.55 * np.exp(-0.5 * (map.wav0 - 643.0) ** 2 / 0.0085 ** 2)
    - 0.02 * np.exp(-0.5 * (map.wav0 - 642.895) ** 2 / 0.0085 ** 2)
    - 0.10 * np.exp(-0.5 * (map.wav0 - 642.97) ** 2 / 0.0085 ** 2)
    - 0.04 * np.exp(-0.5 * (map.wav0 - 643.1) ** 2 / 0.0085 ** 2)
    - 0.12 * np.exp(-0.5 * (map.wav0 - 643.4) ** 2 / 0.0085 ** 2)
    - 0.08 * np.exp(-0.5 * (map.wav0 - 643.25) ** 2 / 0.0085 ** 2)
    - 0.06 * np.exp(-0.5 * (map.wav0 - 642.79) ** 2 / 0.0085 ** 2)
    - 0.03 * np.exp(-0.5 * (map.wav0 - 642.81) ** 2 / 0.0085 ** 2)
    - 0.18 * np.exp(-0.5 * (map.wav0 - 642.63) ** 2 / 0.0085 ** 2)
    - 0.04 * np.exp(-0.5 * (map.wav0 - 642.60) ** 2 / 0.0085 ** 2)
)

# Render the surface map for plotting
res = 300
theta = np.linspace(-180, 180, nt, endpoint=False)
tmp = starry.Map(ydeg=ydeg, udeg=udeg, inc=inc, lazy=False)
tmp[1:] = u
tmp.load("spot", smoothing=0.05, force_psd=True, oversample=2)
img = tmp.render(projection="rect", res=res)
img_ortho = tmp.render(theta=theta, projection="ortho", res=res)

# Set up the plot
fig = plt.figure(figsize=(11, 9))
fig.subplots_adjust(hspace=0.75)
ax = [
    plt.subplot2grid((33, 8), (0, 0), rowspan=12, colspan=5),
    plt.subplot2grid((33, 8), (0, 5), rowspan=12, colspan=3),
]
ax_ortho = [
    plt.subplot2grid((33, 8), (14, n), rowspan=4, colspan=1) for n in range(8)
]
ax_ortho += [
    plt.subplot2grid((33, 8), (18, n), rowspan=4, colspan=1) for n in range(8)
]
ax_data = plt.subplot2grid((33, 8), (23, 0), rowspan=10, colspan=8)

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
for n in range(16):
    tmp.show(ax=ax_ortho[n], image=img_ortho[n])
    ax_ortho[n].annotate(
        r"$%d^\circ$" % theta[n],
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
lons = np.array([-15.0, -55.0])
sz = 5
intensities = tmp.intensity(lat=lats, lon=lons)
spectra = map.spectrum.reshape(-1, 1).dot(intensities.reshape(1, -1)).T
spectra /= np.max(spectra)
c = [plt.get_cmap("plasma")(0.8), plt.get_cmap("plasma")(0.2)]
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
    ax[1].plot(map.wav0, spectra[n], "-", color=c[n])
    ax[1].annotate(
        "%s" % letters[n],
        xy=(map.wav0[0], intensities[n] / np.max(intensities)),
        xycoords="data",
        xytext=(4, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10,
        color=c[n],
    )
    n += 1

# Tweak
ax[1].set_ylabel(r"intensity", fontsize=10)
ax[1].set_xlabel(r"$\lambda$ [nm]", fontsize=10)
ax[1].set_aspect(0.7)
for tick in ax[1].xaxis.get_major_ticks() + ax[1].yaxis.get_major_ticks():
    tick.label.set_fontsize(10)
ax[1].margins(0, 0.2)

# Compute the spectra
flux = map.flux(theta=theta)

# Plot the data
ax_data.plot(map.wav0, spectra[0])
ax_data_twin = ax_data.twinx()
label = "observed"
for n in range(len(flux)):
    ax_data_twin.plot(
        map.wav,
        flux[n] / flux[n].max(),
        color="C1",
        alpha=0.3,
        label=label,
    )
    label = None
ax_data.axvspan(map.wav0[0], map.wav[0], color="k", alpha=0.3)
ax_data.axvspan(map.wav[-1], map.wav0[-1], color="k", alpha=0.3)

# Tweak
ax_data.set_ylabel(r"rest frame spectrum", fontsize=10)
ax_data_twin.set_ylabel(r"observed spectrum", fontsize=10)
ax_data.set_xlabel(
    r"$\lambda$ [nm]", fontsize=12
)
for axis in [ax_data, ax_data_twin]:
    for tick in axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks():
        tick.label1.set_fontsize(10)
        tick.label2.set_fontsize(10)
ax_data.margins(0, None)
ax_data.set_ylim(0.35, 1.05)
ax_data_twin.set_ylim(0.835, 1.0125)

# We're done!
fig.savefig("spot_setup.pdf", bbox_inches="tight", dpi=300)
