import paths
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# Set up the model
exec(open(paths.scripts / "luhman16b_model.py").read())

# Load
solution = np.load(paths.data / "luhman16b_solution.npz", allow_pickle=True)
loss = solution["loss"]
map_soln = solution["map_soln"].item()

# Plot the loss
fig, ax = plt.subplots(1, figsize=(5, 5))
ax.plot(loss)
ax.set_xlabel("iteration number")
ax.set_ylabel("loss")
fig.savefig(paths.figures / "luhman16b_loss.pdf", bbox_inches="tight")

# Plot the rest frame spectra
fig, ax = plt.subplots(4, figsize=(18, 10), sharey=True)
with model:
    for c in range(4):
        # Mask breaks in the spectrum
        # so matplotlib doesn't linearly
        # interpolate
        dw = np.diff(wav0[c])
        dwm = np.mean(dw)
        dws = np.std(dw)
        x = np.array(wav0[c])
        x[1:][np.abs(dw - dwm) > 3 * dws] = np.nan
        ax[c].plot(x, mean_spectrum[c], label="template")
        ax[c].plot(
            x, pmx.eval_in_model(spectrum[c], point=map_soln), label="MAP"
        )
        ax[c].set_ylabel("intensity")
        ax[c].margins(0, None)
ax[0].legend(loc="lower left")
ax[-1].set_xlabel(r"$\lambda$ [nm]", fontsize=16)
fig.savefig(paths.figures / "luhman16b_spectra.pdf", bbox_inches="tight")

# Plot the data & model
fig, ax = plt.subplots(1, 4, figsize=(16, 10), sharey=True)
fig.subplots_adjust(wspace=0.1)
with model:
    for c in range(4):
        # Mask breaks in the spectrum
        # so matplotlib doesn't linearly
        # interpolate
        dw = np.diff(wav[c])
        dwm = np.mean(dw)
        dws = np.std(dw)
        x = np.array(wav[c])
        x[1:][np.abs(dw - dwm) > 3 * dws] = np.nan
        for k in range(14):
            ax[c].plot(
                x, 0.65 * k + flux[c][k], "k.", ms=3, alpha=0.5, zorder=-1
            )
            ax[c].plot(
                x,
                0.65 * k + pmx.eval_in_model(flux_model[c][k], point=map_soln),
                "C1-",
            )
    # Appearance hacks
    for c in range(4):
        if c > 0:
            ax[c].get_yaxis().set_visible(False)
        ax[c].spines["left"].set_visible(False)
        ax[c].spines["right"].set_visible(False)
        ax[c].spines["top"].set_visible(False)
        ax[c].margins(0, None)
        for tick in ax[c].get_xticklabels() + ax[c].get_yticklabels():
            tick.set_fontsize(10)
        ax[c].xaxis.set_major_formatter("{x:.3f}")
        ax[c].set_rasterization_zorder(0)
    ax[0].set_ylim(0.3, 10)
    ax[0].set_yticks([0, 0.5, 1.0])
    ax[0].set_xlim(ax[0].get_xlim()[0] - 0.0005, ax[0].get_xlim()[1])
    ax[0].set_ylim(*ax[0].get_ylim())
    x0 = ax[0].get_xlim()[0]
    y0 = ax[0].get_ylim()[0]
    ax[0].plot([x0, x0], [y0, 1.25], "k-", clip_on=False, lw=0.75)
    ax[0].set_xticks([2.290, 2.292, 2.294, 2.296, 2.298])
    ax[1].set_xticks([2.306, 2.308, 2.310, 2.312, 2.314])
    ax[2].set_xticks([2.322, 2.324, 2.326, 2.328])
    ax[3].set_xticks([2.336, 2.338, 2.340, 2.342, 2.344])
    plt.annotate(
        r"$\lambda$ [$\mu$m]",
        xy=(0.415, 0.03),
        xycoords="figure fraction",
        ha="center",
        clip_on=False,
    )
fig.savefig(paths.figures / "luhman16b_data_model.pdf", bbox_inches="tight")

# Plot the MAP map
with model:
    y_map = pmx.eval_in_model(y, point=map_soln)
    inc_map = pmx.eval_in_model(inc, point=map_soln)
map_map = starry.Map(ydeg, inc=inc_map)
map_map[:, :] = y_map

# Plot figure similar to that in Crossfield et al. (2014)
times = np.array([0.0, 0.8, 1.6, 2.4, 3.2, 4.1])
thetas = 360 * times / period
fig = plt.figure(figsize=(8, 8))
f = 1 / 0.64
ax = [
    plt.axes([0.3225 * f, 0.34 * f, 0.3125, 0.3125]),
    plt.axes([0.44 * f, 0.17 * f, 0.3125, 0.3125]),
    plt.axes([0.3225 * f, 0.0, 0.3125, 0.3125]),
    plt.axes([0.1175 * f, 0.0, 0.3125, 0.3125]),
    plt.axes([0.0, 0.17 * f, 0.3125, 0.3125]),
    plt.axes([0.1175 * f, 0.34 * f, 0.3125, 0.3125]),
]
norm = Normalize(vmin=0.45, vmax=0.56)
for n, axis in enumerate(ax):
    map_map.show(ax=axis, theta=thetas[n], cmap="gist_heat", norm=norm)
    axis.invert_yaxis()
    axis.invert_xaxis()
    angle = np.pi / 3 * (1 - n)
    plt.annotate(
        "{:.1f} hr".format(times[n]),
        xy=(0.515, 0.43),
        xycoords="figure fraction",
        xytext=(80 * np.cos(angle), 65 * np.sin(angle)),
        textcoords="offset points",
        va="center",
        ha="center",
        fontsize=15,
    )
fig.savefig(paths.figures / "luhman16b_map.pdf", bbox_inches="tight")