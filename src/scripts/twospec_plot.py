import paths


# Set up the model
exec(open(paths.scripts / "twospec_model.py").read())

# Load
solution = np.load(paths.data / "twospec_solution.npz", allow_pickle=True)
loss = solution["loss"]
map_soln = solution["map_soln"].item()

# Plot the loss
loss = np.array(loss)
logloss = np.log10(loss)
logloss[loss < 0] = -np.log10(-loss[loss < 0])
fig, ax = plt.subplots(1)
ax.plot(np.arange(len(loss)), logloss, lw=1)
ax.set_ylabel("log-ish loss")
ax.set_xlabel("iteration")
fig.savefig(paths.figures / "twospec_loss.pdf", bbox_inches="tight")


# Get the solution
with model:
    y_inferred = pmx.eval_in_model(map.y, point=map_soln)
    spectrum_inferred = pmx.eval_in_model(map.spectrum, point=map_soln)


# Plot the maps
fig = plot_maps(data["truths"]["y"][:, 0], y_inferred[:, 0], figsize=(8, 7.5))
fig.savefig(paths.figures / "twospec_maps.pdf", bbox_inches="tight", dpi=300)


# Mask portions of the spectrum that don't contribute to the
# observed flux; we can't know anything about those.
mask = np.ones(map.nw0, dtype=bool)
mask[map._unused_idx] = False


# Plot spectrum 1
fig = plot_spectra(
    wav,
    wav0[mask],
    data["truths"]["spectrum"][0][mask],
    spectrum1.tag.test_value[mask],
    spectrum_inferred[0][mask],
    figsize=(8, 2),
)
fig.gca().set_ylabel("background spectrum", fontsize=12)
fig.savefig(
    paths.figures / "twospec_spectra1.pdf", bbox_inches="tight", dpi=300
)


# Plot spectrum 2
fig = plot_spectra(
    wav,
    wav0[mask],
    data["truths"]["spectrum"][1][mask],
    r.tag.test_value * spectrum2.tag.test_value[mask],
    spectrum_inferred[1][mask],
    figsize=(8, 2),
)
fig.gca().set_ylabel("spot spectrum", fontsize=12)
fig.savefig(
    paths.figures / "twospec_spectra2.pdf", bbox_inches="tight", dpi=300
)


# Plot the timeseries
fig = plot_timeseries(
    data,
    y_inferred,
    spectrum_inferred,
    normalized=True,
    overlap=5,
    figsize=(5, 7.5),
)
fig.savefig(
    paths.figures / "twospec_timeseries.pdf", bbox_inches="tight", dpi=300
)
