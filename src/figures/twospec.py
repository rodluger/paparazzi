from utils.plot import plot_timeseries, plot_maps, plot_spectra
import starry
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import starry
import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt
from tqdm.auto import tqdm
import os


np.random.seed(0)


# Dataset settings
flux_err = 1e-4
ydeg = 15
nt = 16
inc = 40
veq = 60000
wav = np.linspace(642.85, 643.15, 70)
wav0 = np.linspace(642.74, 643.26, 300)
theta = np.linspace(-180, 180, nt, endpoint=False)
u = [0.5, 0.25]


# True maps
starry_path = Path(starry.__file__).parents[0]
image1 = np.mean(
    np.flipud(plt.imread(starry_path / "img" / "spot.png"))[:, :, :3],
    axis=2,
)
image2 = 1 - image1


# True intensity ratio (spot / photosphere)
ratio = 0.5


# True spectra (photosphere and spot)
spectrum1 = 1.0 - 0.925 * np.exp(-0.5 * (wav0 - 643.0) ** 2 / 0.0085 ** 2)
spectrum2 = (
    1.0
    - 0.63 * np.exp(-0.5 * (wav0 - 642.97) ** 2 / 0.0085 ** 2)
    - 0.6 * np.exp(-0.5 * (wav0 - 643.08) ** 2 / 0.0085 ** 2)
)


# Optimization settings
niter = 20000  # Number of iterations
lr = 1e-2  # Adam learning rate
sb = 1e-3  # Laplacian regularization strength for the spectrum
wl = 1e-2  # GP lengthscale for the spectrum (nm)
wa = 1e-1  # GP amplitude for the spectrum


# Instantiate the map
map = starry.DopplerMap(
    ydeg=ydeg,
    udeg=len(u),
    nc=2,
    veq=veq,
    inc=inc,
    nt=nt,
    wav=wav,
    wav0=wav0,
    lazy=False,
    vsini_max=40000,
)
map.load(
    maps=[image1, image2],
    spectra=[spectrum1, ratio * spectrum2],
    smoothing=0.075,
)
for n in range(len(u)):
    map[1 + n] = u[n]


# Generate the dataset
flux = map.flux(theta=theta)
flux += flux_err * np.random.randn(*flux.shape)


# Save it to a dict for later
data = dict(
    kwargs=dict(
        ydeg=ydeg,
        udeg=len(u),
        nc=2,
        veq=veq,
        inc=inc,
        vsini_max=40000,
        nt=nt,
        wav=wav,
        wav0=wav0,
    ),
    props=dict(u=u),
    truths=dict(y=map.y, spectrum=map.spectrum),
    data=dict(
        theta=theta,
        flux_err=flux_err,
        flux=flux,
    ),
)


# Set up a pymc3 model so we can optimize
with pm.Model() as model:

    # Instantiate a uniform map
    map = starry.DopplerMap(
        ydeg=ydeg,
        udeg=len(u),
        nc=2,
        veq=veq,
        inc=inc,
        nt=nt,
        wav=wav,
        wav0=wav0,
        lazy=True,
        vsini_max=40000,
    )
    for n in range(len(u)):
        map[1 + n] = u[n]

    # SHT matrix: converts from pixels to Ylms
    A = map.sht_matrix(smoothing=0.075)
    npix = A.shape[1]

    # Prior on the maps
    p = pm.Uniform("p", lower=0.0, upper=1.0, shape=(npix,))
    amp = pm.Uniform("amp", lower=0.0, upper=1.0)
    y1 = amp * tt.dot(A, p)
    y2 = amp * tt.dot(A, (1 - p))
    map[:, :, 0] = y1
    map[:, :, 1] = y2

    # Prior on the intensity ratio
    r = pm.Uniform("r", lower=0.0, upper=0.75)

    # Prior on the spectra
    np.random.seed(0)
    spectrum1 = pm.Bound(pm.Laplace, lower=0.0, upper=1.0 + 1e-4)(
        "spectrum1",
        mu=1.0,
        b=sb,
        shape=(map.nw0,),
        testval=1 - 1e-2 * np.abs(np.random.randn(map.nw0)),
    )
    spectrum2 = pm.Bound(pm.Laplace, lower=0.0, upper=1.0 + 1e-4)(
        "spectrum2",
        mu=1.0,
        b=sb,
        shape=(map.nw0,),
        testval=1 - 1e-2 * np.abs(np.random.randn(map.nw0)),
    )
    map.spectrum = tt.concatenate(
        (
            tt.reshape(spectrum1, (1, -1)),
            r * tt.reshape(spectrum2, (1, -1)),
        ),
        axis=0,
    )

    # Enforce smoothness with a Squared Exponential GP on the spectra
    kernel = wa ** 2 * pm.gp.cov.ExpQuad(1, wl)
    cov = kernel(wav0[:, None]).eval() + 1e-6 * np.eye(map.nw0)
    smoothness1 = pm.Potential(
        "smoothness1",
        pm.MvNormal.dist(mu=np.ones(map.nw0), cov=cov).logp(spectrum1),
    )
    smoothness2 = pm.Potential(
        "smoothness2",
        pm.MvNormal.dist(mu=np.ones(map.nw0), cov=cov).logp(spectrum2),
    )

    # Compute the model
    flux_model = map.flux(theta=theta)

    # Likelihood term
    pm.Normal(
        "obs",
        mu=tt.reshape(flux_model, (-1,)),
        sd=flux_err,
        observed=flux.reshape(
            -1,
        ),
    )


# Optimize!
loss = []
best_loss = np.inf
map_soln = model.test_point
iterator = tqdm(
    pmx.optim.optimize_iterator(pmx.optim.Adam(lr=lr), niter, start=map_soln),
    total=niter,
    disable=os.getenv("CI", "false") == "true",
)
with model:
    for obj, point in iterator:
        iterator.set_description(
            "loss: {:.3e} / {:.3e}".format(obj, best_loss)
        )
        loss.append(obj)
        if obj < best_loss:
            best_loss = obj
            map_soln = point


# Plot the loss
loss = np.array(loss)
logloss = np.log10(loss)
logloss[loss < 0] = -np.log10(-loss[loss < 0])
fig, ax = plt.subplots(1)
ax.plot(np.arange(len(loss)), logloss, lw=1)
ax.set_ylabel("log-ish loss")
ax.set_xlabel("iteration")
fig.savefig("twospec_loss.pdf", bbox_inches="tight")


# Get the solution
with model:
    y_inferred = pmx.eval_in_model(map.y, point=map_soln)
    spectrum_inferred = pmx.eval_in_model(map.spectrum, point=map_soln)


# Plot the maps
fig = plot_maps(data["truths"]["y"][:, 0], y_inferred[:, 0], figsize=(8, 7.5))
fig.savefig("twospec_maps.pdf", bbox_inches="tight", dpi=300)


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
fig.savefig("twospec_spectra1.pdf", bbox_inches="tight", dpi=300)


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
fig.savefig("twospec_spectra2.pdf", bbox_inches="tight", dpi=300)


# Plot the timeseries
fig = plot_timeseries(
    data,
    y_inferred,
    spectrum_inferred,
    normalized=True,
    overlap=5,
    figsize=(5, 7.5),
)
fig.savefig("twospec_timeseries.pdf", bbox_inches="tight", dpi=300)