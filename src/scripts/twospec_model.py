from utils import patch_theano
from utils.plot import plot_timeseries, plot_maps, plot_spectra
import starry
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import starry
import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt
from tqdm.auto import tqdm
import os
import paths

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
    r = pm.Uniform("r", lower=0.0, upper=0.75, testval=0.5)

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