import starry
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import starry
import george
import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt
from tqdm.auto import tqdm
import json
import sys

# Pass `--clobber` when running this script to force rerun
if "--clobber" in sys.argv:
    clobber = True
else:
    clobber = False

# Dataset settings
flux_err = 1e-4
ydeg = 15
nt = 16
inc = 40
veq = 60000
wav = np.linspace(642.85, 643.15, 200)
wav0 = np.linspace(642.75, 643.25, 200)
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
niter = 100000
lr = 1e-4

# Prior on the spectra
spectral_mean1 = np.ones_like(wav0)
spectral_mean2 = np.ones_like(wav0)
spectral_cov1 = 1e-3 * np.ones_like(wav0)
spectral_cov2 = 1e-3 * np.ones_like(wav0)

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
flux = map.flux(normalize=True)
flux += flux_err * np.random.randn(*flux.shape)

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
    r = pm.Uniform("r", lower=0.0, upper=1.0)

    # Prior on the spectra
    np.random.seed(0)
    spectrum1 = pm.Bound(pm.Normal, upper=1.0)(
        "spectrum1",
        mu=spectral_mean1,
        sigma=np.sqrt(spectral_cov1),
        shape=(map.nw0,),
        testval=1 - np.sqrt(spectral_cov1) * np.abs(np.random.randn(map.nw0)),
    )
    spectrum2 = pm.Bound(pm.Normal, upper=1.0)(
        "spectrum2",
        mu=spectral_mean2,
        sigma=np.sqrt(spectral_cov2),
        shape=(map.nw0,),
        testval=1 - np.sqrt(spectral_cov1) * np.abs(np.random.randn(map.nw0)),
    )
    map.spectrum = tt.concatenate(
        (
            tt.reshape(spectrum1, (1, -1)),
            r * tt.reshape(spectrum2, (1, -1)),
        ),
        axis=0,
    )

    # Compute the model
    flux_model = map.flux()

    # Likelihood term
    pm.Normal(
        "obs",
        mu=tt.reshape(flux_model, (-1,)),
        sd=flux_err,
        observed=flux.reshape(
            -1,
        ),
    )

# Run the optimizer or load the saved MAP solution
if clobber or not Path("twospec_map_soln.json").exists():

    # Optimize!
    loss = []
    best_loss = np.inf
    map_soln = model.test_point
    with model:
        for obj, point in tqdm(
            pmx.optim.optimize_iterator(
                pmx.optim.Adam(lr=lr), niter, start=map_soln
            ),
            total=niter,
        ):
            loss.append(obj)
            if obj < best_loss:
                best_loss = obj
                map_soln = point

    # Save to JSON
    map_soln_json = {}
    for key, value in map_soln.items():
        map_soln_json[key] = value.tolist()
    with open("twospec_map_soln.json", "w") as f:
        json.dump(map_soln_json, f)

else:

    # Load JSON
    map_soln = {}
    with open("twospec_map_soln.json", "r") as f:
        map_soln_json = json.load(f)
    for key, value in map_soln.items():
        map_soln[key] = np.array(value)
