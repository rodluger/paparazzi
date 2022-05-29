from utils import patch_theano
import matplotlib.pyplot as plt
import numpy as np
import starry
import starry_process
import paths

# Number of samples to draw
nsamples = 20

# Instantiate the Doppler map
wav = np.linspace(642.85, 643.15, 70)
map = starry.DopplerMap(
    lazy=False,
    ydeg=15,
    nc=1,
    vsini_max=50000,
    nt=16,
    wav=wav,
)

# Get the design matrix conditioned on the following properties
map.inc = 40
map.veq = 50000
map.spectrum = (
    1.0
    - 0.85 * np.exp(-0.5 * (map.wav0 - 643.0) ** 2 / 0.0085 ** 2)
    - 0.40 * np.exp(-0.5 * (map.wav0 - 642.97) ** 2 / 0.0085 ** 2)
    - 0.20 * np.exp(-0.5 * (map.wav0 - 643.1) ** 2 / 0.0085 ** 2)
)
D = map.design_matrix(fix_spectrum=True)

# Sample from a GP with certain spot properties
# Note that in general it is more efficient to *compile* the theano
# function (rather than call the `eval` method); see the starry-process
# docs for details!
sp = starry_process.StarryProcess(r=15.0, n=10.0, mu=30.0, sigma=5.0, c=0.25)
np.random.seed(0)
y = sp.sample_ylm(nsamples=nsamples).eval()
y[:, 0] += 1  # starry processes are zero-mean; we need to offset our baseline

# Transform the covariance into spectral space
samples = (D @ y.T).reshape(map.nt, map.nw, nsamples)
samples = np.rollaxis(samples, -1, 0)  # shape (nsamples, nt, nw)

# Normalize the samples
samples /= samples[:, :, 0].reshape(nsamples, map.nt, 1)

# Plot the samples
fig, ax = plt.subplots(4, nsamples // 4, figsize=(12, 10))
ax = ax.flatten()
for n in range(nsamples):
    for k in range(map.nt):
        ax[n].plot(map.wav, 0.05 * k + samples[n][k], "C0-", lw=0.75)

# Appearance
N = nsamples - nsamples // 4
for n in range(nsamples):
    if n != N:
        ax[n].set_xticklabels([])
        ax[n].set_yticklabels([])
        ax[n].set_xlim(*ax[N].get_xlim())
        ax[n].set_ylim(*ax[N].get_ylim())
for tick in ax[N].get_xticklabels() + ax[N].get_yticklabels():
    tick.set_fontsize(8)
ax[N].set_xlabel(r"$\lambda$ [nm]", fontsize=10)
ax[N].set_ylabel(r"intensity", fontsize=10)

fig.savefig(paths.figures / "gp.pdf", bbox_inches="tight")