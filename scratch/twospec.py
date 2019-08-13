import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import starry
import subprocess
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import block_diag as sparse_block_diag
import theano.sparse as ts

np.random.seed(13)
ferr = 1.0e-4

# Generate a dataset
dop = pp.Doppler(ydeg=15)
dop.generate_data(ferr=ferr)

# HACK: Now let's re-generate it with 2 different spectral
# components with weights 0.75 and 0.25, respectively.
ncomp = 2
L = np.array([0.75, 0.25])

# Get the Ylm decomposition for each component & the baseline
y = np.empty((ncomp, dop.N - 1))
b = np.empty((ncomp, dop.M))
map = starry.Map(15, lazy=False)
map.inc = 40
map.load("/Users/rluger/src/paparazzi/paparazzi/vogtstar.jpg")
y[0] = np.array(map[1:, :])
b[0] = map.flux(theta=dop.theta)
map[1:, :] *= -1
y[1] = np.array(map[1:, :])
b[1] = map.flux(theta=dop.theta)

# Generate two different spectra
s = np.empty((ncomp, dop.Kp))
sigma = 7.5e-6
nlines = 21
mu1 = -0.00005
mu2 = 0.00005
s[0] = 1 - 0.5 * np.exp(-0.5 * (dop.lnlam_padded - mu1) ** 2 / sigma ** 2)
for _ in range(nlines - 1):
    amp = 0.1 * np.random.random()
    mu = 1.5 * (0.5 - np.random.random()) * dop.lnlam_padded.max()
    s[0] -= amp * np.exp(-0.5 * (dop.lnlam_padded - mu) ** 2 / sigma ** 2)
s[1] = 1 - 0.5 * np.exp(-0.5 * (dop.lnlam_padded - mu2) ** 2 / sigma ** 2)
for _ in range(nlines - 1):
    amp = 0.1 * np.random.random()
    mu = 1.5 * (0.5 - np.random.random()) * dop.lnlam_padded.max()
    s[1] -= amp * np.exp(-0.5 * (dop.lnlam_padded - mu) ** 2 / sigma ** 2)

# Re-generate the dataset
F = [None for n in range(ncomp)]
for n in range(ncomp):
    S = s[n].reshape(-1, 1)
    Y = np.append([1], y[n]).reshape(-1, 1)
    A = L[n] * S.dot(Y.T)
    a = A.T.reshape(-1)
    F[n] = dop.D().dot(a).reshape(dop.M, -1) / b[n].reshape(-1, 1)
F = np.sum(F, axis=0)
F += ferr * np.random.randn(*F.shape)
dop.F = F

# Solve for `y`
T = 1.0
X = [None for n in range(ncomp)]
z = [None for n in range(ncomp)]
B = dop._map.X(theta=dop.theta).eval()[:, 1:]
B = np.repeat(B, dop.K, axis=0)
for n in range(ncomp):
    S = sparse_block_diag([s[n].reshape(-1, 1) for j in range(dop.N)])
    DS_ = np.array(dop.D().dot(S).todense())
    Ds0, DS1 = DS_[:, 0], DS_[:, 1:]
    C = Ds0.reshape(-1, 1) * B
    X[n] = L[n] * (DS1 - C)
    z[n] = L[n] * Ds0
X = np.hstack(X)
z = np.sum(z, axis=0)
XTCInv = np.multiply(X.T, (dop._F_CInv / T).reshape(-1))
XTCInvX = XTCInv.dot(X)
cinv = np.ones(ncomp * (dop.N - 1)) / dop.u_sig ** 2
np.fill_diagonal(XTCInvX, XTCInvX.diagonal() + cinv)
cho_C = cho_factor(XTCInvX)
XTXInvy = np.dot(XTCInv, dop.F.reshape(-1) - z)
mu = np.ones(ncomp * (dop.N - 1)) * dop.u_mu
yhat = cho_solve(cho_C, XTXInvy + cinv * mu)

# Plot
map[1:, :] = yhat[:255]
img1 = map.render(projection="rect")[0]
map[1:, :] = yhat[255:]
img2 = map.render(projection="rect")[0]
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(
    img1,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
    vmin=0,
    vmax=1,
)
ax[0, 1].plot(dop.lnlam_padded, s[0])
ax[1, 0].imshow(
    img2,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
    vmin=0,
    vmax=1,
)
ax[1, 1].plot(dop.lnlam_padded, s[1])
plt.show()

# WOOT!
