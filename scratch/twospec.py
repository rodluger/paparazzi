import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import starry
import subprocess
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg import block_diag as dense_block_diag
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import diags
import theano.sparse as ts
from tqdm import tqdm

np.random.seed(13)
ferr = 1.0e-4
res = 300

# Generate a dataset
dop = pp.Doppler(ydeg=15)
dop.generate_data(ferr=ferr, ntheta=32)

# HACK: Now let's re-generate it with 2 different spectral
# components with weights 0.75 and 0.25, respectively.
ncomp = 2
L_true = np.array([0.75, 0.25])

# Get the Ylm decomposition for each component & the baseline
y_true = np.empty((ncomp, dop.N - 1))
b_true = np.empty((ncomp, dop.M * dop.K))
img_true = [None for n in range(ncomp)]
map = starry.Map(15, lazy=False)
map.inc = 40
map.load("/Users/rluger/src/paparazzi/paparazzi/vogtstar.jpg")
y_true[0] = np.array(map[1:, :])
b_true[0] = np.repeat(map.flux(theta=dop.theta), dop.K)
img_true[0] = map.render(projection="rect", res=res)[0]
map[1:, :] *= -1
y_true[1] = np.array(map[1:, :])
b_true[1] = np.repeat(map.flux(theta=dop.theta), dop.K)
img_true[1] = map.render(projection="rect", res=res)[0]

# Generate two different spectra
s_true = np.empty((ncomp, dop.Kp))
sigma = 7.5e-6
nlines = 21
mu1 = -0.00005
mu2 = 0.00005
s_true[0] = 1 - 0.5 * np.exp(-0.5 * (dop.lnlam_padded - mu1) ** 2 / sigma ** 2)
for _ in range(nlines - 1):
    amp = 0.1 * np.random.random()
    mu = 1.5 * (0.5 - np.random.random()) * dop.lnlam_padded.max()
    s_true[0] -= amp * np.exp(-0.5 * (dop.lnlam_padded - mu) ** 2 / sigma ** 2)
s_true[1] = 1 - 0.5 * np.exp(-0.5 * (dop.lnlam_padded - mu2) ** 2 / sigma ** 2)
for _ in range(nlines - 1):
    amp = 0.1 * np.random.random()
    mu = 1.5 * (0.5 - np.random.random()) * dop.lnlam_padded.max()
    s_true[1] -= amp * np.exp(-0.5 * (dop.lnlam_padded - mu) ** 2 / sigma ** 2)

# Re-generate the dataset
F = [None for n in range(ncomp)]
for n in range(ncomp):
    S = s_true[n].reshape(-1, 1)
    Y = np.append([1], y_true[n]).reshape(-1, 1)
    A = L_true[n] * S.dot(Y.T)
    a = A.T.reshape(-1)
    F[n] = dop.D().dot(a).reshape(dop.M, -1) / b_true[n].reshape(dop.M, -1)
F = np.sum(F, axis=0)
F += ferr * np.random.randn(*F.shape)
dop.F = F

# Initialize
L = np.array([0.5, 0.5])
y = np.zeros_like(y_true)
s = np.zeros_like(s_true)
b = np.zeros_like(b_true)

# Estimate `s`
dcf = 10.0
fmean = np.mean(dop.F, axis=0)
fmean -= np.mean(fmean)
diagonals = np.tile(dop.kT()[0].reshape(-1, 1), dop.K)
offsets = np.arange(dop.W)
A = diags(diagonals, offsets, (dop.K, dop.Kp), format="csr")
LInv = dcf ** 2 * dop.ferr ** 2 / dop.vT_sig ** 2 * np.eye(A.shape[1])
s_guess = 1.0 + np.linalg.solve(A.T.dot(A).toarray() + LInv, A.T.dot(fmean))
s[:] = s_guess

T = 50000
dlogT = -0.1
T_arr = 10 ** np.arange(np.log10(T), 0, dlogT)
T_arr = np.append(T_arr, [1.0])
for i in tqdm(range(len(T_arr))):

    # Set the temperature
    T = T_arr[i]

    # Solve for `y`
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
    y = cho_solve(cho_C, XTXInvy + cinv * mu).reshape(ncomp, -1)

    # Solve for `b`
    b = 1 + B.dot(y.T).T

    # Solve for `s`
    DYb = [None for n in range(ncomp)]
    X = [None for n in range(ncomp)]
    offsets = -np.arange(0, dop.N) * dop.Kp
    for n in range(ncomp):
        Y = diags(
            [np.ones(dop.Kp)]
            + [np.ones(dop.Kp) * y[n, j] for j in range(dop.N - 1)],
            offsets,
            shape=(dop.N * dop.Kp, dop.Kp),
        )
        DYb[n] = np.array(dop.D().dot(Y).todense()) / b[n].reshape(-1, 1)
        X[n] = L[n] * DYb[n]
    X = np.hstack(X)
    XTCInv = np.multiply(X.T, (dop._F_CInv / T).reshape(-1))
    XTCInvX = XTCInv.dot(X)
    XTCInvf = np.dot(XTCInv, dop.F.reshape(-1))
    CInv = cho_solve(dop._vT_cho_C, np.eye(dop.Kp))
    CInv = dense_block_diag(*[CInv for n in range(ncomp)])
    CInvmu = cho_solve(dop._vT_cho_C, np.ones(dop.Kp) * dop.vT_mu)
    CInvmu = np.tile(CInvmu, ncomp)
    cho_C = cho_factor(XTCInvX + CInv)
    s = cho_solve(cho_C, XTCInvf + CInvmu).reshape(ncomp, -1)

    # Solve for `L`
    l_mu = 0.5
    l_sig = 0.05
    alpha = DYb[0].dot(s[0])
    beta = DYb[1].dot(s[1])
    f = dop.F.reshape(-1)
    fcinv = dop._F_CInv.reshape(-1)
    num = ((alpha - beta) * fcinv).dot(f - beta) + 2 * l_mu / l_sig ** 2
    den = ((alpha - beta) * fcinv).dot(alpha - beta) + 2 / l_sig ** 2
    L[0] = num / den
    L[1] = 1 - L[0]

    print(L)

plt.plot(dop.F.reshape(-1), "k.", alpha=0.3, ms=3)
model = [None for n in range(ncomp)]
for n in range(ncomp):
    S = s[n].reshape(-1, 1)
    Y = np.append([1], y[n]).reshape(-1, 1)
    A = L[n] * S.dot(Y.T)
    a = A.T.reshape(-1)
    model[n] = dop.D().dot(a).reshape(dop.M, -1) / b[n].reshape(dop.M, -1)
model = np.sum(model, axis=0)
plt.plot(model.reshape(-1))

# Render the maps
img = [None for n in range(ncomp)]
for n in range(ncomp):
    map[1:, :] = y[n]
    img[n] = map.render(projection="rect", res=res)[0]

# Plot the results
fig = plt.figure(figsize=(15, 4))
ax = [
    plt.subplot2grid((2, 7), (0, 0), rowspan=1, colspan=2),
    plt.subplot2grid((2, 7), (1, 0), rowspan=1, colspan=2),
    plt.subplot2grid((2, 7), (0, 2), rowspan=1, colspan=2),
    plt.subplot2grid((2, 7), (1, 2), rowspan=1, colspan=2),
    plt.subplot2grid((2, 7), (0, 4), rowspan=1, colspan=3),
    plt.subplot2grid((2, 7), (1, 4), rowspan=1, colspan=3),
]
norm = np.max(img_true)
for n, img in enumerate([img_true[0], img_true[1], img[0], img[1]]):
    ax[n].imshow(
        img / norm,
        origin="lower",
        extent=(-180, 180, -90, 90),
        cmap="plasma",
        vmin=0,
        vmax=1,
    )
ax[4].plot(dop.lnlam_padded, s_true[0])
ax[4].plot(dop.lnlam_padded, s[0])
ax[5].plot(dop.lnlam_padded, s_true[1])
ax[5].plot(dop.lnlam_padded, s[1])

plt.show()

# WOOT!
