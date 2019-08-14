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
dop = pp.Doppler(ydeg=15, vT_sig=0.01)
dop.generate_data(ferr=ferr)

# HACK: Now let's re-generate it with 2 different spectral
# components with weights 0.75 and 0.25, respectively.
l_true = 0.75

# Get the Ylm decomposition & the baseline
map = starry.Map(15, lazy=False)
map.inc = 40
map.load("/Users/rluger/src/paparazzi/paparazzi/vogtstar.jpg")
y1_true = np.array(map[1:, :])
b_true = np.repeat(map.flux(theta=dop.theta), dop.K)
img_true = [None, None]
img_true[0] = map.render(projection="rect", res=res)[0]
map[1:, :] *= -1
img_true[1] = map.render(projection="rect", res=res)[0]

# Generate two different spectra
s_true = np.empty((2, dop.Kp))
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
S = s_true[0].reshape(-1, 1)
Y = np.append([1.0], y1_true).reshape(-1, 1)
A = l_true * S.dot(Y.T)
a = A.T.reshape(-1)
F1 = dop.D().dot(a).reshape(dop.M, -1) / b_true.reshape(dop.M, -1)
S = s_true[1].reshape(-1, 1)
Y = np.append([1.0], -y1_true).reshape(-1, 1)
A = (1 - l_true) * S.dot(Y.T)
a = A.T.reshape(-1)
F2 = dop.D().dot(a).reshape(dop.M, -1) / (1 - (b_true - 1)).reshape(dop.M, -1)
F = F1 + F2
F += ferr * np.random.randn(*F.shape)
dop.F = F

# Initialize
l = 0.5
y1 = np.zeros_like(y1_true)
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

# Tempering schedule
T = 50000
dlogT = -0.1
T_arr = 10 ** np.arange(np.log10(T), 0, dlogT)
T_arr = np.append(T_arr, [1.0])

# DEBUG
s = s_true
# T_arr = [10000.0]

# Iterate
for i in tqdm(range(len(T_arr))):

    # Set the temperature
    T = T_arr[i]

    # Solve for `y1`
    B1 = dop._map.X(theta=dop.theta).eval()[:, 1:]
    B1 = np.repeat(B1, dop.K, axis=0)
    S_1 = sparse_block_diag([s[0].reshape(-1, 1) for j in range(dop.N)])
    tmp = np.array(dop.D().dot(S_1).todense())
    Ds0_1, DS1_1 = tmp[:, 0], tmp[:, 1:]
    Ds0_1 = Ds0_1.reshape(-1, 1)
    S_2 = sparse_block_diag([s[1].reshape(-1, 1) for j in range(dop.N)])
    tmp = np.array(dop.D().dot(S_2).todense())
    Ds0_2, DS1_2 = tmp[:, 0], tmp[:, 1:]
    Ds0_2 = Ds0_2.reshape(-1, 1)
    X = l * DS1_1 - l * (Ds0_1 * B1) - (1 - l) * DS1_2 + (1 - l) * (Ds0_2 * B1)
    z = (l * Ds0_1 + (1 - l) * Ds0_2).reshape(-1)
    XTCInv = np.multiply(X.T, (dop._F_CInv / T).reshape(-1))
    XTCInvX = XTCInv.dot(X)
    cinv = np.ones(dop.N - 1) / dop.u_sig ** 2
    np.fill_diagonal(XTCInvX, XTCInvX.diagonal() + cinv)
    cho_C = cho_factor(XTCInvX)
    XTXInvy = np.dot(XTCInv, dop.F.reshape(-1) - z)
    mu = np.ones(dop.N - 1) * dop.u_mu
    y1 = cho_solve(cho_C, XTXInvy + cinv * mu)

    # Solve for `b`
    b = 1.0 + B1.dot(y1.T).T

    # Solve for `s`
    offsets = -np.arange(0, dop.N) * dop.Kp
    Y = diags(
        [np.ones(dop.Kp)]
        + [np.ones(dop.Kp) * y1[j] for j in range(dop.N - 1)],
        offsets,
        shape=(dop.N * dop.Kp, dop.Kp),
    )
    DYb1 = np.array(dop.D().dot(Y).todense()) / b.reshape(-1, 1)
    X1 = l * DYb1
    Y = diags(
        [np.ones(dop.Kp)]
        + [-np.ones(dop.Kp) * y1[j] for j in range(dop.N - 1)],
        offsets,
        shape=(dop.N * dop.Kp, dop.Kp),
    )
    DYb2 = np.array(dop.D().dot(Y).todense()) / (1 - (b - 1)).reshape(-1, 1)
    X2 = (1 - l) * DYb2
    X = np.hstack((X1, X2))
    XTCInv = np.multiply(X.T, (dop._F_CInv / T).reshape(-1))
    XTCInvX = XTCInv.dot(X)
    XTCInvf = np.dot(XTCInv, dop.F.reshape(-1))
    CInv = cho_solve(dop._vT_cho_C, np.eye(dop.Kp))
    CInv = dense_block_diag(CInv, CInv)
    CInvmu = cho_solve(dop._vT_cho_C, np.ones(dop.Kp) * dop.vT_mu)
    CInvmu = np.tile(CInvmu, 2)
    cho_C = cho_factor(XTCInvX + CInv)
    s = cho_solve(cho_C, XTCInvf + CInvmu).reshape(2, -1)

    # Solve for `L`
    l_mu = 0.5
    l_sig = 0.001
    alpha = DYb1.dot(s[0])
    beta = DYb2.dot(s[1])
    f = dop.F.reshape(-1)
    fcinv = dop._F_CInv.reshape(-1)
    num = ((alpha - beta) * fcinv).dot(f - beta) + 2 * l_mu / l_sig ** 2
    den = ((alpha - beta) * fcinv).dot(alpha - beta) + 2 / l_sig ** 2
    l = num / den

    # DEBUG
    print(l)

# Compute the final model
S = s[0].reshape(-1, 1)
Y = np.append([1.0], y1).reshape(-1, 1)
A = l * S.dot(Y.T)
a = A.T.reshape(-1)
M1 = dop.D().dot(a).reshape(dop.M, -1) / b.reshape(dop.M, -1)
S = s[1].reshape(-1, 1)
Y = np.append([1.0], -y1).reshape(-1, 1)
A = (1 - l) * S.dot(Y.T)
a = A.T.reshape(-1)
M2 = dop.D().dot(a).reshape(dop.M, -1) / (1 - (b - 1)).reshape(dop.M, -1)
M = M1 + M2

# Plot the model
fig = plt.figure()
plt.plot(dop.F.reshape(-1), "k.", alpha=0.3, ms=3)
plt.plot(M.reshape(-1))

# Render the maps
img = [None, None]
map[1:, :] = y1
img[0] = map.render(projection="rect", res=res)[0]
map[1:, :] = -y1
img[1] = map.render(projection="rect", res=res)[0]

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
