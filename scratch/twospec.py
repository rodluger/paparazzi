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
import theano
import theano.tensor as tt
import theano.sparse as ts
from tqdm import tqdm
from inspect import getmro

# Initialize some stuff
niter = 100
lr = 1e-5
np.random.seed(13)
ferr = 1.0e-4
res = 300
dop = pp.Doppler(ydeg=15, vT_sig=0.1)
dop.generate_data(ferr=ferr, ntheta=32)
B1 = dop._map.X(theta=dop.theta).eval()[:, 1:]
B1 = np.repeat(B1, dop.K, axis=0)


def is_theano(*objs):
    """
    Return ``True`` if any of ``objs`` is a ``Theano`` object.

    """
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


def model(y1, s):
    # Theano or numpy?
    if is_theano(y1, s):
        math = tt
    else:
        math = np

    # Compute the baseline
    b = 1.0 + math.dot(B1, y1)

    # Compute the first component
    A1 = math.dot(
        math.reshape(s[0], (-1, 1)),
        math.reshape(math.concatenate([[1.0], y1]), (1, -1)),
    )
    a1 = math.reshape(math.transpose(A1), (-1,))
    if math == tt:
        M1 = math.reshape(ts.dot(dop.D(), a1), (dop.M, -1))
    else:
        M1 = math.reshape(dop.D().dot(a1), (dop.M, -1))
    M1 /= math.reshape(b, (dop.M, -1))

    # Compute the second component
    A2 = math.dot(
        math.reshape(s[1], (-1, 1)),
        math.reshape(math.concatenate([[1.0], -y1]), (1, -1)),
    )
    a2 = math.reshape(math.transpose(A2), (-1,))
    if math == tt:
        M2 = math.reshape(ts.dot(dop.D(), a2), (dop.M, -1))
    else:
        M2 = math.reshape(dop.D().dot(a2), (dop.M, -1))
    M2 /= math.reshape((1 - (b - 1)), (dop.M, -1))

    return M1 + M2


def loss(y1, s):
    # Theano or numpy?
    if is_theano(y1, s):
        math = tt
    else:
        math = np
    M = model(y1, s)
    b = 1.0 + math.dot(B1[:: dop.K], y1)
    r = math.reshape(dop.F - M, (-1,))
    cov = math.reshape(dop._F_CInv, (-1,))
    lnlike = -0.5 * math.sum(r ** 2 * cov)
    lnprior = (
        -0.5 * math.sum((y1 - dop.u_mu) ** 2 / dop.u_sig ** 2)
        - 0.5 * math.sum((b - dop.baseline_mu) ** 2 / dop.baseline_sig ** 2)
        - 0.5
        * math.dot(
            math.dot(math.reshape((s[0] - 0.75), (1, -1)), dop._vT_CInv),
            math.reshape((s[0] - 0.75), (-1, 1)),
        )[0, 0]
        - 0.5
        * math.dot(
            math.dot(math.reshape((s[1] - 0.25), (1, -1)), dop._vT_CInv),
            math.reshape((s[1] - 0.25), (-1, 1)),
        )[0, 0]
    )
    return -(lnlike + lnprior)


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
    mu = 2.1 * (0.5 - np.random.random()) * dop.lnlam_padded.max()
    s_true[0] -= amp * np.exp(-0.5 * (dop.lnlam_padded - mu) ** 2 / sigma ** 2)
s_true[1] = 1 - 0.5 * np.exp(-0.5 * (dop.lnlam_padded - mu2) ** 2 / sigma ** 2)
for _ in range(nlines - 1):
    amp = 0.1 * np.random.random()
    mu = 2.1 * (0.5 - np.random.random()) * dop.lnlam_padded.max()
    s_true[1] -= amp * np.exp(-0.5 * (dop.lnlam_padded - mu) ** 2 / sigma ** 2)
s_true[0] *= l_true
s_true[1] *= 1 - l_true

# Re-generate the dataset
dop.F = model(y1_true, s_true)
dop.F += ferr * np.random.randn(*dop.F.shape)
loss_true = loss(y1_true, s_true)

# Initialize
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
s[0] = 0.8 * s_guess
s[1] = 0.2 * s_guess

# Tempering schedule
T = 50000
dlogT = -0.025
T_arr = 10 ** np.arange(np.log10(T), 0, dlogT)
T_arr = np.append(T_arr, [1.0])
niter_bilin = len(T_arr)

# Iterate
print("Running bi-linear solver...")
loss_val = np.zeros(niter_bilin + niter)
for i in tqdm(range(niter_bilin)):

    # Set the temperature
    T = T_arr[i]

    # Solve for `y1`
    S_1 = sparse_block_diag([s[0].reshape(-1, 1) for j in range(dop.N)])
    tmp = np.array(dop.D().dot(S_1).todense())
    Ds0_1, DS1_1 = tmp[:, 0], tmp[:, 1:]
    Ds0_1 = Ds0_1.reshape(-1, 1)
    S_2 = sparse_block_diag([s[1].reshape(-1, 1) for j in range(dop.N)])
    tmp = np.array(dop.D().dot(S_2).todense())
    Ds0_2, DS1_2 = tmp[:, 0], tmp[:, 1:]
    Ds0_2 = Ds0_2.reshape(-1, 1)
    X = DS1_1 - (Ds0_1 * B1) - DS1_2 + (Ds0_2 * B1)
    z = (Ds0_1 + Ds0_2).reshape(-1)
    XTCInv = np.multiply(X.T, (dop._F_CInv / T).reshape(-1))
    XTCInvX = XTCInv.dot(X)
    cinv = np.ones(dop.N - 1) / dop.u_sig ** 2
    np.fill_diagonal(XTCInvX, XTCInvX.diagonal() + cinv)
    cho_C = cho_factor(XTCInvX)
    XTXInvy = np.dot(XTCInv, dop.F.reshape(-1) - z)
    mu = np.ones(dop.N - 1) * dop.u_mu
    y1 = cho_solve(cho_C, XTXInvy + cinv * mu)

    # Solve for `s`
    b = 1.0 + B1.dot(y1)
    offsets = -np.arange(0, dop.N) * dop.Kp
    Y = diags(
        [np.ones(dop.Kp)]
        + [np.ones(dop.Kp) * y1[j] for j in range(dop.N - 1)],
        offsets,
        shape=(dop.N * dop.Kp, dop.Kp),
    )
    X1 = np.array(dop.D().dot(Y).todense()) / b.reshape(-1, 1)
    Y = diags(
        [np.ones(dop.Kp)]
        + [-np.ones(dop.Kp) * y1[j] for j in range(dop.N - 1)],
        offsets,
        shape=(dop.N * dop.Kp, dop.Kp),
    )
    X2 = np.array(dop.D().dot(Y).todense()) / (1 - (b - 1)).reshape(-1, 1)
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

    # Compute the loss
    loss_val[i] = loss(y1, s)

# Theano nonlienar solve: setup
y1 = theano.shared(y1)
s = theano.shared(s)
theano_loss = loss(y1, s)
best_loss = theano_loss.eval()
best_y1 = y1.eval()
best_s = s.eval()

# Optimize
print("Running non-linear solver...")
upd = pp.utils.NAdam(theano_loss, [y1, s], lr=lr)
train = theano.function([], [y1, s, theano_loss], updates=upd)
for i in tqdm(niter_bilin + np.arange(niter)):
    y1_val, s_val, loss_val[i] = train()
    if loss_val[i] < best_loss:
        best_loss = loss_val[i]
        best_y1 = y1_val
        best_s = s_val
y1 = best_y1
s = best_s

# Plot the loss
fig = plt.figure()
plt.plot(loss_val)
plt.axhline(loss_true, color="C1", ls="--")
plt.yscale("log")

# Compute the final model
M = model(y1, s)
print(loss(y1, s))

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
