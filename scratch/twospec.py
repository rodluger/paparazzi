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
import celerite


def is_theano(*objs):
    """
    Return ``True`` if any of ``objs`` is a ``Theano`` object.

    """
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


# Initialize some stuff
np.random.seed(13)
ferr = 5.0e-4
res = 300
dop = pp.Doppler(ydeg=15)
dop.generate_data(ferr=ferr)
D = dop.D()
kT = dop.kT()
theta = dop.theta
K = dop.K
Kp = dop.Kp
W = dop.W
N = dop.N
M = dop.M
lnlam_padded = dop.lnlam_padded
B1 = dop._map.X(theta=dop.theta).eval()[:, 1:]
B1 = np.repeat(B1, K, axis=0)

# Get the Ylm decomposition & the baseline
map = starry.Map(15, lazy=False)
map.inc = 40

# Add a spot, then subtract the median & reload
map.add_spot(-1.0, sigma=0.25, lat=30)
I = map.render(projection="rect", res=res)[0]
I -= np.nanmedian(I)
I = np.flipud(I)
map.load(I)
y1_true = np.array(map[1:, :])
b_true = np.repeat(map.flux(theta=theta), K)
img_true = map.render(projection="rect", res=res)[0]

# Generate two different spectra
s0_true = np.empty(Kp)
s1_true = np.empty(Kp)
sigma = 7.5e-6
nlines = 21
mu1 = -0.00005
mu2 = 0.00005
s0_true = 1 - 0.5 * np.exp(-0.5 * (lnlam_padded - mu1) ** 2 / sigma ** 2)
for _ in range(nlines - 1):
    amp = 0.1 * np.random.random()
    mu = 2.1 * (0.5 - np.random.random()) * lnlam_padded.max()
    s0_true -= amp * np.exp(-0.5 * (lnlam_padded - mu) ** 2 / sigma ** 2)
s1_true = 1 - 0.5 * np.exp(-0.5 * (lnlam_padded - mu2) ** 2 / sigma ** 2)
for _ in range(nlines - 1):
    amp = 0.1 * np.random.random()
    mu = 2.1 * (0.5 - np.random.random()) * lnlam_padded.max()
    s1_true -= amp * np.exp(-0.5 * (lnlam_padded - mu) ** 2 / sigma ** 2)

# This is the amplitude of the perturbation for the 2nd spectrum
w_true = -0.5

# Priors
w_mu = -0.5
w_sig = 1e-12  # We assume we know this exactly.
s0_mu = s0_true
s0_sig = 1e-12  # We assume we know this exactly.
s0_rho = 0.0
s1_mu = s1_true
s1_sig = 1e-12  # We assume we know this exactly.
s1_rho = 0.0
y1_mu = 0.0
y1_sig = np.array(
    [l ** -2.5 for l in range(1, map.ydeg + 1) for m in range(-l, l + 1)]
)
b_mu = 1.0
b_sig = 0.1
dcf = 10.0

# Optimization params
T = 1.01  # 500000
dlogT = -0.025
niter = 0
lr = 1e-3

# Compute the GP on the spectra
if s0_rho > 0.0:
    kernel = celerite.terms.Matern32Term(np.log(s0_sig), np.log(s0_rho))
    gp = celerite.GP(kernel)
    s0_C = gp.get_matrix(lnlam_padded)
else:
    s0_C = np.eye(Kp) * s0_sig ** 2
s0_cho_C = cho_factor(s0_C)
s0_CInv = cho_solve(s0_cho_C, np.eye(Kp))
s0_CInvmu = cho_solve(s0_cho_C, np.ones(Kp) * s0_mu)
if s1_rho > 0.0:
    kernel = celerite.terms.Matern32Term(np.log(s1_sig), np.log(s1_rho))
    gp = celerite.GP(kernel)
    s1_C = gp.get_matrix(lnlam_padded)
else:
    s1_C = np.eye(Kp) * s1_sig ** 2
s1_cho_C = cho_factor(s1_C)
s1_CInv = cho_solve(s1_cho_C, np.eye(Kp))
s1_CInvmu = cho_solve(s1_cho_C, np.ones(Kp) * s1_mu)
s_CInv = dense_block_diag(s0_CInv, s1_CInv)
s_CInvmu = np.append(s0_CInvmu, s1_CInvmu)


# Define the model
def model(y1, s0, s1, w):
    if is_theano(y1, s0, s1, w):
        math = tt
    else:
        math = np

    # Compute the constant component
    D0 = (D[:, :Kp]).toarray()
    m1 = math.reshape(math.dot(D0, math.reshape(s0, (-1, 1))), (M, -1))

    # Compute the variable component
    A2 = math.dot(
        math.reshape(s1, (-1, 1)),
        math.reshape(math.concatenate([[1.0], y1]), (1, -1)),
    )
    a2 = math.reshape(math.transpose(A2), (-1,))
    if math == tt:
        m2 = math.reshape(ts.dot(D, a2), (M, -1))
    else:
        m2 = math.reshape(D.dot(a2), (M, -1))

    # Remove the baseline
    b = math.reshape(1.0 + math.dot(B1, y1), (M, -1))
    m2 /= b

    return (1 - w) * m1 + w * m2


# Define the loss function
def loss(y1, s0, s1, w):
    if is_theano(y1, s0, s1, w):
        math = tt
    else:
        math = np
    M = model(y1, s0, s1, w)
    b = 1.0 + math.dot(B1[::K], y1)
    r = math.reshape(F - M, (-1,))
    lnlike = -0.5 * math.sum(r ** 2 / ferr ** 2)
    lnprior = (
        -0.5 * math.sum((y1 - y1_mu) ** 2 / y1_sig ** 2)
        - 0.5 * math.sum((b - b_mu) ** 2 / b_sig ** 2)
        - 0.5
        * math.dot(
            math.dot(math.reshape((s0 - s0_mu), (1, -1)), s0_CInv),
            math.reshape((s0 - s0_mu), (-1, 1)),
        )[0, 0]
        - 0.5
        * math.dot(
            math.dot(math.reshape((s1 - s1_mu), (1, -1)), s1_CInv),
            math.reshape((s1 - s1_mu), (-1, 1)),
        )[0, 0]
        - 0.5 * (w - w_mu) ** 2 / w_sig ** 2
    )
    return -lnlike, -lnprior


# Re-generate the dataset
F = model(y1_true, s0_true, s1_true, w_true)
F += ferr * np.random.randn(*F.shape)
like_true, prior_true = loss(y1_true, s0_true, s1_true, w_true)

# Estimate `s0` from the prior mean
s0 = s0_mu

# Estimate `s1` from the deconvolution
fmean = np.mean(F - (D[:, :Kp] * s0).reshape(M, -1), axis=0)
fmean -= np.mean(fmean)
diagonals = np.tile(kT[0].reshape(-1, 1), K)
offsets = np.arange(W)
A = diags(diagonals, offsets, (K, Kp), format="csr")
LInv = dcf ** 2 * ferr ** 2 / s1_sig ** 2 * np.eye(A.shape[1])
s1 = 1.0 + np.linalg.solve(A.T.dot(A).toarray() + LInv, A.T.dot(fmean))

# DEBUG: The method above isn't great
s1 = s1_mu

# Initialize `w` at the prior mean
w = w_mu

# Tempering schedule
if T > 1.0:
    T_arr = 10 ** np.arange(np.log10(T), 0, dlogT)
    T_arr = np.append(T_arr, [1.0])
else:
    T_arr = np.array([1.0])
niter_bilin = len(T_arr)

# Iterate
print("Running bi-linear solver...")
like_val = np.zeros(niter_bilin + niter)
prior_val = np.zeros(niter_bilin + niter)
for i in tqdm(range(niter_bilin)):

    # Set the temperature
    T = T_arr[i]

    # Solve for `y1` w/ linear baseline approximation
    M1 = D[:, :Kp] * s0
    S_2 = sparse_block_diag([s1.reshape(-1, 1) for j in range(N)])
    tmp = np.array(D.dot(S_2).todense())
    Ds0_2, DS1_2 = tmp[:, 0], tmp[:, 1:]
    X = w * (Ds0_2.reshape(-1, 1) * B1 + DS1_2)
    z = F.reshape(-1) - (1 - w) * M1 - w * Ds0_2
    XTCInv = X.T / ferr ** 2 / T
    XTCInvX = XTCInv.dot(X)
    cinv = np.ones(N - 1) / y1_sig ** 2
    np.fill_diagonal(XTCInvX, XTCInvX.diagonal() + cinv)
    cho_C = cho_factor(XTCInvX)
    XTXInvy = np.dot(XTCInv, z)
    mu = np.ones(N - 1) * y1_mu
    y1 = cho_solve(cho_C, XTXInvy + cinv * mu)

    # Solve for `s0` and `s1`
    b = 1.0 + B1.dot(y1)
    offsets = -np.arange(0, N) * Kp
    Y = diags(
        [np.ones(Kp)] + [np.zeros(Kp) for j in range(N - 1)],
        offsets,
        shape=(N * Kp, Kp),
    )
    X1 = (1 - w) * np.array(D.dot(Y).todense())
    Y = diags(
        [np.ones(Kp)] + [np.ones(Kp) * y1[j] for j in range(N - 1)],
        offsets,
        shape=(N * Kp, Kp),
    )
    X2 = w * np.array(D.dot(Y).todense()) / b.reshape(-1, 1)
    X = np.hstack((X1, X2))
    XTCInv = X.T * (w ** 2 / ferr ** 2 / T)
    XTCInvX = XTCInv.dot(X)
    XTCInvf = np.dot(XTCInv, F.reshape(-1))
    cho_C = cho_factor(XTCInvX + s_CInv)
    s0, s1 = cho_solve(cho_C, XTCInvf + s_CInvmu).reshape(2, -1)

    # Solve for w?
    D0 = (D[:, :Kp]).toarray()
    M1 = np.reshape(np.dot(D0, np.reshape(s0, (-1, 1))), (M, -1))
    A2 = np.dot(
        np.reshape(s1, (-1, 1)),
        np.reshape(np.concatenate([[1.0], y1]), (1, -1)),
    )
    a2 = np.reshape(np.transpose(A2), (-1,))
    if np == tt:
        M2 = np.reshape(ts.dot(D, a2), (M, -1))
    else:
        M2 = np.reshape(D.dot(a2), (M, -1))
    b = np.reshape(1.0 + np.dot(B1, y1), (M, -1))
    M2 /= b
    X = (M2 - M1).reshape(-1)
    z = (F - M1).reshape(-1)
    num = X.dot(z) / ferr ** 2 + w_mu / w_sig ** 2
    den = X.dot(X) / ferr ** 2 + 1.0 / w_sig ** 2
    w = num / den

    # Compute the loss
    like_val[i], prior_val[i] = loss(y1, s0, s1, w)

if niter > 0:

    # Non-linear
    print("Running non-linear solver...")

    # Theano nonlinear solve: setup
    y1 = theano.shared(y1)
    s0 = theano.shared(s0)
    s1 = theano.shared(s1)
    w = theano.shared(w)
    like, prior = loss(y1, s0, s1, w)
    best_loss = (like + prior).eval()
    best_y1 = y1.eval()
    best_s0 = s0.eval()
    best_s1 = s1.eval()
    best_w = w.eval()

    # Optimize
    upd = pp.utils.NAdam(like + prior, [y1], lr=lr)
    train = theano.function([], [y1, s0, s1, w, like, prior], updates=upd)
    for i in tqdm(niter_bilin + np.arange(niter)):
        y1_val, s0_val, s1_val, w_val, like_val[i], prior_val[i] = train()
        if like_val[i] + prior_val[i] < best_loss:
            best_loss = like_val[i] + prior_val[i]
            best_y1 = y1_val
            best_s0 = s0_val
            best_s1 = s1_val
            best_w = w_val
    y1 = best_y1
    s0 = best_s0
    s1 = best_s1
    w = best_w

# Plot the loss
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(14, 4))
ax[0].plot(like_val)
ax[0].axhline(like_true, color="C1", ls="--")
ax[1].plot(prior_val)
ax[1].axhline(prior_true, color="C1", ls="--")
ax[2].plot(like_val + prior_val)
ax[2].axhline(like_true + prior_true, color="C1", ls="--")
ax[0].set_yscale("log")

# Plot the model
fig = plt.figure()
plt.plot(F.reshape(-1), "k.", alpha=0.3, ms=3)
plt.plot(model(y1, s0, s1, w).reshape(-1))

# Render the maps
img = None
map[1:, :] = y1
img = map.render(projection="rect", res=res)[0]

# Plot the results
fig = plt.figure(figsize=(10, 4))
ax = [
    plt.subplot2grid((2, 5), (0, 0), rowspan=1, colspan=2),
    plt.subplot2grid((2, 5), (1, 0), rowspan=1, colspan=2),
    plt.subplot2grid((2, 5), (0, 2), rowspan=1, colspan=3),
    plt.subplot2grid((2, 5), (1, 2), rowspan=1, colspan=3),
]
norm = np.max(img_true)
for n, img in enumerate([img_true, img]):
    ax[n].imshow(
        img / norm,
        origin="lower",
        extent=(-180, 180, -90, 90),
        cmap="plasma",
        vmin=0,
        vmax=1,
    )
ax[2].plot(lnlam_padded, s0_true)
ax[2].plot(lnlam_padded, s0)
ax[3].plot(lnlam_padded, s1_true)
ax[3].plot(lnlam_padded, s1)

plt.show()

# WOOT!
