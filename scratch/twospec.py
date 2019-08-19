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
ferr = 1.0e-4
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
lnlam = dop.lnlam
lnlam_padded = dop.lnlam_padded
B = dop._map.X(theta=dop.theta).eval()
B = np.repeat(B, K, axis=0)

# Get the Ylm decomposition & the baseline
map = starry.Map(15, lazy=False)
map.inc = 40

# Generate a map, 10% level
map.load("spot")
y_true = np.array(map[:, :])
y_true *= 0.10
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

# Priors
s0_mu = s0_true  # 1.0
s0_sig = 1e-12  # 0.3
s0_rho = 3.0e-5
s1_mu = 1.0
s1_sig = 0.1
s1_rho = 3.0e-5
y_mu = np.zeros(map.N)
y_sig = np.ones(map.N) * 0.01
y_mu[0] = 0.0
y_sig[0] = 0.25
b_mu = 1.0
b_sig = 0.1
dcf = 10.0

# Optimization params
T = 1
dlogT = -0.02
niter = 100
lr = 1e-4

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
def model(y, s0, s1):
    if is_theano(y, s0, s1):
        math = tt
    else:
        math = np

    # Compute the background component
    # TODO: Speed this up
    y0 = np.zeros(map.N)
    y0[0] = 1.0
    A = math.dot(math.reshape(s0, (-1, 1)), math.reshape(y0, (1, -1)))
    a = math.reshape(math.transpose(A), (-1,))
    if math == tt:
        M0 = math.reshape(ts.dot(D, a), (M, -1))
    else:
        M0 = math.reshape(D.dot(a), (M, -1))

    # Compute the variable component
    A = math.dot(math.reshape(s1, (-1, 1)), math.reshape(y, (1, -1)))
    a = math.reshape(math.transpose(A), (-1,))
    if math == tt:
        M1 = math.reshape(ts.dot(D, a), (M, -1))
    else:
        M1 = math.reshape(D.dot(a), (M, -1))

    # Remove the baseline
    b = math.reshape(1.0 + math.dot(B, y), (M, -1))

    return (M0 + M1) / b


# Define the loss function
def loss(y, s0, s1):
    if is_theano(y, s0, s1):
        math = tt
    else:
        math = np
    b = 1.0 + math.dot(B[::K], y)
    r = math.reshape(F - model(y, s0, s1), (-1,))
    lnlike = -0.5 * math.sum(r ** 2 / ferr ** 2)
    lnprior = (
        -0.5 * math.sum((y - y_mu) ** 2 / y_sig ** 2)
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
    )
    return -lnlike, -lnprior


# Re-generate the dataset
F = model(y_true, s0_true, s1_true)
F += ferr * np.random.randn(*F.shape)
like_true, prior_true = loss(y_true, s0_true, s1_true)

# Initialize `s0`
if s0_sig < 1e-10:
    # Initialize `s0` at the prior mean
    s0 = s0_mu
else:
    # Estimate `s0` from the deconvolved spectrum
    fmean = np.mean(F, axis=0)
    fmean -= np.mean(fmean)
    diagonals = np.tile(kT[0].reshape(-1, 1), K)
    offsets = np.arange(W)
    A = diags(diagonals, offsets, (K, Kp), format="csr")
    LInv = dcf ** 2 * ferr ** 2 / s1_sig ** 2 * np.eye(A.shape[1])
    s0 = 1.0 + np.linalg.solve(A.T.dot(A).toarray() + LInv, A.T.dot(fmean))

# Initialize `s1`
if s1_sig < 1e-10:
    # Initialize `s1` at the prior mean
    s1 = s1_mu
else:
    # Estimate `s1` from the deconvolved spectrum
    fmean = np.mean(F, axis=0) - D[:K, :Kp] * s0
    fmean -= np.mean(fmean)
    diagonals = np.tile(kT[0].reshape(-1, 1), K)
    offsets = np.arange(W)
    A = diags(diagonals, offsets, (K, Kp), format="csr")
    LInv = dcf ** 2 * ferr ** 2 / s1_sig ** 2 * np.eye(A.shape[1])
    s1 = 1.0 + np.linalg.solve(A.T.dot(A).toarray() + LInv, A.T.dot(fmean))

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

    # Solve for `y` w/ linear baseline approximation
    S0 = sparse_block_diag([s0.reshape(-1, 1) for j in range(N)])
    Ds0 = np.array(D.dot(S0).todense())[:, 0]
    S1 = sparse_block_diag([s1.reshape(-1, 1) for j in range(N)])
    DS1 = np.array(D.dot(S1).todense())
    X = DS1 - (Ds0.reshape(-1, 1) * B)
    XTCInv = X.T / ferr ** 2 / T
    XTCInvX = XTCInv.dot(X)
    cinv = np.ones(N) / y_sig ** 2
    np.fill_diagonal(XTCInvX, XTCInvX.diagonal() + cinv)
    cho_C = cho_factor(XTCInvX)
    XTXInvy = np.dot(XTCInv, F.reshape(-1) - Ds0.reshape(-1))
    mu = np.ones(N) * y_mu
    y = cho_solve(cho_C, XTXInvy + cinv * mu)

    # Solve for `s0` and `s1`
    offsets = -np.arange(0, N) * Kp
    Y0 = diags(
        [np.ones(Kp)] + [np.zeros(Kp) for j in range(N - 1)],
        offsets,
        shape=(N * Kp, Kp),
    )
    X0 = np.array(D.dot(Y0).todense())
    Y1 = diags(
        [np.ones(Kp) * y[j] for j in range(N)], offsets, shape=(N * Kp, Kp)
    )
    X1 = np.array(D.dot(Y1).todense())
    b = np.reshape(1.0 + np.dot(B, y), (M, -1))
    X = np.hstack((X0, X1)) / b.reshape(-1, 1)
    XTCInv = X.T / ferr ** 2 / T
    XTCInvX = XTCInv.dot(X)
    XTCInvf = np.dot(XTCInv, F.reshape(-1))
    cho_C = cho_factor(XTCInvX + s_CInv)
    s0, s1 = cho_solve(cho_C, XTCInvf + s_CInvmu).reshape(2, -1)

    # Compute the loss
    like_val[i], prior_val[i] = loss(y, s0, s1)

if niter > 0:

    # Non-linear
    print("Running non-linear solver...")

    # Theano nonlinear solve: setup
    y = theano.shared(y)
    s0 = theano.shared(s0)
    s1 = theano.shared(s1)
    like, prior = loss(y, s0, s1)
    best_loss = (like + prior).eval()
    best_y = y.eval()
    best_s0 = s0.eval()
    best_s1 = s1.eval()

    # Variables to optimize
    theano_vars = [y]
    if s0_sig > 1e-10:
        theano_vars += [s0]
    if s1_sig > 1e-10:
        theano_vars += [s1]

    # Optimize
    upd = pp.utils.NAdam(like + prior, theano_vars, lr=lr)
    train = theano.function([], [y, s0, s1, like, prior], updates=upd)
    for i in tqdm(niter_bilin + np.arange(niter)):
        y_val, s0_val, s1_val, like_val[i], prior_val[i] = train()
        if like_val[i] + prior_val[i] < best_loss:
            best_loss = like_val[i] + prior_val[i]
            best_y = y_val
            best_s0 = s0_val
            best_s1 = s1_val
    y = best_y
    s0 = best_s0
    s1 = best_s1

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
plt.plot(model(y, s0, s1).reshape(-1))

# Render the map
map[1:, :] = y[1:] / y[0]
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
