"""
Show that we can solve the Doppler imaging problem
when the stellar spectrum is a linear combination
of two different "eigenspectra".

Specifically, there is one spectral component (s0) that
is present everywhere at the same intensity, and one
component (s1) whose weight varies spatially with the
spot. This has the effect of causing the spot to have
one spectrum and the rest of the star to have a
different one.

"""
import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt
import starry
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
from matplotlib.patches import Rectangle


def is_theano(*objs):
    """
    Return ``True`` if any of ``objs`` is a ``Theano`` object.

    """
    for obj in objs:
        for c in getmro(type(obj)):
            if c is theano.gof.graph.Node:
                return True
    return False


# Initialize some stuff. We'll instantiate a ``Doppler``
# object so we can use its properties (mostly the convolution
# kernels & Doppler design matrix) later on. We're going
# to re-generate the data below, but we call ``generate_data``
# to force computation of the various things we'll need.
np.random.seed(13)
ferr = 1.0e-4
res = 300
dop = pp.Doppler(ydeg=15, u=[0.5, 0.25])
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
B1 = dop._map.X(theta=dop.theta).eval()[:, 1:]
B1 = np.repeat(B1, K, axis=0)

# Generate a spot map, get the baseline, and render the image
map = starry.Map(15)
map.inc = 40
map.load("spot")
y1_true = np.array(map[1:, :].eval())
b_true = np.repeat(map.flux(theta=theta).eval(), K)
img_true = map.render(projection="rect", res=res).eval()

# Generate two spectra with lines of different depths
# and cenetered at different wavelengths
mu1 = -0.00015
mu2 = 0.00015
sigma = 1.4e-5  # ~ 10 km / s
line1 = -0.5 * np.exp(-0.5 * (lnlam_padded - mu1) ** 2 / sigma ** 2)
line2 = -0.9 * np.exp(-0.5 * (lnlam_padded - mu2) ** 2 / sigma ** 2)

# Construct the two "eigenspectra".
# There's no particular logic here; I simply found a linear combination of the
# two vectors above that gives me plausible-looking spectra in the spot and in
# the background. In the end it doesn't really matter how I construct this.
s0_true = 1 + line1 + line2
s1_true = 1 + line1 - 0.75 * line2

# Priors: assuming we know NOTHING, and placing generous
# Gaussian priors on things.
s0_mu = 1.0
s0_sig = 0.01
s0_rho = 3.0e-5
s1_mu = 1.0
s1_sig = 0.01
s1_rho = 3.0e-5
y1_mu = 0
y1_sig = 0.01
b_sig = 0.1
dcf = 10.0

# Optimization params. Our data is pretty high SNR,
# so we need to do some tempering, plus a little
# nonlinear refinement.
T = 5000.0
dlogT = -0.025
niter = 2000
lr = 3e-4

# Pre-compute the GP on the spectral components
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
def model(y1, s0, s1):
    if is_theano(y1, s0, s1):
        math = tt
    else:
        math = np

    # Compute the background component
    A = math.dot(
        math.reshape(s0, (-1, 1)),
        math.reshape(
            math.concatenate(([1.0], math.zeros_like(y1)), axis=0), (1, -1)
        ),
    )
    a = math.reshape(math.transpose(A), (-1,))
    if math == tt:
        M0 = math.reshape(ts.dot(D, a), (M, -1))
    else:
        M0 = math.reshape(D.dot(a), (M, -1))

    # Compute the spot component
    A = math.dot(
        math.reshape(s1, (-1, 1)),
        math.reshape(math.concatenate(([1.0], y1), axis=0), (1, -1)),
    )
    a = math.reshape(math.transpose(A), (-1,))
    if math == tt:
        M1 = math.reshape(ts.dot(D, a), (M, -1))
    else:
        M1 = math.reshape(D.dot(a), (M, -1))

    # Remove the baseline
    b = math.reshape(2.0 + math.dot(B1, y1), (M, -1))

    return (M0 + M1) / b


# Define the loss function
def loss(y1, s0, s1):
    if is_theano(y1, s0, s1):
        math = tt
    else:
        math = np
    b_rel = math.dot(B1[::K], y1)
    r = math.reshape(F - model(y1, s0, s1), (-1,))
    lnlike = -0.5 * math.sum(r ** 2 / ferr ** 2)
    lnprior = (
        -0.5 * math.sum((y1 - y1_mu) ** 2 / y1_sig ** 2)
        - 0.5 * math.sum(b_rel ** 2 / b_sig ** 2)
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


# Now, actually generate the dataset
F = model(y1_true, s0_true, s1_true)
F += ferr * np.random.randn(*F.shape)
like_true, prior_true = loss(y1_true, s0_true, s1_true)

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
best_loss = np.inf
best_y1 = np.zeros(N - 1)
best_s0 = s0
best_s1 = s1
for i in tqdm(range(niter_bilin)):

    # Set the temperature
    T = T_arr[i]

    # Solve for `y1` w/ linear baseline approximation
    S0 = sparse_block_diag([s0.reshape(-1, 1) for j in range(N)])
    Ds0 = np.array(D.dot(S0).todense())[:, 0]
    S1 = sparse_block_diag([s1.reshape(-1, 1) for j in range(N)])
    tmp = np.array(D.dot(S1).todense())
    Ds1, DS1 = tmp[:, 0], tmp[:, 1:]
    X = DS1 - ((Ds0 + Ds1).reshape(-1, 1) * B1 * 0.5)
    XTCInv = X.T / (2 * ferr) ** 2 / T
    XTCInvX = XTCInv.dot(X)
    cinv = np.ones(N - 1) / y1_sig ** 2
    np.fill_diagonal(XTCInvX, XTCInvX.diagonal() + cinv)
    cho_C = cho_factor(XTCInvX)
    XTXInvy = np.dot(XTCInv, 2 * F.reshape(-1) - (Ds0 + Ds1).reshape(-1))
    mu = np.ones(N - 1) * y1_mu
    y1 = cho_solve(cho_C, XTXInvy + cinv * mu)
    b = np.reshape(2.0 + np.dot(B1, y1), (M, -1))

    # Solve for `s0` and `s1`
    offsets = -np.arange(0, N) * Kp
    Y0 = diags(
        [np.ones(Kp)] + [np.zeros(Kp) for j in range(N - 1)],
        offsets,
        shape=(N * Kp, Kp),
    )
    X0 = np.array(D.dot(Y0).todense())
    Y1 = diags(
        [np.ones(Kp)] + [np.ones(Kp) * y1[j] for j in range(N - 1)],
        offsets,
        shape=(N * Kp, Kp),
    )
    X1 = np.array(D.dot(Y1).todense())
    X = np.hstack((X0, X1)) / b.reshape(-1, 1)
    XTCInv = X.T / ferr ** 2 / T
    XTCInvX = XTCInv.dot(X)
    XTCInvf = np.dot(XTCInv, F.reshape(-1))
    cho_C = cho_factor(XTCInvX + s_CInv)
    s0, s1 = cho_solve(cho_C, XTCInvf + s_CInvmu).reshape(2, -1)

    # Compute the loss
    like_val[i], prior_val[i] = loss(y1, s0, s1)

    # Did it improve?
    if like_val[i] + prior_val[i] < best_loss:
        best_loss = like_val[i] + prior_val[i]
        best_y1 = y1
        best_s0 = s0
        best_s1 = s1

# Set to best values
y1 = best_y1
s0 = best_s0
s1 = best_s1

if niter > 0:

    # Non-linear
    print("Running non-linear solver...")

    # Theano nonlinear solve: setup
    y1 = theano.shared(y1)
    s0 = theano.shared(s0)
    s1 = theano.shared(s1)
    like, prior = loss(y1, s0, s1)
    best_loss = (like + prior).eval()
    best_y1 = y1.eval()
    best_s0 = s0.eval()
    best_s1 = s1.eval()

    # Variables to optimize
    theano_vars = [y1]
    if s0_sig > 1e-10:
        theano_vars += [s0]
    if s1_sig > 1e-10:
        theano_vars += [s1]

    # Optimize
    upd = pp.utils.NAdam(like + prior, theano_vars, lr=lr)
    train = theano.function([], [y1, s0, s1, like, prior], updates=upd)
    for i in tqdm(niter_bilin + np.arange(niter)):
        y1_val, s0_val, s1_val, like_val[i], prior_val[i] = train()
        if like_val[i] + prior_val[i] < best_loss:
            best_loss = like_val[i] + prior_val[i]
            best_y1 = y1_val
            best_s0 = s0_val
            best_s1 = s1_val
    y1 = best_y1
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
fig.savefig("twospec_loss.pdf", bbox_inches="tight")

# Print for the record
print("True loss: %.2f" % (like_true + prior_true))
print("Best loss: %.2f" % np.min(like_val + prior_val))

# Plot the model
fig = plt.figure()
plt.plot(F.reshape(-1), "k.", alpha=0.3, ms=3)
plt.plot(model(y1, s0, s1).reshape(-1))
fig.savefig("twospec_data_model.pdf", bbox_inches="tight")

# Render the inferred map
map[1:, :] = y1
img = map.render(projection="rect", res=res).eval()

# Get the full map matrix, `A` (true)
A0_true = np.dot(
    np.reshape(s0_true, (-1, 1)),
    np.reshape(np.append([1.0], np.zeros(map.N - 1)), (1, -1)),
).T
A1_true = np.dot(
    np.reshape(s1_true, (-1, 1)),
    np.reshape(np.append([1.0], y1_true), (1, -1)),
).T
A_true = A0_true + A1_true

# Get the full map matrix, `A` (inferred)
A0 = np.dot(
    np.reshape(s0, (-1, 1)),
    np.reshape(np.append([1.0], np.zeros(map.N - 1)), (1, -1)),
).T
A1 = np.dot(
    np.reshape(s1, (-1, 1)), np.reshape(np.append([1.0], y1_true), (1, -1))
).T
A = A0 + A1

# Points where we'll evaluate the spectrum. One corresponds to
# a place within the spot; the other is outside the spot.
lats = np.array([10.0, 25.0])
lons = np.array([-15.0, -62.0])

# Get the change of basis matrix from Ylm to intensity, `P`
xyz = map.ops.latlon_to_xyz(lats * np.pi / 180.0, lons * np.pi / 180.0)
P = map.ops.pT(xyz[0], xyz[1], xyz[2])
P = ts.dot(P, map.ops.A1).eval()

# Compute the local spectrum at each point
I_true = P.dot(A_true)
I = P.dot(A)

# Set up the plot
fig = plt.figure(figsize=(15, 6))
ax = [
    plt.subplot2grid((2, 10), (0, 0), rowspan=1, colspan=4),
    plt.subplot2grid((2, 10), (1, 0), rowspan=1, colspan=4),
    plt.subplot2grid((2, 10), (0, 4), rowspan=1, colspan=3),
    plt.subplot2grid((2, 10), (1, 4), rowspan=2, colspan=3),
    plt.subplot2grid((2, 10), (0, 7), rowspan=1, colspan=3),
    plt.subplot2grid((2, 10), (1, 7), rowspan=1, colspan=3),
]

# Show the images
vmax = np.nanmax(img_true)
ax[0].imshow(
    img_true / vmax,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
    vmin=0,
    vmax=1,
)
ax[1].imshow(
    img / vmax,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
    vmin=0,
    vmax=1,
)
ax[0].set_ylabel("true", fontsize=22)
ax[1].set_ylabel("inferred", fontsize=22)
for axis in [ax[0], ax[1]]:
    latlines = np.linspace(-90, 90, 7)[1:-1]
    lonlines = np.linspace(-180, 180, 13)
    for lat in latlines:
        axis.axhline(lat, color="k", lw=0.5, alpha=0.5, zorder=100)
    for lon in lonlines:
        axis.axvline(lon, color="k", lw=0.5, alpha=0.5, zorder=100)
    axis.set_xticks(lonlines)
    axis.set_yticks(latlines)
    for tick in axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)

# Plot the spectra @ the evaluation points
sz = 5
n = 0
map[1:, :] = y1_true
intensities = map.intensity(x=xyz[0], y=xyz[1]).eval()
intensities /= vmax
c = [plt.get_cmap("plasma")(i) for i in intensities]
letters = ["A", "B"]
for lat, lon in zip(lats, lons):
    for i in [0, 1]:
        ax[i].plot(
            [lon - sz, lon - sz],
            [lat - sz, lat + sz],
            "-",
            color="w",
            zorder=101,
        )
        ax[i].plot(
            [lon + sz, lon + sz],
            [lat - sz, lat + sz],
            "-",
            color="w",
            zorder=101,
        )
        ax[i].plot(
            [lon - sz, lon + sz],
            [lat - sz, lat - sz],
            "-",
            color="w",
            zorder=101,
        )
        ax[i].plot(
            [lon - sz, lon + sz],
            [lat + sz, lat + sz],
            "-",
            color="w",
            zorder=101,
        )
        ax[i].annotate(
            "%s" % letters[n],
            xy=(lon + sz, lat + sz),
            xycoords="data",
            xytext=(2, -2),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10,
            color="w",
        )
    for i, intens in zip([2, 3], [I_true, I]):
        ax[i].plot(lnlam_padded, intens[n] / intens[0, 0], "-", color=c[n])
        ax[i].annotate(
            "%s" % letters[n],
            xy=(lnlam_padded[0], intens[n, 0] / intens[0, 0]),
            xycoords="data",
            xytext=(4, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=10,
            color=c[n],
        )
    n += 1
ax[2].set_title("local spectra", fontsize=22, y=1.1)

# Plot the "eigenspectra"
ax[4].plot(lnlam_padded, s0_true, "C0")
ax[4].plot(lnlam_padded, s1_true, "C2")
ax[5].plot(lnlam_padded, s0, "C0")
ax[5].plot(lnlam_padded, s1, "C2")
for n, dy in zip([0, 1], [4, -14]):
    for i in [4, 5]:
        ax[i].annotate(
            r"$s_{%d}$" % (n + 1),
            xy=(lnlam_padded[0], 1),
            xycoords="data",
            xytext=(4, dy),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10,
            color="C%d" % (n * 2),
        )
ax[4].set_title("components", fontsize=22, y=1.1)

for i in [2, 3, 4, 5]:
    ax[i].margins(0, None)
    ax[i].axis("off")

fig.savefig("twospec.pdf", bbox_inches="tight")
