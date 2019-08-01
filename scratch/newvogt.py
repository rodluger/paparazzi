import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
import theano.sparse as ts
from tqdm import tqdm
from scipy.linalg import block_diag as dense_block_diag
from scipy.sparse import block_diag as sparse_block_diag
np.random.seed(13)


def current_best():
    """
    Currently getting loss = 2600 (loss_true = 1500).

    """
    D = pp.Doppler(ydeg=15)
    D.generate_data(ferr=1e-3)


    # DEBUG
    '''
    truth = D.vT_true
    truth -= np.mean(truth)
    fmean = D.T[0].dot(truth)

    fmean = np.mean(D.F, axis=0)
    fmean -= np.mean(fmean)

    A = D.T[0]
    LInv = 1e-8 * np.eye(D.vT_true.shape[0])
    estimate = np.linalg.solve(A.T.dot(A).toarray() + LInv, A.T.dot(fmean))
    plt.figure()
    plt.plot(truth)
    plt.plot(estimate)
    
    plt.figure()
    plt.plot(fmean)
    plt.plot(D.T[0].dot(estimate))

    plt.show()
    quit()
    '''

    D.u = D.u_true
    D.vT = D.vT_true
    loss_true = D.loss()

    fmean = np.mean(D.F, axis=0)
    fmean -= np.mean(fmean)
    A = D.T[0]
    LInv = 1e-5 * np.eye(A.shape[1])
    vT_guess = 1 + np.linalg.solve(A.T.dot(A).toarray() + LInv, A.T.dot(fmean))

    D.vT = vT_guess
    D.compute_u()
    D.compute_vT()


    loss = D.solve(T=1, vT_guess=D.vT, u_guess=D.u, niter=500)
    print(D.loss())

    fig, ax = plt.subplots(1, figsize=(6, 8))
    ax.plot(loss)
    ax.axhline(loss_true, color="C1", ls="--")
    ax.set_yscale("log")

    fig = plt.figure()
    plt.plot(D.vT_true)
    plt.plot(D.vT)

    D.show(projection="rect")
    plt.show()


current_best()
quit()


# Generate data
dop = pp.Doppler(ydeg=1)
dop.generate_data(ferr=1e-5)
dop.u = dop.u_true
dop.vT = dop.vT_true
loss_true = dop.loss()
u11 = dop.u_true[2]

# Priors
u_guess = np.array([-0.1, 0.1, u11])
u_mu = 0.0
u_sig = 0.01
baseline_mu = 1.0
baseline_sig = 0.1

# Vars
dop.u = u_guess
dop.vT = dop.vT_true
u = theano.shared(dop.u[:2])
vT = theano.shared(dop.vT)

# Compute the model (non-linear)

D = ts.as_sparse_variable(dop.D)
a = tt.reshape(tt.dot(tt.reshape(
                      tt.concatenate([[1.0], u, [u11]]), (-1, 1)), 
                      tt.reshape(vT, (1, -1))), (-1,))
dop._map[1:, :] = tt.concatenate([u, [u11]])
b = dop._map.flux(theta=dop.theta)
B = tt.reshape(b, (-1, 1))
M = tt.reshape(ts.dot(D, a), (dop.M, -1)) / B


'''
# Compute the model (bi-linear)
V = sparse_block_diag([dop.vT.reshape(-1, 1) for n in range(dop.N)])
A = np.array(dop.D.dot(V).todense())
B = dop._map.X(theta=dop.theta).eval()
B = np.repeat(B, dop.K, axis=0)
import pdb; pdb.set_trace()
'''

# Compute the loss
r = tt.reshape(dop.F - M, (-1,))
cov = tt.reshape(dop._F_CInv, (-1,))
lnlike = -0.5 * tt.sum(r ** 2 * cov)
lnprior = -0.5 * tt.sum((u - u_mu) ** 2 / u_sig ** 2) + \
            -0.5 * tt.sum((b - baseline_mu) ** 2 / baseline_sig ** 2)
loss = -(lnlike + lnprior)

niter = 1000
u_val = np.empty((niter + 1, 2))
loss_val = np.ones(niter + 1) * np.inf
u_val[0] = u_guess[:2]
loss_val[0] = dop.loss()
upd = pp.utils.NAdam(loss, [u])
train = theano.function([], [u, loss], updates=upd)
for n in tqdm(1 + np.arange(niter)):
    u_val[n], loss_val[n] = train()

# GRID
loss_g = theano.function([], [loss])

# Find the minimum along uA = constant
res = 1000
uA = u_val[-1, 0]
loss_B = np.zeros(res)
eps = 3e-3
uB_arr = np.linspace(u_val[-1, 1] - eps, u_val[-1, 1] + eps, res)
for i, uB in tqdm(enumerate(uB_arr), total=res):
    u.set_value(np.array([uA, uB]))
    loss_B[i] = loss_g()[0]
uB = uB_arr[np.argmin(loss_B)]

# Compute the slope & intercept of the valley
uA_true = dop.u_true[0]
uB_true = dop.u_true[1]
m = (uB - uB_true) / (uA - uA_true)
b = uB_true - m * uA_true

# Plot the loss along the valley
res = 1000
loss_valley = np.zeros(res)
uA_arr = np.linspace(-0.2, 0.2, res)
for i, uA in tqdm(enumerate(uA_arr), total=res):
    uB = m * uA + b
    u.set_value(np.array([uA, uB]))
    loss_valley[i] = loss_g()[0]
fig = plt.figure()
plt.plot(uA_arr, loss_valley)
plt.plot(u_val[-1][0], loss_val[-1], 'o')
plt.yscale("log")

if False:

    # Plot the gradient in the vicinity of the current point
    grad = theano.function([], tt.grad(loss, [u]))
    eps = 1e-3
    extent = (u_val[np.argmin(loss_val)][0] - eps, 
        u_val[np.argmin(loss_val)][0] + eps, 
        u_val[np.argmin(loss_val)][1] - eps, 
        u_val[np.argmin(loss_val)][1] + eps)
    res = 50
    X = np.empty((res, res))
    Y = np.empty((res, res))
    U = np.empty((res, res))
    V = np.empty((res, res))
    L = np.empty((res, res))
    for i, uA in tqdm(enumerate(np.linspace(extent[0], extent[1], res)), total=res):
        for j, uB in enumerate(np.linspace(extent[2], extent[3], res)):
            u.set_value(np.array([uA, uB]))
            U[j, i], V[j, i] = -grad()[0]
            X[j, i] = uA
            Y[j, i] = uB
            L[j, i] = loss_g()[0]
    fig = plt.figure()
    plt.imshow(np.log10(L), origin="lower", extent=extent)
    plt.quiver(X, Y, U, V)
    plt.plot(u_val[np.argmin(loss_val)][0], u_val[np.argmin(loss_val)][1], 'o')
    plt.plot(u_val[:, 0], u_val[:, 1], color="r", lw=0.5)

if True:

    # Brute force loss grid
    extent = (-0.2, 0.2, -0.2, 0.2)
    res = 30
    loss_grid = np.empty((res, res))
    for i, uA in tqdm(enumerate(np.linspace(extent[0], extent[1], res)), total=res):
        for j, uB in enumerate(np.linspace(extent[2], extent[3], res)):
            u.set_value(np.array([uA, uB]))
            loss_grid[j, i] = loss_g()[0]
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(np.log10(loss_grid), origin="lower", extent=extent)
    ax.plot(u_val[:, 0], u_val[:, 1], color="C1")
    ax.plot(u_val[-1, 0], u_val[-1, 1], "o", color="C2")
    ax.plot(dop.u_true[0], dop.u_true[1], 'o', mec="r", mfc='none', ms=10)
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlim(extent[0], extent[1])
    #x = np.linspace(-0.2, 0.2, 10)
    #y = m * x + b
    #ax.plot(x, y, "r--", lw=0.5)

# Loss evol
fig, ax = plt.subplots(1, figsize=(6, 6))
ax.plot(loss_val)
ax.set_yscale("log")
ax.axhline(loss_true, color="C1", ls="--")
plt.show()