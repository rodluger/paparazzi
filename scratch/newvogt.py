import paparazzi as pp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import celerite
from scipy.linalg import cho_factor
np.random.seed(13)


def learn_everything(high_snr=False):
    """

    """
    # High or low SNR?
    if high_snr:
        ferr = 1e-4
        niter_bilinear = 25
        niter_adam = 2000
    else:
        ferr = 1e-3
        niter_bilinear = 6
        niter_adam = 0

    # Generate data
    D = pp.Doppler(ydeg=15)
    D.generate_data(ferr=ferr)

    # Compute loss @ true value
    D.u = D.u_true
    D.vT = D.vT_true
    loss_true = D.loss()

    # Estimate the rest frame spectrum from the deconvolved mean spectrum
    fmean = np.mean(D.F, axis=0)
    fmean -= np.mean(fmean)
    A = D.T[0]
    LInv = 1e-4 * np.eye(A.shape[1])
    vT_guess = 1 + np.linalg.solve(A.T.dot(A).toarray() + LInv, A.T.dot(fmean))
    D.vT = vT_guess

    # Pre-compute the Cholesky factorization of the GP for vT
    sig = 0.3
    rho = 3.e-5
    kernel = celerite.terms.Matern32Term(np.log(sig), np.log(rho))
    gp = celerite.GP(kernel)
    C = gp.get_matrix(D.lnlam_padded)
    cho_C = cho_factor(C)

    # Iterative Bi-linear solve
    loss = np.zeros(niter_bilinear)
    for i in tqdm(range(niter_bilinear)):
        D.compute_u()
        D.compute_vT(cho_C=cho_C)
        loss[i] = D.loss()
    
    # Refine with NAdam, slow learning rate
    if niter_adam > 0:
        loss = np.append(loss, 
            D.solve(vT_guess=D.vT, u_guess=D.u, niter=niter_adam, lr=1e-4))

    # Plot the results
    fig, ax = plt.subplots(1, figsize=(6, 8))
    ax.plot(loss)
    ax.axhline(loss_true, color="C1", ls="--")
    ax.set_yscale("log")

    fig = plt.figure()
    plt.plot(D.vT_true)
    plt.plot(D.vT)

    fig = plt.figure()
    plt.plot(D.F.reshape(-1))
    plt.plot(D.model.reshape(-1))

    D.show(projection="rect")
    plt.show()

learn_everything()