# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


def randu(lo=0, hi=1, size=1):
    """
    Return random numbers sampled uniformly between
    ``lo`` and ``hi``.

    """
    return lo + (hi - lo) * np.random.rand(size)


def I_num(xi, I0, beta=0):
    """
    Return the shifted spectrum at a point, computed numerically.
    
    """
    alpha = np.log(1 - beta)
    xi0 = xi + alpha
    return np.interp(xi0, xi, I0)


def S_num(xi, I0, beta=0, npts=300):
    """
    Return the disk-integrated spectrum, computed numerically.
    
    """
    S = np.zeros_like(xi)
    A = 0
    for x in np.linspace(-1, 1, npts + 2)[1:-1]:
        for y in np.linspace(-1, 1, npts + 2)[1:-1]:
            if (x ** 2 + y ** 2) <= 1:
                S += I_num(xi, I0, beta=beta * x)
                A += 1
    return np.pi * S / A


def S(xi, I0, beta=0):
    """
    Return the disk-integrated spectrum, computed analytically
    with a convolution.
    
    """
    gam = ((1 - np.exp(xi)) / beta)
    xsq = gam ** 2
    integ = 2 * np.sqrt(1 - xsq[xsq <= 1])
    res = convolve(I0, integ, mode='same')
    norm = np.pi / convolve(np.ones_like(I0), integ, mode='same')
    return res * norm


def plot_figure(beta=0.05, amp=[1.0], mu=[1.0], sigma=[0.01],
                npts=300, lam_range=(0.9, 1.1),
                pad=0, nlam=1000, noise=0.0):
    """
    Plot the figure for the paper.

    """
    # An evenly sampled timeseries in xi = log(wavelength)
    # Add padding to remove edge effects
    xi = np.linspace(np.log(lam_range[0] - pad), 
                     np.log(lam_range[1] + pad), nlam)
    lam = np.exp(xi)
    inds = (lam >= lam_range[0]) & (lam <= lam_range[1])

    # Gaussian absorption lines
    I0 = np.ones_like(lam)
    for a, m, s in zip(amp, mu, sigma):
        I0 -= a * np.exp(-0.5 * (lam - m) ** 2 / s ** 2)
    
    # Add noise
    I0[inds] += noise * np.random.randn(len(I0[inds]))

    # New figure
    fig, ax = plt.subplots(2, figsize=(6, 6), sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0.1)
    ax[0].set_xlim(*lam_range)

    # Plot the shifted line
    ax[0].plot(lam, I0, 'k', lw=1)
    ax[0].plot(lam, I_num(xi, I0, beta=beta), 'C3', 
               lw=1, alpha=0.5)
    ax[0].plot(lam, I_num(xi, I0, beta=-beta), 'C0', 
               lw=1, alpha=0.5)
    ax[0].set_ylabel("local intensity")

    # Plot the disk-integrated line
    ax[1].plot(lam, S_num(xi, I0, beta=beta, npts=npts) / np.pi, 'C0', 
               label="Broadened (numerical)")
    ax[1].plot(lam, S(xi, I0, beta=beta) / np.pi, 'C1--', 
               label="Broadened (FFT)")
    ax[1].legend(fontsize=8, loc="lower left")
    ax[1].set_xlabel("wavelength [arbitrary units]")
    ax[1].set_ylabel("integrated intensity")
    ymax = ax[1].get_ylim()[1] + 0.01
    ymin = ax[1].get_ylim()[0] - 0.05
    ax[1].set_ylim(ymin, ymax)

    # Save
    fig.savefig("convolve.pdf", bbox_inches="tight")





if __name__ == "__main__":
    np.random.seed(12)
    npts = 300
    beta = 0.01
    nlines = 5
    amp = np.ones(nlines)
    lam_range = (0.9, 1.1)


    mu = randu(lam_range[0], lam_range[1], nlines)
    sigma = 10 ** (-4 + 1 * np.random.rand(nlines))
    nlam = 999
    pad = 0.05
    noise = 0.0

    plot_figure(beta=beta, amp=amp, mu=mu, sigma=sigma, lam_range=lam_range, 
                npts=npts, nlam=nlam, noise=noise, pad=pad)