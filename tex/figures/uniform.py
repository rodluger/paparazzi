# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, gamma
import mpmath

def _hyp2f1(a, b, c, z):
    """
    The Hypergeometric function 2F1 for 
    complex-valued arguments.
    
    """
    res = mpmath.hyp2f1(a, b, c, z)
    return float(res.real) + float(res.imag) * 1j

# Vectorize it
hyp2f1 = np.vectorize(_hyp2f1)


def randu(lo=0, hi=1, size=1):
    """
    Return random numbers sampled uniformly between
    ``lo`` and ``hi``.

    """
    return lo + (hi - lo) * np.random.rand(size)


def I_num(xi, I0, alpha=0):
    """
    Return the shifted spectrum, computed numerically.
    
    """
    xi0 = xi + alpha
    return np.interp(xi0, xi, I0)


def S_num(xi, I0, alpha=0, npts=1000):
    """
    Return the disk-integrated spectrum, computed numerically.
    
    """
    S = np.zeros_like(xi)
    A = 0
    beta = np.exp(alpha) - 1
    for x in np.linspace(-1, 1, npts + 2)[1:-1]:
        beta_x = beta * x
        alpha_x = np.log(1 + beta_x)
        S += I_num(xi, I0, alpha=alpha_x) * np.sqrt(1 - x ** 2)
        A += np.sqrt(1 - x ** 2)
    return np.pi * S / A


def I(xi, I0, alpha=0):
    """
    Return the shifted spectrum, computed via an FFT.
    
    """
    # Number of wavelength bins
    N = len(xi)
    
    # Take the FFT
    fI0 = np.fft.rfft(I0)
    k = np.fft.rfftfreq(N, xi[1] - xi[0])
    
    # Apply the translation
    fI0 *= np.exp(2 * np.pi * alpha * 1j * k)
    
    # Take the inverse FFT and return
    return np.fft.irfft(fI0, N)


def S(xi, I0, alpha=0, method="exact", order=2, kmax=None):
    """
    Return the disk-integrated spectrum, computed via an FFT.
    This uses the non-relativistic Doppler formula.

    """
    # Number of wavelength bins
    N = len(xi)
    
    # Take the FFT
    fI0 = np.fft.rfft(I0)
    k = np.fft.rfftfreq(N, xi[1] - xi[0])
    
    # Apply the integral of the translation 
    beta = 1 - np.exp(alpha) 

    if method == "exact":
        # Non-rel, exact
        fJ = np.pi * hyp2f1(-k * np.pi * 1j, 0.5 - k * np.pi * 1j, 
                            2, beta ** 2)

    elif method == "bessel":
        # Non-rel, approx
        fJ = np.append([np.pi], 
                jv(1, 2 * np.pi * k[1:] * beta) / (k[1:] * beta))

    else:
        # Non-rel, series expansion, Poor convergence :(
        c = np.zeros((order + 1, len(k)), dtype="complex128")
        for n in range(order + 1):
            for j in range(len(k)):
                coeff = mpmath.binomial(2 * np.pi * k[j] * 1j, n)
                coeff = float(coeff.real) + float(coeff.imag) * 1j
                c[n, j] = (-1) ** n * coeff
        integ = np.zeros(order + 1)
        for n in range(0, order + 1, 2):
            integ[n] = np.sqrt(np.pi) * \
                gamma(0.5 * (1 + n)) / gamma(2 + 0.5 * n)        
        fJ = np.sum([c[n] * integ[n] * beta ** n 
                     for n in range(order + 1)], axis=0)

    # Low-pass filter?
    if kmax is not None:
        fJ[kmax + 1:] = 0.0

    # Take the inverse FFT and return
    return np.fft.irfft(fI0 * fJ, N)


def plot_figure(beta=0.05, amp=[1.0], mu=[1.0], sigma=[0.01], method="exact",
                order=2, kmax=None, npts=1000, lam_range=(0.9, 1.1),
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

    # Non-relativistic Doppler param
    alpha = np.log(1 - beta)

    # New figure
    fig, ax = plt.subplots(2, figsize=(6, 6), sharex=True, sharey=False)
    fig.subplots_adjust(hspace=0.1)
    ax[0].set_xlim(*lam_range)

    # Plot the shifted line
    ax[0].plot(lam, I0, 'k--', label="Rest frame")
    ax[0].plot(lam, I_num(xi, I0, alpha=alpha), 'C0', 
               label="Shifted (numerical)")
    ax[0].plot(lam, I(xi, I0, alpha=alpha), 'C1--', 
               label="Shifted (FFT)")
    ax[0].legend(fontsize=10, loc="lower left")
    ax[0].set_ylabel("local intensity")

    # Plot the disk-integrated line
    ax[1].plot(lam, S_num(xi, I0, alpha=alpha, npts=npts) / np.pi, 'C0', 
               label="Broadened (numerical)")
    ax[1].plot(lam, S(xi, I0, alpha=alpha, method=method, 
                      order=order, kmax=kmax) / np.pi, 
               'C1--', label="Broadened (FFT)")
    ax[1].legend(fontsize=10, loc="lower left")
    ax[1].set_xlabel("wavelength [arbitrary units]")
    ax[1].set_ylabel("integrated intensity")

    # Save
    fig.savefig("uniform.pdf", bbox_inches="tight")





if __name__ == "__main__":
    np.random.seed(12)
    npts = 1000
    beta = 0.01
    nlines = 5
    amp = np.ones(nlines)
    lam_range = (0.9, 1.1)


    mu = randu(lam_range[0], lam_range[1], nlines)
    sigma = 10 ** (-4 + 1 * np.random.rand(nlines))
    nlam = 1000
    method = "bessel"
    order = 2
    kmax = 50
    pad = 0.05
    noise = 0.01

    plot_figure(beta=beta, amp=amp, mu=mu, sigma=sigma, method=method,
                order=order, kmax=kmax, lam_range=lam_range, 
                npts=npts, nlam=nlam, noise=noise, pad=pad)