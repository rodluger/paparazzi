# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
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


def S(xi, I0, alpha=0, npts=1000):
    """
    Return the disk-integrated spectrum, computed via an FFT.
    
    """
    # Number of wavelength bins
    N = len(xi)
    
    # Take the FFT
    fI0 = np.fft.rfft(I0)
    k = np.fft.rfftfreq(N, xi[1] - xi[0])
    
    # Apply the integral of the translation   
    beta = 1 - np.exp(alpha) 
    fI0 *= np.pi * hyp2f1(-k * np.pi * 1j, 0.5 - k * np.pi * 1j, 2, beta ** 2)
    
    # Take the inverse FFT and return
    return np.fft.irfft(fI0, N)


def plot_figure():
    """
    Plot the figure for the paper.

    """
    # An evenly sampled timeseries in xi = log(wavelength)
    xi = np.linspace(-0.1, 0.1, 1000)
    lam = np.exp(xi)

    # A gaussian absorption line
    amp = 1.0
    mu = 1.0
    sigma = 0.01
    I0 = 1 - amp * np.exp(-0.5 * (lam - mu) ** 2 / sigma ** 2)

    # A moderate doppler shift
    beta = 0.05
    alpha = np.log(1 - beta)

    # New figure
    fig, ax = plt.subplots(2, figsize=(6, 6), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.1)
    ax[0].set_xlim(0.9, 1.1)

    # Plot the shifted line
    ax[0].plot(lam, I0, 'k--', label="Rest frame")
    ax[0].plot(lam, I_num(xi, I0, alpha=alpha), 'C0', 
               label="Shifted (numerical)")
    ax[0].plot(lam, I(xi, I0, alpha=alpha), 'C1--', 
               label="Shifted (FFT)")
    ax[0].legend(fontsize=10, loc="lower left")
    ax[0].set_ylabel("local intensity")

    # Plot the disk-integrated line
    ax[1].plot(lam, S_num(xi, I0, alpha=0) / np.pi, 'k--', label="Rest frame");
    ax[1].plot(lam, S_num(xi, I0, alpha=alpha) / np.pi, 'C0', 
               label="Broadened (numerical)")
    ax[1].plot(lam, S(xi, I0, alpha=alpha) / np.pi, 'C1--', 
               label="Broadened (FFT)")
    ax[1].legend(fontsize=10, loc="lower left")
    ax[1].set_xlabel("wavelength [arbitrary units]")
    ax[1].set_ylabel("integrated intensity")

    # Save
    fig.savefig("uniform.pdf", bbox_inches="tight")


if __name__ == "__main__":
    plot_figure()