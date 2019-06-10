"""
Plot the convergence propoerties of the Taylor expansion
for a rigidly rotating uniform star.
"""
import numpy as np
import sympy
from sympy import *
from sympy.stats import *
import matplotlib.pyplot as plt


# Initialize SymPy
print("Using sympy version", sympy.__version__)
#init_session(quiet=True)
s_I0, s_lam, s_mu, s_sigma, s_amp, s_beta = \
    symbols(r'I_0 \lambda \mu \sigma A \beta')


def P(n, k):
    """
    Return the incomplete Bell polynomial needed to compute
    D/Dbeta.
    
    """
    if k > n:
        return 0
    series = [(-1) ** j * factorial(j) for j in range(1, n - k + 2)]
    return bell(n, k, series)


def _S(N=10, integrated=True):
    """
    Return the Nth order approximation to the disk-integrated 
    spectrum as a SymPy expression.
    
    """
    # The rest frame line profile
    s_I0 = 1 - s_amp * exp(-Rational(1, 2) * (s_lam - s_mu) ** 2 / s_sigma ** 2)

    # The higher order terms
    SN = 0
    
    if integrated:
        n_range = range(2, N + 1, 2)
    else:
        n_range = range(1, N + 1)
    
    for n in n_range:
        sum_k = 0
        for k in range(1, n + 1):
            sum_k += s_lam ** k * P(n, k) * diff(s_I0, s_lam, k)
        if integrated:
            beta_term = (sqrt(pi) / pi) * s_beta ** n * \
                gamma(Rational(n, 2) + Rational(1, 2)) / gamma(Rational(n, 2) + 2)
        else:
            beta_term = s_beta ** n
        SN += beta_term * sum_k / factorial(n)
    
    # Return
    return s_I0 + SN


def S(lam, beta=0, mu=0.5, sigma=0.001, amp=1.0, N=10, integrated=True):
    """
    Return the disk-integrated spectrum as numpy array,
    computed using the Taylor expansion.
    
    """
    if beta == 0:
        N = 0
    args = (s_lam, s_mu, s_sigma, s_amp, s_beta)
    return lambdify(args, _S(N=N, integrated=integrated), "numpy")(
                lam, mu, sigma, amp, beta
            )

def S_num(lam, beta=0, mu=0.5, sigma=0.001, amp=1.0, npts=1000, integrated=True):
    """
    Return the disk-integrated spectrum as numpy array,
    computed numerically.
    
    """
    # The rest frame line profile
    I0 = 1 - amp * np.exp(-0.5 * (lam - mu) ** 2 / sigma ** 2)
    
    if integrated:    
        S = np.zeros_like(lam)
        A = 0
        for x in np.linspace(-1, 1, npts + 2)[1:-1]:
            lam0 = lam / (1 + beta * x)
            I = np.interp(lam0, lam, I0)
            S += I * np.sqrt(1 - x ** 2)
            A += np.sqrt(1 - x ** 2)
        return S / A
    else:
        lam0 = lam / (1 + beta)
        return np.interp(lam0, lam, I0)


def plot_convergence():
    mu = 0.5
    sigma = 0.001
    npts = 1000
    lam = np.linspace(mu - 5 * sigma, mu + 5 * sigma, npts)
    lam_hr = np.linspace(mu - 5 * sigma, mu + 5 * sigma, npts * 100)
    N_arr = list(np.arange(0, 9, 2))
    beta_arr = np.logspace(-5, -2, 20)
    rms = np.zeros((len(N_arr), len(beta_arr)))
    for i, beta in enumerate(beta_arr):
        true_S_hr = S_num(lam_hr, beta=beta, sigma=sigma)
        true_S = np.interp(lam, lam_hr, true_S_hr)
        for j, N in enumerate(N_arr):
            rms[j, i] = np.sqrt(np.mean((S(lam, beta=beta, N=N, sigma=sigma) 
                                         - true_S) ** 2))

    fig = plt.figure(figsize=(8, 4))
    for j, N in enumerate(N_arr):
        plt.plot(beta_arr / sigma * mu, rms[j], '.-', 
                 label=r"$N = {0}$".format(N))
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(fontsize=9)
    plt.ylabel("rms error")
    plt.xlabel(r"$\xi\left( \mu, \sigma, \beta \right)$")
    plt.gca().set_xlim(*plt.gca().get_xlim())
    plt.axvspan(3, plt.gca().get_xlim()[1], color="k", alpha=0.2)
    plt.annotate("does not \nconverge", xy=(0.9, 0.5), 
                 xycoords="axes fraction", fontsize=8)
    fig.savefig("convergence.pdf", bbox_inches='tight')


def plot_local_spectrum():
    mu = 0.5
    sigma = 0.001
    npts = 1000
    lam = np.linspace(mu - 5 * sigma, mu + 5 * sigma, npts)
    S0 = S(lam, beta=0, mu=mu, sigma=sigma, integrated=False)

    beta = np.linspace(1e-3, 5e-3, 11)[:-1]
    fig, ax = plt.subplots(5, 2, figsize=(5, 8), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.125)
    for i, axis in enumerate(ax.T.flatten()):
        axis.plot(lam, S0, color="k", alpha=0.5, ls="--", lw=1)
        axis.plot(lam, S_num(lam, beta=beta[i], mu=mu, 
                             sigma=sigma, integrated=False))
        axis.plot(lam, S(lam, beta=beta[i], mu=mu, sigma=sigma, 
                         N=10, integrated=False))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.annotate(r"$\xi = {:.1f}$".format(beta[i] / (sigma / mu)), 
                      xy=(0, 0), xycoords="axes fraction",
                      xytext=(5, 5), textcoords="offset points", fontsize=10)
        axis.margins(0, None)
    fig.savefig("local_spectrum.pdf", bbox_inches='tight')


def plot_disk_integrated_spectrum():
    mu = 0.5
    sigma = 0.001
    npts = 1000
    lam = np.linspace(mu - 5 * sigma, mu + 5 * sigma, npts)
    S0 = S(lam, beta=0, mu=mu, sigma=sigma, integrated=True)

    beta = np.linspace(1e-3, 5e-3, 11)[:-1]
    fig, ax = plt.subplots(5, 2, figsize=(5, 8), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.125)
    for i, axis in enumerate(ax.T.flatten()):
        axis.plot(lam, S0, color="k", alpha=0.5, ls="--", lw=1)
        axis.plot(lam, S_num(lam, beta=beta[i], mu=mu, 
                             sigma=sigma, integrated=True))
        axis.plot(lam, S(lam, beta=beta[i], mu=mu, sigma=sigma, 
                         N=10, integrated=True))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.annotate(r"$\xi = {:.1f}$".format(beta[i] / (sigma / mu)), 
                      xy=(0, 0), xycoords="axes fraction",
                      xytext=(5, 5), textcoords="offset points", fontsize=10)
        axis.margins(0, None)
    fig.savefig("disk_integrated_spectrum.pdf", bbox_inches='tight')


if __name__ == "__main__":
    plot_convergence()
    plot_local_spectrum()
    plot_disk_integrated_spectrum()