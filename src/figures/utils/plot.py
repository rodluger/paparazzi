# -*- coding: utf-8 -*-
"""
Setup for the SPOT problem.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import starry


def plot_timeseries(map, theta, flux, normalized=False, overlap=8.0):
    # Plot the "Joy Division" graph
    fig = plt.figure(figsize=(5, 11.5))
    ax_img = [
        plt.subplot2grid((map.nt, 8), (n, 0), rowspan=1, colspan=1)
        for n in range(map.nt)
    ]
    ax_f = [plt.subplot2grid((map.nt, 8), (0, 1), rowspan=1, colspan=7)]
    ax_f += [
        plt.subplot2grid(
            (map.nt, 8),
            (n, 1),
            rowspan=1,
            colspan=7,
            sharex=ax_f[0],
            sharey=ax_f[0],
        )
        for n in range(1, map.nt)
    ]

    flux = flux.reshape(map.nt, -1)
    model = map.flux(theta, normalize=normalized).reshape(map.nt, -1)

    for n in range(map.nt):
        map.show(theta=theta[n], ax=ax_img[n], res=300)

        for l in ax_img[n].get_lines():
            l.set_lw(0.5 * l.get_lw())
            l.set_alpha(0.75)
        ax_img[n].set_rasterization_zorder(100)

        ax_f[n].plot(
            map.wav,
            flux[n] - model[n, 0],
            "k.",
            ms=2,
            alpha=0.75,
            clip_on=False,
        )
        ax_f[n].plot(
            map.wav, model[n] - model[n, 0], "C1-", lw=1, clip_on=False
        )
        ax_f[n].axis("off")
    fac = (np.max(model) - np.min(model)) / overlap
    ax_f[0].set_ylim(-fac, fac)

    return fig


def plot_maps(map, y_true, cho_ycov):
    fig, ax = plt.subplots(3, figsize=(8, 11.5))
    fig.subplots_adjust(hspace=0.3)
    norm01 = Normalize(vmin=0, vmax=1)

    # Plot the true image
    y = map[:, :]
    map[:, :] = y_true
    image = map.render(projection="moll")
    max_true = np.nanmax(image)
    image /= max_true
    map.show(
        ax=ax[0],
        image=image,
        projection="moll",
        colorbar=True,
        colorbar_size="3%",
        colorbar_pad=0.1,
        norm=norm01,
    )
    map[:, :] = y
    ax[0].annotate(
        "true",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(10, -5),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=16,
    )

    # Plot the inferred image
    image = map.render(projection="moll")
    max_inf = np.nanmax(image)
    image /= max_inf
    map.show(
        ax=ax[1],
        image=image,
        projection="moll",
        colorbar=True,
        colorbar_size="3%",
        colorbar_pad=0.1,
        norm=norm01,
    )
    ax[1].annotate(
        "inferred",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(10, -5),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=16,
    )

    # Plot the pixel uncertainty image
    _, _, P, Q, _, _ = starry.Map(map.ydeg).get_pixel_transforms()
    L = np.tril(cho_ycov)
    W = P @ L
    y_sig = Q @ np.sqrt(np.diag(W @ W.T))
    map[:, :] = y_sig
    image = map.render(projection="moll")
    image /= max_inf
    map.show(
        ax=ax[2],
        image=image,
        projection="moll",
        colorbar=True,
        colorbar_size="3%",
        colorbar_pad=0.1,
        norm=Normalize(vmin=0),
    )
    cax = fig.axes[-1]
    cax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    map[:, :] = y
    ax[2].annotate(
        "uncert",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(10, -5),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=16,
    )

    return fig