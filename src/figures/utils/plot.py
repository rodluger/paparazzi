# -*- coding: utf-8 -*-
"""
Setup for the SPOT problem.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import starry


def plot_timeseries(data, normalized=False, overlap=8.0):
    # Get the data
    theta = data["data"]["theta"]
    if normalized:
        flux = data["data"]["flux"]
    else:
        flux = data["data"]["flux0"]
    flux = flux.reshape(data["kwargs"]["nt"], -1)

    # Instantiate the map
    map = starry.DopplerMap(lazy=False, **data["kwargs"])
    map[:, :] = data["truths"]["y"]
    map.spectrum = data["truths"]["spectrum"]
    for n in range(map.udeg):
        map[1 + n] = data["props"]["u"][n]
    model = map.flux(theta, normalize=normalized).reshape(map.nt, -1)

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

    for n in range(map.nt):
        map.show(theta=theta[n], ax=ax_img[n], res=300)

        for l in ax_img[n].get_lines():
            if l.get_lw() < 1:
                l.set_lw(0.5 * l.get_lw())
                l.set_alpha(0.75)
            else:
                l.set_lw(1.25)
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


def plot_maps(y_true, y_inferred, y_uncert, y_uncert_max=0.15):
    # Set up the plot
    fig, ax = plt.subplots(3, figsize=(8, 11.5))
    fig.subplots_adjust(hspace=0.3)
    norm01 = Normalize(vmin=0, vmax=1)

    # Instantiate a map
    ydeg = int(np.sqrt(len(y_true)) - 1)
    map = starry.Map(ydeg=ydeg, lazy=False)

    # Plot the true image
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
    map[:, :] = y_inferred
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
    map[:, :] = y_uncert
    image = map.render(projection="moll")
    image /= max_inf
    map.show(
        ax=ax[2],
        image=image,
        projection="moll",
        colorbar=True,
        colorbar_size="3%",
        colorbar_pad=0.1,
        norm=Normalize(vmin=0, vmax=y_uncert_max),
    )
    cax = fig.axes[-1]
    cax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
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


def plot_spectra(wav0, s_true, s_guess, s_inferred, s_uncert):
    fig, ax = plt.subplots(1, figsize=(8, 3))

    ax.plot(wav0, s_true, "C0-", label="true")
    ax.plot(wav0, s_guess, "C1--", lw=1, label="guess")
    ax.plot(wav0, s_inferred, "C1-", label="inferred")
    ax.fill_between(
        wav0,
        s_inferred - s_uncert,
        s_inferred + s_uncert,
        color="C1",
        alpha=0.25,
    )

    ax.set_xlabel(r"$\lambda$ [nm]", fontsize=18)
    ax.set_ylabel("rest frame spectrum", fontsize=18)
    ax.legend(loc="lower left", fontsize=12)

    return fig