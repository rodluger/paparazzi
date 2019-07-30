# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import paparazzi as pp
from scipy.sparse import hstack, vstack, csr_matrix, block_diag
import starry
np.random.seed(11)


def RAxisAngle(axis=[0, 1, 0], theta=0):
    """
    Rotate an arbitrary point by an axis and an angle.

    """
    cost = np.cos(theta)
    sint = np.sin(theta)

    return np.reshape([
        cost + axis[0] * axis[0] * (1 - cost),
        axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
        axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
        axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
        cost + axis[1] * axis[1] * (1 - cost),
        axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
        axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
        axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
        cost + axis[2] * axis[2] * (1 - cost)
    ], [3, 3])


def get_ortho_latitude_lines(inc=np.pi/2, obl=0, dlat=np.pi/6, npts=1000):
    """
    Return the lines of constant latitude on an orthographic projection.

    """
    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Latitude lines
    res = []
    latlines = np.arange(-np.pi/2, np.pi/2, dlat)[1:]
    for lat in latlines:

        # Figure out the equation of the ellipse
        y0 = np.sin(lat) * si
        a = np.cos(lat)
        b = a * ci
        x = np.linspace(-a, a, npts)
        y1 = y0 - b * np.sqrt(1 - (x / a) ** 2)
        y2 = y0 + b * np.sqrt(1 - (x / a) ** 2)

        # Mask lines on the backside
        if (si != 0):
            if inc > np.pi/2:
                ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                y1[y1 < ymax] = np.nan
                ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                y2[y2 < ymax] = np.nan
            else:
                ymax = y1[np.argmax(x ** 2 + y1 ** 2)]
                y1[y1 > ymax] = np.nan
                ymax = y2[np.argmax(x ** 2 + y2 ** 2)]
                y2[y2 > ymax] = np.nan

        # Rotate them
        for y in (y1, y2):
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr, yr))

    return res


def get_ortho_longitude_lines(inc=np.pi/2, obl=0, theta=0, 
                              dlon=np.pi/6, npts=1000):
    """
    Return the lines of constant longitude on an orthographic projection.
    """

    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Are we (essentially) equator-on?
    equator_on = (inc > 88 * np.pi / 180) and (inc < 92 * np.pi / 180)

    # Longitude grid lines
    res = []
    if equator_on:
        offsets = np.arange(-np.pi / 2, np.pi / 2, dlon)
    else:
        offsets = np.arange(0, 2 * np.pi, dlon)
    
    for offset in offsets:

        # Super hacky, sorry. This can probably
        # be coded up more intelligently.
        if equator_on:
            sgns = [1]
            if np.cos(theta + offset) >= 0:
                bsgn = 1
            else:
                bsgn = -1
        else:
            bsgn = 1
            if np.cos(theta + offset) >= 0:
                sgns = np.array([1, -1])
            else:
                sgns = np.array([-1, 1])

        for lon, sgn in zip([0, np.pi], sgns):

            # Viewed at i = 90
            y = np.linspace(-1, 1, npts)
            b = bsgn * np.sin(lon - theta - offset)
            x = b * np.sqrt(1 - y ** 2)
            z = sgn * np.sqrt(np.abs(1 - x ** 2 - y ** 2))

            if equator_on:

                pass
            
            else:

                # Rotate by the inclination
                R = RAxisAngle([1, 0, 0], np.pi / 2 - inc)
                v = np.vstack((x.reshape(1, -1), 
                               y.reshape(1, -1), 
                               z.reshape(1, -1)))
                x, y, _ = np.dot(R, v)

                # Mask lines on the backside
                if (si != 0):
                    if inc < np.pi/2:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[:imax + 1] = np.nan
                    else:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[imax:] = np.nan

            # Rotate by the obliquity
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr, yr))
    
    return res


def plot_main():
    """
    Plot the main figure showing the equation ``f = D a``.

    """
    # Settings for this figure
    ydeg = 2
    ntheta = 11
    inc = 40.
    vsini = 80.
    nlam = 11

    #
    # Compute stuff!
    #

    # Compute the g-functions and the corresponding Toeplitz matrices
    lnlam = np.linspace(-6e-4, 6e-4, nlam)
    doppler = pp.Doppler(ydeg=ydeg, vsini=vsini, inc=inc)
    doppler._set_lnlam(lnlam)
    g = doppler.g
    T = doppler.T

    # Pad them to reveal the structure
    for n in range(doppler.N):
        T[n] = T[n].toarray()
        T[n][:, 0] = np.nan
        T[n][:, -1] = np.nan
        T[n] = csr_matrix(T[n])
        T[n] = vstack((T[n], np.nan * np.ones((2, T[n].shape[1]))))

    # Tensordot with rotation matrices to get the full Doppler matrix
    theta = np.linspace(0, 2 * np.pi, ntheta)
    sini = np.sin(inc * np.pi / 180.)
    cosi = np.cos(inc * np.pi / 180.)
    axis = [0, sini, cosi]
    R = [doppler._R(axis, t) for t in theta]
    D = [None for t in range(ntheta)]
    for t in range(ntheta):
        TR = [None for n in range(doppler.N)]
        for l in range(ydeg + 1):
            idx = slice(l ** 2, (l + 1) ** 2)
            TR[idx] = np.tensordot(R[t][l].T, T[idx], axes=1)
        D[t] = hstack(TR)
    D = vstack(D).toarray()
    D /= np.nanmax(D)

    # The starry coefficient matrix
    vT = 1 - 0.5 * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (1e-4) ** 2)
    map = starry.Map(ydeg)
    map.inc = inc
    map.add_spot(-1, lon=-180)
    u = np.array(map.y.eval())
    A = u.reshape(-1, 1).dot(vT.reshape(1, -1))

    # Pad it, normalize it, and reshape into a vector
    a = np.hstack((A, np.nan * np.ones((A.shape[0], 2)))).reshape(-1, 1)
    a /= np.nanmax(a)

    # The full matrix
    Dpad = (a.shape[0] - D.shape[0]) // 2
    D = np.pad(D, ((Dpad, Dpad), (0, 0)), "constant", constant_values=np.nan)
    pad = np.nan * np.ones((a.shape[0], 6))
    Da = np.hstack((D, pad, a, a))

    #
    # Plot stuff!
    #

    # Plot the `D` matrix dotted into the `a` vector
    fig, ax = plt.subplots(1, figsize=(12, 8))
    cmap = plt.get_cmap("inferno")
    Da[Da == 0] = -99
    cmap.set_under((0.9, 0.9, 0.9))
    ax.imshow(Da, cmap=cmap, vmin=-1.1, vmax=1.1)
    ax.axis("off")

    # Re-compute stuff at hi res for better plotting
    lnlam = np.linspace(-6e-4, 6e-4, nlam * 11)
    doppler = pp.Doppler(ydeg=ydeg, vsini=vsini, inc=inc)
    doppler._set_lnlam(lnlam)
    g = doppler.g
    T = doppler.T
    vT = 1 - 0.5 * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (1e-4) ** 2)
    A = u.reshape(-1, 1).dot(vT.reshape(1, -1))
    D = [None for t in range(ntheta)]
    for t in range(ntheta):
        TR = [None for n in range(doppler.N)]
        for l in range(ydeg + 1):
            idx = slice(l ** 2, (l + 1) ** 2)
            TR[idx] = np.tensordot(R[t][l].T, T[idx], axes=1)
        D[t] = hstack(TR)
    D = vstack(D).toarray()

    # Compute & plot the resulting spectrum image
    F = np.dot(D, A.reshape(-1)).reshape(ntheta, -1)
    F /= np.nanmedian(F, axis=1).reshape(-1, 1)
    f = np.hstack((F[:, ::11], np.nan * np.ones((F.shape[0], 2)))).reshape(-1, 1)
    fpad = (Da.shape[0] - f.shape[0]) // 2
    f = np.pad(f, ((fpad, fpad), (0, 0)), "constant", constant_values=np.nan)
    f /= np.nanmax(f)
    f = np.hstack((f, f))
    axins = ax.inset_axes([-0.35, 0, 0.1, 1])
    axins.imshow(f, cmap=cmap)
    axins.axis('off')

    # Plot the spectra
    x0 = -0.425
    width = 0.1
    pad = 0.0136
    y0 = 0.855
    height = 0.0622
    for n in range(ntheta):
        axins = ax.inset_axes([x0, y0 - n * (height + pad), width, height])
        axins.plot(lnlam, F[n], "k-", alpha=0.3)
        axins.set_ylim(0.6, 1.45)
        axins.axis("off")

    # Plot the g functions
    x0 = 0.0075
    width = 0.092
    pad = 0.0132
    y0 = 0.95
    height = 0.05
    for n in range(doppler.N):
        axins = ax.inset_axes([x0 + n * (width + pad), y0, width, height])
        axins.axis('off')
        axins.plot(g[n], "k-", alpha=0.3)

    # Plot the spectral components
    x0 = 1.03
    width = 0.1
    pad = 0.011
    y0 = 0.90
    height = 0.1
    for n in range(doppler.N):
        axins = ax.inset_axes([x0, y0 - n * (height + pad), width, height])
        axins.axis('off')
        axins.plot(A[n], doppler.lnlam_padded, "k-", alpha=0.3)
        axins.set_xlim(-1.1, 1.1)
        axins.set_ylim(doppler.lnlam_padded[0], doppler.lnlam_padded[-1])

    # Plot the map orientation
    x0 = -0.125
    width = 0.1
    pad = 0.0136
    y0 = 0.855
    height = 0.0622
    for n in range(ntheta):
        axins = ax.inset_axes([x0, y0 - n * (height + pad), width, height])
        x = np.linspace(-1, 1, 10000)
        y = np.sqrt(1 - x ** 2)
        axins.plot(x, y, 'k-', alpha=1, lw=1, zorder=101)
        axins.plot(x, -y, 'k-', alpha=1, lw=1, zorder=101)
        lat_lines = get_ortho_latitude_lines(inc=inc * np.pi / 180)
        for x, y in lat_lines:
            axins.plot(x, y, 'k-', lw=0.5, alpha=0.25, zorder=100)
        lon_lines = get_ortho_longitude_lines(inc=inc * np.pi / 180, theta=np.pi + theta[n])
        for n, l in enumerate(lon_lines):
            if n == 0:
                axins.plot(l[0], l[1], 'r-', lw=1.25, alpha=1, zorder=100)
            else:
                axins.plot(l[0], l[1], 'k-', lw=0.5, alpha=0.25, zorder=100)
        axins.set_aspect(1)
        axins.axis('off')

    # Label stuff
    ax.annotate(r"$K + W$", xy=(8, 158), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(0, 0), textcoords="offset points")

    ax.annotate(r"$K$", xy=(0, 149), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(-7, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$N$", xy=(75, 0), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(0, 20), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$M$", xy=(0, 84), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(-65, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$K + W$", xy=(160, 8), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(12, 0), textcoords="offset points",
                clip_on=False, rotation=90)

    ax.annotate(r"$N$", xy=(160, 84), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(75, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$=$", xy=(0, 84), xycoords="data", 
                ha="center", va="center", fontsize=16,
                xytext=(-95, 0), textcoords="offset points")

    ax.annotate(r"$M$", xy=(0, 84), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(-200, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$K$", xy=(0, 19), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(-115, 0), textcoords="offset points",
                clip_on=False)

    fig.savefig("linalg.pdf", bbox_inches="tight")


def plot_u():
    """
    Plot the linear equation for `u`.
    
    """
    # Settings for this figure
    ydeg = 2
    ntheta = 11
    inc = 40.
    vsini = 80.
    nlam = 11

    #
    # Compute stuff!
    #

    # Compute the g-functions and the corresponding Toeplitz matrices
    lnlam = np.linspace(-6e-4, 6e-4, nlam)
    doppler = pp.Doppler(ydeg=ydeg, vsini=vsini, inc=inc)
    doppler._set_lnlam(lnlam)
    g = doppler.g
    T = doppler.T

    # Pad them to reveal the structure
    for n in range(doppler.N):
        T[n] = T[n].toarray()
        T[n][:, 0] = np.nan
        T[n][:, -1] = np.nan
        T[n] = csr_matrix(T[n])
        T[n] = vstack((T[n], np.nan * np.ones((2, T[n].shape[1]))))

    # Tensordot with rotation matrices to get the full Doppler matrix
    theta = np.linspace(0, 2 * np.pi, ntheta)
    sini = np.sin(inc * np.pi / 180.)
    cosi = np.cos(inc * np.pi / 180.)
    axis = [0, sini, cosi]
    R = [doppler._R(axis, t) for t in theta]
    D = [None for t in range(ntheta)]
    for t in range(ntheta):
        TR = [None for n in range(doppler.N)]
        for l in range(ydeg + 1):
            idx = slice(l ** 2, (l + 1) ** 2)
            TR[idx] = np.tensordot(R[t][l].T, T[idx], axes=1)
        D[t] = hstack(TR)
    D = vstack(D).toarray()
    D /= np.nanmax(D)

    # The spectrum matrix
    vT0 = 1 - 0.5 * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (1e-4) ** 2)
    vT1 = 1 - 0.25 * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (0.5e-4) ** 2) \
            - 0.25 * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (2.5e-4) ** 2)
    vT2 = 1 - 0.5 * (1 + 0.5 * np.sin(2e4 * doppler.lnlam_padded)) * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (2e-4) ** 2)
    vT0 = np.append(vT0, [np.nan, np.nan])
    vT1 = np.append(vT1, [np.nan, np.nan])
    vT2 = np.append(vT2, [np.nan, np.nan])
    vT = np.hstack((vT0.reshape(-1, 1), vT1.reshape(-1, 1), vT2.reshape(-1, 1)))
    V = block_diag([vT.reshape(-1, 3) for n in range(doppler.N)]).toarray()
    V[np.where(np.isnan(V))[0], :] = np.nan
    V /= np.nanmax(V)

    # The Ylm coefficient matrix
    map = starry.Map(ydeg)
    map.inc = inc
    map.add_spot(-1, lon=-180)
    u0 = np.array(map.y.eval())
    u1 = u0 * 0.5 * np.random.randn(doppler.N)
    u2 = u0 * 0.1 * np.random.randn(doppler.N)

    # Pad it
    u = np.hstack((u0.reshape(-1, 1), u1.reshape(-1, 1), u2.reshape(-1, 1)))
    u = np.pad(u, ((0, 0), (0, 2)), "constant", constant_values=np.nan).reshape(-1, 1)
    upad = (V.shape[0] - u.shape[0]) // 2
    u = np.pad(u, ((upad, upad), (0, 0)), "constant", constant_values=np.nan)
    pad = np.nan * np.ones((V.shape[0], 6))
    Vu = np.hstack((V, pad, u))

    # Everything now
    Dpad = (V.shape[0] - D.shape[0]) // 2
    D = np.pad(D, ((Dpad, Dpad), (0, 0)), "constant", constant_values=np.nan)
    pad = np.nan * np.ones((V.shape[0], 10))
    DVu = np.hstack((D, pad, Vu))

    #
    # Plot stuff!
    #

    fig, ax = plt.subplots(1, figsize=(12, 8))
    cmap = plt.get_cmap("inferno")
    DVu[DVu == 0] = -99
    cmap.set_under((0.9, 0.9, 0.9))
    ax.imshow(DVu, cmap=cmap, vmin=-1.1, vmax=1.1)
    ax.axis("off")

    # Re-compute stuff at hi res for better plotting
    lnlam = np.linspace(-6e-4, 6e-4, nlam * 11)
    doppler = pp.Doppler(ydeg=ydeg, vsini=vsini, inc=inc)
    doppler._set_lnlam(lnlam)
    g = doppler.g
    T = doppler.T
    vT0 = 1 - 0.5 * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (1e-4) ** 2)
    vT1 = 1 - 0.25 * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (0.5e-4) ** 2) \
            - 0.25 * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (2.5e-4) ** 2)
    vT2 = 1 - 0.5 * (1 + 0.5 * np.sin(2e4 * doppler.lnlam_padded)) * np.exp(-0.5 * doppler.lnlam_padded ** 2 / (2e-4) ** 2)
    vT = np.vstack((vT0.reshape(1, -1), vT1.reshape(1, -1), vT2.reshape(1, -1)))
    u = np.hstack((u0.reshape(-1, 1), u1.reshape(-1, 1), u2.reshape(-1, 1)))
    A = u.dot(vT)
    D = [None for t in range(ntheta)]
    for t in range(ntheta):
        TR = [None for n in range(doppler.N)]
        for l in range(ydeg + 1):
            idx = slice(l ** 2, (l + 1) ** 2)
            TR[idx] = np.tensordot(R[t][l].T, T[idx], axes=1)
        D[t] = hstack(TR)
    D = vstack(D).toarray()

    # Compute & plot the resulting spectrum image
    F = np.dot(D, A.reshape(-1)).reshape(ntheta, -1)
    F /= np.nanmedian(F, axis=1).reshape(-1, 1)
    f = np.hstack((F[:, ::11], np.nan * np.ones((F.shape[0], 2)))).reshape(-1, 1)
    fpad = (DVu.shape[0] - f.shape[0]) // 2
    f = np.pad(f, ((fpad, fpad), (0, 0)), "constant", constant_values=np.nan)
    f /= np.nanmax(f)
    f = np.hstack((f, f))
    axins = ax.inset_axes([-0.2975, 0, 0.1, 1])
    axins.imshow(f, cmap=cmap)
    axins.axis('off')

    # Plot the spectra
    x0 = -0.3735
    width = 0.1
    pad = 0.0135
    y0 = 0.855
    height = 0.0622
    for n in range(ntheta):
        axins = ax.inset_axes([x0, y0 - n * (height + pad), width, height])
        axins.plot(lnlam, F[n], "k-", alpha=0.3)
        axins.set_ylim(0.6, 1.45)
        axins.axis("off")

    # Plot the g functions
    x0 = 0.0075
    width = 0.073
    pad = 0.0132
    y0 = 0.95
    height = 0.05
    for n in range(doppler.N):
        axins = ax.inset_axes([x0 + n * (width + pad), y0, width, height])
        axins.axis('off')
        axins.plot(g[n], "k-", alpha=0.3)

    # Plot the spectral components
    x0 = 0.8325
    width = 0.1275
    y0 = 1.02
    height = 0.06
    axins = ax.inset_axes([x0, y0, width, height])
    axins.plot(vT0, doppler.lnlam_padded, "k-", alpha=0.3)
    axins.plot(vT1 + 1, doppler.lnlam_padded, "k-", alpha=0.3)
    axins.plot(vT2 + 2, doppler.lnlam_padded, "k-", alpha=0.3)
    axins.axis("off")

    # Plot the map orientation
    x0 = -0.125
    width = 0.1
    pad = 0.0136
    y0 = 0.855
    height = 0.0622
    for n in range(ntheta):
        axins = ax.inset_axes([x0, y0 - n * (height + pad), width, height])
        x = np.linspace(-1, 1, 10000)
        y = np.sqrt(1 - x ** 2)
        axins.plot(x, y, 'k-', alpha=1, lw=1, zorder=101)
        axins.plot(x, -y, 'k-', alpha=1, lw=1, zorder=101)
        lat_lines = get_ortho_latitude_lines(inc=inc * np.pi / 180)
        for x, y in lat_lines:
            axins.plot(x, y, 'k-', lw=0.5, alpha=0.25, zorder=100)
        lon_lines = get_ortho_longitude_lines(inc=inc * np.pi / 180, theta=np.pi + theta[n])
        for n, l in enumerate(lon_lines):
            if n == 0:
                axins.plot(l[0], l[1], 'r-', lw=1.25, alpha=1, zorder=100)
            else:
                axins.plot(l[0], l[1], 'k-', lw=0.5, alpha=0.25, zorder=100)
        axins.set_aspect(1)
        axins.axis('off')

    # Label stuff
    ax.annotate(r"$K + W$", xy=(8, 158), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(0, 0), textcoords="offset points")

    ax.annotate(r"$K$", xy=(0, 149), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(-7, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$N$", xy=(75, 0), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(0, 20), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$M$", xy=(0, 84), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(-65, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$K + W$", xy=(166, 8), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(71, 0), textcoords="offset points",
                clip_on=False, rotation=90)

    ax.annotate(r"$N$", xy=(160, 84), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(110, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$P$", xy=(160, 64.25), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(103, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$N$", xy=(163, 160), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(35, -45), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$N$", xy=(163, 84), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(-15, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$P$", xy=(163, 160), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(66, -30), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$=$", xy=(0, 84), xycoords="data", 
                ha="center", va="center", fontsize=16,
                xytext=(-95, 0), textcoords="offset points")

    ax.annotate(r"$M$", xy=(0, 84), xycoords="data", 
                ha="center", va="center", fontsize=12,
                xytext=(-220, 0), textcoords="offset points",
                clip_on=False)

    ax.annotate(r"$K$", xy=(0, 19), xycoords="data", 
                ha="center", va="center", fontsize=8,
                xytext=(-115, 0), textcoords="offset points",
                clip_on=False)


    fig.savefig("linalg_u.pdf", bbox_inches="tight")
    


if __name__ == "__main__":
    plot_main()
    plot_u()