# -*- coding: utf-8 -*-
"""
Plots an image of the Doppler design matrix for a low
degree inference problem.

"""
import matplotlib.pyplot as plt
import numpy as np
import paparazzi as pp
from scipy.sparse import hstack, vstack, csr_matrix, block_diag, diags
import starry

np.random.seed(11)


def RAxisAngle(axis=[0, 1, 0], theta=0):
    """
    Rotate an arbitrary point by an axis and an angle.

    """
    cost = np.cos(theta)
    sint = np.sin(theta)

    return np.reshape(
        [
            cost + axis[0] * axis[0] * (1 - cost),
            axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
            axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
            axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
            cost + axis[1] * axis[1] * (1 - cost),
            axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
            axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
            axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
            cost + axis[2] * axis[2] * (1 - cost),
        ],
        [3, 3],
    )


def get_ortho_latitude_lines(inc=np.pi / 2, obl=0, dlat=np.pi / 6, npts=1000):
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
    latlines = np.arange(-np.pi / 2, np.pi / 2, dlat)[1:]
    for lat in latlines:

        # Figure out the equation of the ellipse
        y0 = np.sin(lat) * si
        a = np.cos(lat)
        b = a * ci
        x = np.linspace(-a, a, npts)
        y1 = y0 - b * np.sqrt(1 - (x / a) ** 2)
        y2 = y0 + b * np.sqrt(1 - (x / a) ** 2)

        # Mask lines on the backside
        if si != 0:
            if inc > np.pi / 2:
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


def get_ortho_longitude_lines(
    inc=np.pi / 2, obl=0, theta=0, dlon=np.pi / 6, npts=1000
):
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
                v = np.vstack(
                    (x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1))
                )
                x, y, _ = np.dot(R, v)

                # Mask lines on the backside
                if si != 0:
                    if inc < np.pi / 2:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[: imax + 1] = np.nan
                    else:
                        imax = np.argmax(x ** 2 + y ** 2)
                        y[imax:] = np.nan

            # Rotate by the obliquity
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr, yr))

    return res


# Settings for this figure
ydeg = 2
ntheta = 11
inc = 40.0
vsini = 80.0
nlam = 121

#
# Compute stuff!
#

# Compute the g-functions and the corresponding Toeplitz matrices
lnlam = np.linspace(-6e-4, 6e-4, nlam)
doppler = pp.Doppler(ydeg=ydeg, vsini=vsini, inc=inc)
doppler._set_lnlam(lnlam)
kT = doppler.kT()
T = [None for n in range(doppler.N)]
for n in range(doppler.N):
    diagonals = np.tile(kT[n].reshape(-1, 1), doppler.K)
    if np.any(diagonals):
        diagonals[diagonals == 0] = 1e-5
    offsets = np.arange(doppler.W)
    T[n] = diags(
        diagonals,
        offsets,
        (doppler.K, doppler.K + doppler.W - 1),
        format="csr",
    )

# Tensordot with rotation matrices to get the full Doppler matrix
theta = np.linspace(0, 2 * np.pi, ntheta)
theta[-1] = 0
sini = np.sin(inc * np.pi / 180.0)
cosi = np.cos(inc * np.pi / 180.0)
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

# Plot the `D` matrix dotted into the `a` vector
fig, ax = plt.subplots(1, figsize=(11, 9.25))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
cmap = plt.get_cmap("inferno")
D[D == 0] = -99
cmap.set_under((0.9, 0.9, 0.9))
ax.imshow(D, cmap=cmap, vmin=-1.1, vmax=1.1)
ax.axis("off")

# Plot the Ylms
x0 = 0.0
width = 1.0 / 9.0
pad = 0.0
y0 = 1.0
height = 1.0 / 9.0
map = starry.Map(ydeg=2, lazy=False)
n = 0
for l in range(ydeg + 1):
    for m in range(-l, l + 1):
        axins = ax.inset_axes([x0 + n * (width + pad), y0, width, height])
        map.reset()
        if n > 0:
            map[l, m] = 1.0
        img = map.render(res=500)[0]
        axins.imshow(img, origin="lower", cmap=cmap, extent=(-1, 1, -1, 1))
        axins.set_xlim(-1.2, 1.2)
        axins.set_ylim(-1.2, 1.2)
        n += 1
        axins.axis("off")

# Plot the map orientation
x0 = -1.0 / 11.0
width = 1.0 / 11.0
pad = 0.0
y0 = 1.0 - 1.0 / 11.0
height = 1.0 / 11.0
for n in range(ntheta):
    axins = ax.inset_axes([x0, y0 - n * (height + pad), width, height])
    x = np.linspace(-1, 1, 10000)
    y = np.sqrt(1 - x ** 2)
    axins.plot(x, y, "k-", lw=1, zorder=102)
    axins.plot(x, -y, "k-", lw=1, zorder=102)
    r = 1.035
    x = np.linspace(-r, r, 10000)
    y = np.sqrt(r ** 2 - x ** 2)
    axins.plot(x, y, "w-", lw=1, zorder=103)
    axins.plot(x, -y, "w-", lw=1, zorder=103)
    lat_lines = get_ortho_latitude_lines(inc=inc * np.pi / 180)
    for x, y in lat_lines:
        axins.plot(x, y, "#aaaaaa", lw=0.5, zorder=100)
    lon_lines = get_ortho_longitude_lines(
        inc=inc * np.pi / 180, theta=np.pi + theta[n]
    )
    for n, l in enumerate(lon_lines):
        if n == 0:
            axins.plot(l[0], l[1], "r-", lw=1.5, zorder=101)
        else:
            axins.plot(l[0], l[1], "#aaaaaa", lw=0.5, zorder=100)
    axins.plot(0, np.sin(inc * np.pi / 180), "ro", zorder=104, ms=4)
    axins.set_aspect(1)
    axins.axis("off")

fig.savefig("linalg.pdf", bbox_inches="tight", dpi=300)
