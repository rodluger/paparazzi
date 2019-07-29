# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import paparazzi as pp
from scipy.sparse import hstack, vstack, csr_matrix
import starry


def RAxisAngle(axis=[0, 1, 0], theta=0):
    """

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


# DEBUG
plt.switch_backend("Qt5Agg")


ydeg = 2
ntheta = 11
inc = 40.

# Compute the Toeplitz matrices
doppler = pp.Doppler(np.linspace(-6e-4, 6e-4, 11), 
    ydeg=ydeg, vsini=80., inc=inc, P=1.0)
g = doppler._g()
T = doppler._T()

# Pad them to reveal the structure
for n in range(doppler.N):
    T[n] = T[n].toarray()
    T[n][:, 0] = np.nan
    T[n][:, -1] = np.nan
    T[n] = csr_matrix(T[n])
    T[n] = vstack((T[n], np.nan * np.ones((2, T[n].shape[1]))))

# Tensordot with rotation matrices to get the full Doppler matrix
theta = np.linspace(0, 2 * np.pi, ntheta)
R = [doppler._R(doppler._axis, t) for t in theta]
D = [None for t in range(ntheta)]
for t in range(ntheta):
    TR = [None for n in range(doppler.N)]
    for l in range(ydeg + 1):
        idx = slice(l ** 2, (l + 1) ** 2)
        TR[idx] = np.tensordot(R[t][l].T, T[idx], axes=1)
    D[t] = hstack(TR)
D = vstack(D).toarray()
D /= np.nanmax(D)

# The coefficient matrix
vT = 1 - 0.5 * np.exp(-0.5 * doppler.lam_padded ** 2 / (1e-4) ** 2)
map = starry.Map(ydeg)
map.inc = inc
map.add_spot(-1)
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

# Plot the matrices
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(Da)
ax.axis("off")

# Re-compute stuff at hi res
doppler = pp.Doppler(np.linspace(-6e-4, 6e-4, 111), 
    ydeg=ydeg, vsini=80., inc=inc, P=1.0)
g = doppler._g()
vT = 1 - 0.5 * np.exp(-0.5 * doppler.lam_padded ** 2 / (2e-4) ** 2)
A = u.reshape(-1, 1).dot(vT.reshape(1, -1))

# Plot the g functions
x0 = 0.0075
width = 0.092
pad = 0.0132
y0 = 0.95
height = 0.05
for n in range(doppler.N):
    axins = ax.inset_axes([x0 + n * (width + pad), y0, width, height])
    axins.axis('off')
    axins.plot(g[n])

# Plot the spectral components
x0 = 1.03
width = 0.1
pad = 0.011
y0 = 0.90
height = 0.1
for n in range(doppler.N):
    axins = ax.inset_axes([x0, y0 - n * (height + pad), width, height])
    axins.axis('off')
    axins.plot(A[n], doppler.lam_padded)
    axins.set_xlim(-1.1, 1.1)
    axins.set_ylim(doppler.lam_padded[0], doppler.lam_padded[-1])

# Plot the map orientation
x0 = -0.1
width = 0.1
pad = 0.0136
y0 = 0.855
height = 0.0622
for n in range(ntheta):
    axins = ax.inset_axes([x0, y0 - n * (height + pad), width, height])
    x = np.linspace(-1, 1, 10000)
    y = np.sqrt(1 - x ** 2)
    axins.plot(x, y, 'k-', alpha=1, lw=1)
    axins.plot(x, -y, 'k-', alpha=1, lw=1)
    lat_lines = get_ortho_latitude_lines(inc=inc * np.pi / 180)
    for x, y in lat_lines:
        axins.plot(x, y, 'k-', lw=0.5, alpha=0.25, zorder=100)
    lon_lines = get_ortho_longitude_lines(inc=inc * np.pi / 180, theta=theta[n])
    for n, l in enumerate(lon_lines):
        if n == 0:
            axins.plot(l[0], l[1], 'r-', lw=1.5, alpha=1, zorder=100)
        else:
            axins.plot(l[0], l[1], 'k-', lw=0.5, alpha=0.25, zorder=100)
    axins.set_aspect(1)
    axins.axis('off')
plt.show()
