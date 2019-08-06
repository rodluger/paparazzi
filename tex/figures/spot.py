# -*- coding: utf-8 -*-
"""
Show the effect a rotating spot has on an absorption line.

"""
import matplotlib.pyplot as plt
import numpy as np
import starry
import paparazzi as pp


# Get the Ylm expansion of a Gaussian spot
ydeg = 20
N = (ydeg + 1) ** 2
map = starry.Map(ydeg, lazy=False)
map.add_spot(amp=-0.12, sigma=0.05, lat=30, lon=0)
ylm = np.array(map.y)

# Check that the specific intensity is positive everywhere
assert np.nanmin(map.render()) > 0

# Generate the dataset
vsini = 40.0 # km/s
nt = 15
theta = np.append([-180], np.linspace(-90, 90, nt))
dop = pp.Doppler(ydeg, vsini=vsini, inc=90)
dop.generate_data(u=ylm, R=3.e5, nlam=149, sigma=2.e-5, 
                  nlines=1, theta=theta, ferr=0.0)
lnlam = dop.lnlam
F0 = dop.F[0]
F = dop.F[1:]

# Render the images
img = map.render(theta=theta[1:], res=100)

# Set up the plot
fig, ax = plt.subplots(nt, 3, figsize=(6, 12))
fig.subplots_adjust(hspace=0.3)
cmap = plt.get_cmap("plasma")
vmin = np.nanmin(img)
vmax = np.nanmax(img)
rng = vmax - vmin
vmin -= 0.1 * rng
vmax += 0.1 * rng

# Plot each spectrum
for t in range(nt):

    # Plot spectrum
    ax[t, 1].plot(dop.lnlam, F0, "k:", lw=1, alpha=0.5)
    ax[t, 1].plot(dop.lnlam, F[t], "k-")
    ax[t, 1].axis('off')
    
    # Plot residuals
    color = [cmap(x) for x in np.linspace(0.75, 0.0, 5)]
    lw = np.linspace(2.5, 0.5, 5)
    alpha = np.linspace(0.25, 1, 5)
    for i in range(5):
        ax[t, 2].plot(lnlam, F[t] - F0, ls="-", 
                      lw=lw[i], color=color[i], alpha=alpha[i])
    ax[t, 2].axis('off')
    ax[t, 2].set_ylim(-0.02, 0.02)
    
    # Plot current stellar image
    ax[t, 0].imshow(img[t], origin="lower", 
                    extent=(-1, 1, -1, 1),
                    cmap=cmap, vmin=vmin,
                    vmax=vmax)
    ax[t, 0].set_xlim(-3, 1.05)
    ax[t, 0].set_ylim(-1.05, 1.05)
    ax[t, 0].axis('off')

ax[0, 1].set_title("spectrum", y=1.4)
ax[0, 2].set_title("residuals", y=1.4)
fig.savefig("spot.pdf", bbox_inches="tight")