import numpy as np
import starry


def generate_data(
    nc=1,
    flux_err=1e-4,
    ydeg=15,
    u=[0.5, 0.25],
    nt=16,
    inc=40,
    veq=60000,
    smoothing=0.075,
    theta=None,
    wav=np.linspace(642.85, 643.15, 200),
    **kwargs
):

    # Instantiate the Doppler map
    map = starry.DopplerMap(
        lazy=False,
        ydeg=ydeg,
        udeg=len(u),
        nc=nc,
        veq=veq,
        inc=inc,
        nt=nt,
        wav=wav,
    )

    # Limb darkening (TODO: fix __setitem__)
    map._u = np.append([-1.0], u)

    # Component surface images
    if nc == 1:
        images = ["spot"]
    elif nc == 2:
        images = ["star", "spot"]
    else:
        raise NotImplementedError("")

    # Component spectra
    if nc == 1:
        spectra = (
            1.0
            - 0.55 * np.exp(-0.5 * (map.wav0 - 643.0) ** 2 / 0.0085 ** 2)
            - 0.02 * np.exp(-0.5 * (map.wav0 - 642.895) ** 2 / 0.0085 ** 2)
            - 0.10 * np.exp(-0.5 * (map.wav0 - 642.97) ** 2 / 0.0085 ** 2)
            - 0.04 * np.exp(-0.5 * (map.wav0 - 643.1) ** 2 / 0.0085 ** 2)
            - 0.12 * np.exp(-0.5 * (map.wav0 - 643.4) ** 2 / 0.0085 ** 2)
            - 0.08 * np.exp(-0.5 * (map.wav0 - 643.25) ** 2 / 0.0085 ** 2)
            - 0.06 * np.exp(-0.5 * (map.wav0 - 642.79) ** 2 / 0.0085 ** 2)
            - 0.03 * np.exp(-0.5 * (map.wav0 - 642.81) ** 2 / 0.0085 ** 2)
            - 0.18 * np.exp(-0.5 * (map.wav0 - 642.63) ** 2 / 0.0085 ** 2)
            - 0.04 * np.exp(-0.5 * (map.wav0 - 642.60) ** 2 / 0.0085 ** 2)
        )
    elif nc == 2:
        mu = np.array([643.025, 642.975])
        amp = np.array([0.550, 0.550])
        raise NotImplementedError("TODO")
    else:
        raise NotImplementedError("")

    # Load the component maps
    map.load(maps=images, spectra=spectra, smoothing=smoothing)

    # Generate unnormalized data
    flux0 = map.flux(theta=theta, normalize=False)
    flux0 += flux_err * np.random.randn(*flux0.shape)

    # Generate normalized data
    flux = map.flux(theta=theta, normalize=True)
    flux += flux_err * np.random.randn(*flux.shape)

    # Reset the map
    def reset_spectrum():
        map._spectrum = np.ones((map.nc, map.nw0_))

    def reset_map():
        map._y = np.zeros((map.Ny, map.nc))
        map._y[0] = 1.0

    map.reset_spectrum = reset_spectrum
    map.reset_map = reset_map

    return map, flux0, flux
