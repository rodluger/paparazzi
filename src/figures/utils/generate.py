import numpy as np
import starry


def generate_data(
    flux_err=2e-4,
    ydeg=15,
    u=[0.5, 0.25],
    nt=16,
    inc=40,
    veq=60000,
    vsini_max=50000,
    smoothing=0.075,
    theta=None,
    wav=np.linspace(642.85, 643.15, 70),
    wav0=np.linspace(642.74, 643.26, 300),
    seed=0,
    image="spot",
    **kwargs
):
    # Set the seed
    np.random.seed(seed)

    # Instantiate the Doppler map
    udeg = len(u)
    map = starry.DopplerMap(
        lazy=False,
        ydeg=ydeg,
        udeg=udeg,
        nc=1,
        veq=veq,
        inc=inc,
        vsini_max=vsini_max,
        nt=nt,
        wav=wav,
        wav0=wav0,
    )

    # Limb darkening
    for n in range(len(u)):
        map[1 + n] = u[n]

    # Rest frame spectrum
    spectrum = (
        1.0
        - 0.85 * np.exp(-0.5 * (map.wav0 - 643.0) ** 2 / 0.0085 ** 2)
        - 0.40 * np.exp(-0.5 * (map.wav0 - 642.97) ** 2 / 0.0085 ** 2)
        - 0.20 * np.exp(-0.5 * (map.wav0 - 643.1) ** 2 / 0.0085 ** 2)
    )

    # Load the component maps
    map.load(maps=[image], spectra=spectrum, smoothing=smoothing)

    # Get rotational phases
    if theta is None:
        theta = np.linspace(-180, 180, nt, endpoint=False)

    # Generate unnormalized data. Scale the error so it's
    # the same magnitude relative to the baseline as the
    # error in the normalized dataset so we can more easily
    # compare the inference results in both cases
    flux0 = map.flux(theta=theta, normalize=False)
    flux0_err = flux_err * np.median(flux0)
    flux0 += flux0_err * np.random.randn(*flux0.shape)

    # Generate normalized data
    flux = map.flux(theta=theta, normalize=True)
    flux += flux_err * np.random.randn(*flux.shape)

    return dict(
        kwargs=dict(
            ydeg=ydeg,
            udeg=udeg,
            nc=1,
            veq=veq,
            inc=inc,
            vsini_max=vsini_max,
            nt=nt,
            wav=wav,
            wav0=wav0,
        ),
        props=dict(u=u),
        truths=dict(y=map.y, spectrum=map.spectrum),
        data=dict(
            theta=theta,
            flux0_err=flux0_err,
            flux_err=flux_err,
            flux0=flux0,
            flux=flux,
        ),
    )
