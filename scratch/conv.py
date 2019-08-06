import theano
import theano.tensor as tt
import numpy as np
import matplotlib.pyplot as plt
import starry
import paparazzi as pp
from tqdm import tqdm


def conv1d(vector, kernel, flip=False):
    """
    One-dimensional linear convolution of a vector with a 1d kernel.

    """
    return tt.nnet.conv2d(
        tt.shape_padleft(vector, 3),
        tt.shape_padleft(kernel, 3),
        input_shape=(1, 1, 1, -1),
        filter_shape=(1, 1, 1, -1),
        filter_flip=flip,
    )[0, 0, 0, :]


def convdot(vectors, kernels, N):
    f = conv1d(vectors[0], kernels[0])
    for n in range(1, N):
        f += conv1d(vectors[n], kernels[n])
    return f


def flux(lam, t, A, ydeg=5, vsini=40.0, inc=60.0, P=1.0):
    N = (ydeg + 1) ** 2
    doppler = pp.Doppler(lam, ydeg=ydeg, vsini=vsini, inc=inc, P=P)
    G = doppler._g()
    map = starry.Map(ydeg, nw=len(doppler.lam_padded))
    axis = map.ops.get_axis(inc * np.pi / 180.0, 0.0)
    theta = (2 * np.pi / P * t) % (2 * np.pi)
    F = tt.as_tensor_variable(
        [
            convdot(map.ops.rotate(axis, th, A), G, N)
            for n, th in enumerate(theta)
        ]
    )
    return F


# Linear
R = 3e5
nlam = 199
nt = 3
P = 1
inc = 90.0
ydeg = 5
vsini = 40.0
sigma = 7.5e-6
dlam = np.log(1.0 + 1.0 / R)
lam = np.arange(-(nlam // 2), nlam // 2 + 1) * dlam
t = np.linspace(-0.5 * P, 0.5 * P, nt + 1)[:-1]
doppler = pp.Doppler(lam, ydeg=ydeg, vsini=vsini, inc=inc, P=P)
D = doppler.D(t=t)
lam_padded = doppler.lam_padded
vT = np.ones_like(lam_padded)
vT = 1 - 0.5 * np.exp(-0.5 * lam_padded ** 2 / sigma ** 2)
map = starry.Map(ydeg)
map.load("vogtstar.jpg")
u = np.array(map.y.eval())
A = u.reshape(-1, 1).dot(vT.reshape(1, -1))

for i in tqdm(range(100)):
    F = D.dot(A.reshape(-1)).reshape(nt, -1)

# Conv
A_t = tt.dmatrix()
F_t = theano.function(
    [A_t], flux(lam, t, A_t, ydeg=ydeg, vsini=vsini, inc=inc, P=P)
)

for i in tqdm(range(100)):
    foo = F_t(A)

fig, ax = plt.subplots(nt)
for n in range(nt):
    ax[n].plot(F[n])
    ax[n].plot(foo[n], "--")
plt.show()
