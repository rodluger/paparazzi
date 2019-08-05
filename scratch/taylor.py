import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

# Dimensions
M = 100
N = 2

# Generate a dataset
A = np.random.randn(M, N)
B = np.random.randn(M, N)
w_true = np.random.randn(N)
f = np.dot(A, w_true) / np.dot(B, w_true)

# Taylor expansion
w0 = w_true + 0.01 * np.random.randn(N)

Aw0 = np.dot(A, w0)
Bw0 = np.dot(B, w0)


f0 = np.dot((Aw0 / Bw0 ** 2).reshape(-1, 1) * B, w0)
y = f - f0
X = (1 / Bw0).reshape(-1, 1) * A - (Aw0 / Bw0 ** 2).reshape(-1, 1) * B

fig = plt.figure()
plt.plot(f)
plt.plot(f0)
print(np.sum((f - f0) ** 2))

plt.plot(f0 + np.dot(X, w_true))
print(np.sum((f - (f0 + np.dot(X, w_true))) ** 2))

what = np.linalg.solve(np.dot(X.T, X) + 1e-10 * np.eye(N), np.dot(X.T, y))
plt.plot(f0 + np.dot(X, what))
print(np.sum((f - (f0 + np.dot(X, what))) ** 2))

fig = plt.figure()
plt.plot(w_true)
plt.plot(what)
plt.show()