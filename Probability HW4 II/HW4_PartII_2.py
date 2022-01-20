import numpy as np

def a(n):
    Z = np.random.standard_normal(n)
    Y = 0
    for i in range(n):
        Y += np.cos(Z[i]) + np.sin(2 * Z[i])
    return Y / n

def b(n):
    X = np.random.uniform(-1, 1, n)
    Y = np.random.uniform(-1, 1, n)
    f = 0
    for i in range(n):
        if np.square(X[i] - 0.2) + 2 * np.square(Y[i] + 0.3) <= 0.25:
            f += 1
    return 4 * f / n

print("(a)")
print("n = 10^3:")
print([a(1000) for i in range(20)])
print("n = 10^5:")
print([a(100000) for j in range(20)])
print()

print("(b)")
print("n = 10^1:")
print(b(10))
print("n = 10^3:")
print(b(1000))
print("n = 10^5:")
print(b(100000))
print("n = 10^7:")
print(b(10000000))
