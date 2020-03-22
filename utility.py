import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from functools import partial

def get_utility(beta0=2):
    """Donné une liste de tuples (r_0, \partial r) retourne une fonction 
    d'utilité linéaire par pièce selon les critères demandés.
    """
    r1 = -3.5
    beta1 = 1
    beta2 = 0.6
    beta3 = 0.3
    r3 = 3
    intercept1 = (beta1 - beta0) * r1
    intercept = (beta2 - beta3) * r3

    def u(r):
        lins = [beta0 * r + intercept1, r, beta2 * r, beta3 * r + intercept]
        return np.array(lins).min(axis=0)
        # return np.minimum(r, beta1 * r, beta2 * r + intercept)

    return u


def get_cvx_utility(beta0=2):
    r1 = -2.5
    beta1 = 1
    beta2 = 0.6
    beta3 = 0.3
    r3 = 3
    intercept1 = (beta1 - beta0) * r1
    intercept = (beta2 - beta3) * r3

    def u(r):
        u = cp.minimum(beta0 * r + intercept1, r, beta2 * r, beta3 * r + intercept)
        return u

    return u

# %%
def get_cvx_utility2():
    r0 = 0
    m0 = 1
    r1 = 2
    m1 = 0.5
    r2 = 3
    m2 = 0.2

    b = {0:0}
    r = {}
    m = {}
    rs_ms = [(r0,m0)] + [(r1,m1), (r2,m2)]
    for i, (_r, _m) in enumerate(rs_ms):
        r[i] = _r
        m[i] = _m

    def y(n, _r):
        return m[n]*_r + b[n]

    for n in range(1, len(rs_ms)):
        b[n] = y(n-1, r[n]) - m[n]*r[n]

    def _y(_r,n):
        return y(n=n,_r=_r)

    ys = [partial(_y,n=n) for n in range(len(rs_ms))]

    def u(_r):
        u = cp.minimum(*(y(_r) for y in ys))
        return u
    return u


u = get_cvx_utility2()


# u = get_utility()
# r = np.linspace(-5,5,100)
# plt.plot(r, u(r))
# # plt.ylim(-5, 5)



# %%
