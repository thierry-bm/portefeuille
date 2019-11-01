import numpy as np
import cvxpy as cp


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


u = get_utility()
r = np.linspace(-5,5,100)
plt.plot(r, u(r))
# plt.ylim(-5, 5)

