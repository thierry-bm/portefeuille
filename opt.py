import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import scipy.stats as stats
import cvxpy as cp

from utility import get_utility, get_cvx_utility
from main import get_return, get_timeseries

u = get_utility()

universe = ["XBB.TO", "XDV.TO", "XIU.TO"]

timeseries = {tckr: get_timeseries(tckr) for tckr in universe}
rs = {tckr: get_return(ts) for tckr, ts in timeseries.items()}

t = pd.concat(rs.values(), axis=1)
t = t.fillna(0)


def np_objective(w: np.ndarray, l=0.1):
    obj = u(np.dot(t, w)).mean() - 0.5 * la.norm(w) ** 2
    return obj



n, p = t.shape

cp_u = get_cvx_utility()
w = cp.Variable(p)
gamma = cp.Parameter(nonneg=True)
objective = 1 / n * cp.sum(cp_u(cp.matmul(t, w))) - 1 / 2 * gamma * cp.norm2(w) ** 2
constraints = [cp.sum(w) == 1, w >= 0]


def solve(gamm):
    gamma.value = gamm
    prob = cp.Problem(cp.Maximize(objective), constraints)
    reg_prob = cp.Problem(cp.Maximize(objective), constraints)
    result = reg_prob.solve()
    return result, w.value

