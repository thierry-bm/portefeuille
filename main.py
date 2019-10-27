# Backtest

import pandas as pd
import numpy as np
import scipy.optimize as opt
import cvxpy as cvx


def get_timeseries(symbol: str):
    filename = f"data/{symbol}.csv"
    t = pd.read_csv(filename, index_col="Date", parse_dates=["Date"])["Adj Close"]
    return t


def get_return(t: pd.Series, nb_days=365, weeks=None, months=None):
    if months is not None:
        weeks = months * 4
    if weeks is not None:
        nb_days = weeks * 7
    
    # Ajout d'un index sur tous les jours. On assume que les rendements
    # sont persistés, par exemple ceux du vendredi égalent ceux du samedi et
    # du dimanche
    idx = pd.date_range(start=t.index.min(), end=t.index.max(), freq="D")
    t = t.reindex(idx, method="ffill")
    vi = t
    vf = t.shift(-nb_days)
    r = vf / vi - 1

    # Annualisation
    r = (1 + r) ** (365 / nb_days) - 1
    r = 100 * r[~r.isna()]
    return r


universe = ["XBB.TO", "XDV.TO", "XIU.TO", "MSFT"]

tss = {tckr: get_timeseries(tckr) for tckr in universe}
rs = {tckr: get_return(ts) for tckr, ts in tss.items()}

matrix = pd.DataFrame()
t = pd.DataFrame(dict(zip(universe, rs)))

mu = t.mean().values
sigma = t.cov().values


def minus_markowitz(w):
    print(w)
    utility = mu @ w - 0.01 * w @ sigma @ w
    return -utility


p = len(universe)
w = cvx.Variable(p)
gamma = cvx.Parameter(nonneg=True)
s0 = cvx.Parameter(nonneg=True)

objective = mu.T * w - gamma * cvx.quad_form(w, sigma)
constraints = [cvx.sum(w) == 1, w >= 0]
prob = cvx.Problem(cvx.Maximize(objective), constraints)
reg_prob = cvx.Problem(cvx.Maximize(objective - s0 * cvx.norm(w, 1)), constraints)


# opt.minimize(minus_markowitz, x0=np.array([0, 0, 0, 0]))

