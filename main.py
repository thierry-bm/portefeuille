# Backtest

import pandas as pd
import numpy as np
import scipy.optimize as opt
import yfinance as yf
import cvxpy as cp

from static import get_data
# from opt import get_cvx_utility
from utility import get_utility


def get_timeseries(symbol: str) -> pd.Series:
    filename = f"data/{symbol}.csv"
    t = pd.read_csv(filename, index_col="Date", parse_dates=["Date"])[
        "Adj Close"
    ]
    t.name = symbol
    return t


def get_target_return(
    t: pd.Series, nb_days=365, weeks=None, months=None, trim_data=False
) -> pd.Series:
    """Retourne le rendement obtenu si par exemple on achète le titre à une certaine
    date et qu'on le revend nb_days plus tard.
    """

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
    r = 100 * r

    # On peut maintenant supprimer l'index contenant les fds pour ne conserver que 
    # jours définis par t. 
    r_idx = r.index.intersection(t.index)
    r = r.loc[r_idx]

    # On conserve uniquement les rendements qui contiennent de l'information Par ex. si
    # t donne de l'information jusqu'au 2019-10-31 et que nb_days = 365, alors on peut
    # s'attendre à ce que r soit défini jusqu'à environ 2018-10-31
    # On détermine les lignes telles que tous les rendements sont N/A, puis on les
    # supprime.
    r = r[~r.isna().all(axis=1)]

    if trim_data:
        if isinstance(r, pd.Series):
            r = r[~r.isna()]
        elif isinstance(r, pd.DataFrame):
            r = r[~r.isna().any(axis=1)]
    return r


def get_feature_return(t: pd.Series, nb_days=[1, 7, 30]):
    """`t` représente une série temporelle des prix d'un actif. 
    On retourne un dataframe dont chaque colonne représente les rendements des 
    derniers jours, spécifiés par l'argument `nb_days`.
    """
    idx = pd.date_range(start=t.index.min(), end=t.index.max(), freq="D")
    t = t.reindex(idx, method="ffill")

    u = {}
    vf = t
    for day_shift in nb_days:
        vi = t.shift(day_shift)
        r = vf / vi - 1
        r = (1 + r) ** (365 / day_shift) - 1
        r *= 100
        u[f"r_{day_shift}"] = r

    u = pd.DataFrame(u)
    return u


def main():
    u = get_utility()
    t = get_data()
    v = get_target_return(t, months=12)
    l = v.apply(u).mean(axis=0)
    l.sort_values()


# data = get_data()
# r = get_target_return(data)

# cp_u = get_cvx_utility()

# n, p = t.shape
# w = cp.Variable(p)
# gamma = cp.Parameter(nonneg=True)
# objective = 1 / n * cp.sum(cp_u(cp.matmul(t, w)))  # - gamma * cp.norm1(w)
# constraints = [cp.sum(w) == 1, w >= 0]
# prob = cp.Problem(cp.Maximize(objective), constraints)

# gamma.value = 5
# prob.solve(verbose=True, solver="SCS")

# Un autre objectif serait de calibrer l'objectif pour battre le SP500 sur une
# courbe d'utilité.

# Faire un petit exemple simple et l'analyser


# TODO
# Faire une belle grille de tous les profils de rendement avec le ticker et
# leur description.


# universe = ["XBB.TO", "XDV.TO", "XIU.TO", "MSFT"]

# tss = {tckr: get_timeseries(tckr) for tckr in universe}
# rs = {tckr: get_return(ts) for tckr, ts in tss.items()}

# matrix = pd.DataFrame()
# t = pd.DataFrame(dict(zip(universe, rs)))

# mu = t.mean().values
# sigma = t.cov().values


# def minus_markowitz(w):
#     print(w)
#     utility = mu @ w - 0.01 * w @ sigma @ w
#     return -utility


# p = len(universe)
# w = cvx.Variable(p)
# gamma = cvx.Parameter(nonneg=True)
# s0 = cvx.Parameter(nonneg=True)

# objective = mu.T * w - gamma * cvx.quad_form(w, sigma)
# constraints = [cvx.sum(w) == 1, w >= 0]
# prob = cvx.Problem(cvx.Maximize(objective), constraints)
# reg_prob = cvx.Problem(cvx.Maximize(objective - s0 * cvx.norm(w, 1)), constraints)


# opt.minimize(minus_markowitz, x0=np.array([0, 0, 0, 0]))

