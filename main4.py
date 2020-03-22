"""Même chose que main3 mais on fait varier l'horizon
"""

# %%
from scipy.optimize import minimize
from static import get_data, fund_name
import utility as ut
from main import get_target_return
import cvxpy as cp
import pandas as pd

#%%
u = ut.get_cvx_utility()

#%%
t = get_data()
r = get_target_return(t, months=3)
r = r.fillna(0)  # A voir si c'est necessaire?

#%%
n, p = r.shape
w_cp = cp.Variable(p)
λ = 1
reg = λ * cp.norm1(w_cp)
objective = cp.Maximize(1 / n * cp.sum(u(r.values @ w_cp)) - reg)
constraints = [w_cp >= 0, cp.sum(w_cp) == 1]
prob = cp.Problem(objective, constraints)
result = prob.solve(solver="ECOS", verbose=True)

# %%
w = pd.Series(w_cp.value, index=t.columns)
w = w[w >= 0.01]  # On ne garde que les titres avec un poids >= 1%
w.sort_values(ascending=False).plot.bar()


# %%
w.index = w.index.map(fund_name)
w.sort_values(ascending=False)

"""
Résultats : 
w
iShares Edge MSCI Min Vol USA Index ETF                    0.307989
iShares S&P/TSX Capped Consumer Staples Index ETF          0.214012
iShares Core Canadian Universe Bond Index ETF              0.180307
iShares NASDAQ 100 Index ETF (CAD-Hedged)                  0.160678
iShares S&P/TSX Capped Financials Index ETF                0.048117
iShares S&P/TSX Global Gold Index ETF                      0.046030
iShares S&P/TSX Global Base Metals Index ETF               0.022374
iShares S&P/TSX Capped Information Technology Index ETF    0.015584
dtype: float64
"""

# %%
"""Deux observations: d'abord on semble privilégier des titres qui ne versent pas
forcément des dividendes (à part le Universe Bond Index). Il s'agit alors de recalibrer
cette fonction. 

Autre observation: on remarque que l'optimisation nous retourne de nombreux titres. 
Peut-être un peu trop nombreux. Il serait alors temps d'introduire une régularisation 
ℓ-1.

"""
