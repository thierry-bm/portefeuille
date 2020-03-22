"""Le but de module est de trouver des coefficients $k$ qui donnent dans quelle 
proportion devrait être faite notre portefeuille. On ajoute une régularisation $ℓ-1$.

On veut de plus être indépendant au temps, donc trouver des coefficients qui sont
valides peu importe ce qui se passe.

La fonction objectif serait donc la suivante :
max EU(w) = max. 1/n Σ u(r_i^T w)

Autrement dit, on alloue des poids globaux (w ∈ R^p) pour chaque feature.
"""

# %%
from scipy.optimize import minimize
from static import get_data
import utility as ut
from main import get_target_return
import cvxpy as cp
import pandas as pd

#%%
u = ut.get_cvx_utility()

#%%
t = get_data()
r = get_target_return(t, months=12)
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

"""
Résultats : 
w
XBB.TO    0.301021
XFN.TO    0.079503
XGD.TO    0.020709
XQQ.TO    0.396716
XST.TO    0.202051
dtype: float64
"""

#%%
# Une question: est ce qu'on favorise uniquement les titres dont on détient un grand
# échantillon (eg. les ETF qui existent depuis longtemps)

longevity = t.apply(lambda series: series.first_valid_index())
"""
Ainsi on obtient une longévité moyenne qui remonte à 2010-06-14
Nos titres remontent aux périodes suivantes : 

longevity[w.index]
XBB.TO   2000-11-23
XFN.TO   2001-03-29
XGD.TO   2001-03-29
XQQ.TO   2011-05-10
XST.TO   2012-01-24
dtype: datetime64[ns]

La réponse ne paraît donc pas si évidente.
"""

