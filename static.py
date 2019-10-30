import pandas as pd

static_information = pd.read_excel("Data/ProductScreener.xlsx", index_col=0)

# En ce moment, on entraînerait sur des rendements d'une période pré-déterminée,
# par exemple 2 mois. On aurait donc accès aux rendements en tant que variable
# aléatoire. On peut d'abord considérer uniquement ces rendements. Si c'était le
# cas, alors on supposerait que M =  {R_i | i \in \N}.

# Alors quel est le but du jeu encore?
# max_w EU(w^T R) - \lambda||w||^2 - \mu||w||

# Ça pourrait être déjà un bon point de départ. On peut par la suite comparer
# aux différences d'un portefeuille de Markowitz.

# Par contre, à terme on veut tout de même ajouter des variables de marché, ou
# peut-être même les balance sheets qui semblent être présentes avec la
# librairie yfinance. Il faudrait peut être aussi réfléchir à comment mettre au
# point des stratégies plus élaborées.

# Paradoxe: En ce moment, aucun rebalancement ne serait nécessaire, les
# conditions n'évoluant pas. Il faut donc trouver un moyen d'incorporer
# l'information actuelle.