import pandas as pd
import yfinance as yf

static_information = pd.read_excel("Data/ProductScreener.xlsx", index_col=0)

tickers_to_remove = ["CIE", "CJP", "CLU", "CRQ", "CWO"]

tickers = [
    ticker + ".TO"
    for ticker in static_information.index
    if not (
        ticker.endswith(".S")
        or ticker.endswith(".U")
        or len(ticker) > 3
        or ticker in tickers_to_remove
    )
]


def reacquire_data(end_dt: str):
    data = yf.download(" ".join(tickers), start="2000-01-01", end=end_dt, threads=True)
    data = data["Adj Close"]
    data.to_excel("Data/HistoricalData.xlsx")


def get_data() -> pd.DataFrame:
    data = pd.read_excel("data/HistoricalData.xlsx", index_col=0, parse_dates=True)
    return data


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
