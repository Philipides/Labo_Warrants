"""
Scorer : attribue un score global /100 à chaque warrant.

Critères et pondérations (paramétrables)
─────────────────────────────────────────
1. Delta optimal     25 pts  |delta| idéal entre 0.35 et 0.65
2. Theta faible      20 pts  moins de decay = mieux
3. Levier            20 pts  élasticité calculée normalisée
4. Liquidité         15 pts  score_liquidite déjà calculé par l'analyzer
5. Vega/prix         10 pts  sensibilité vola par euro investi
6. Spread            10 pts  spread bid/ask le plus serré

La normalisation est toujours relative au sous-ensemble filtré par
l'utilisateur (même sous-jacent, même type CALL/PUT), de sorte que
le scoring est toujours comparatif et non absolu.
"""

import pandas as pd
import numpy as np


# Pondérations par défaut — la somme doit valoir 100
DEFAULT_WEIGHTS = {
    "delta":      25,
    "theta":      20,
    "levier":     20,
    "liquidite":  15,
    "vega":       10,
    "spread":     10,
}


def score(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Calcule le score global de chaque warrant.

    Parameters
    ----------
    df      : DataFrame enrichi par analyzer.enrich()
    weights : dict optionnel pour modifier les pondérations

    Returns
    -------
    DataFrame avec colonnes score_* et score_total ajoutées.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    df = df.copy()

    # Each raw scorer returns 0..1 ; multiply by weight → contribution 0..weight
    df["score_delta"]       = _score_delta(df)          * weights["delta"]
    df["score_theta"]       = _score_theta(df)          * weights["theta"]
    df["score_levier"]      = _score_levier(df)         * weights["levier"]
    df["score_liquidite_w"] = (df["score_liquidite"] / 100) * weights["liquidite"]
    df["score_vega"]        = _score_vega(df)           * weights["vega"]
    df["score_spread"]      = _score_spread(df)         * weights["spread"]

    df["score_total"] = (
        df["score_delta"]
        + df["score_theta"]
        + df["score_levier"]
        + df["score_liquidite_w"]
        + df["score_vega"]
        + df["score_spread"]
    ).round(1)

    return df.sort_values("score_total", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions de scoring individuelles — retournent un score 0..1

def _score_delta(df: pd.DataFrame) -> pd.Series:
    """
    Zone optimale : |delta| entre 0.35 et 0.65.
    Pénalité progressive hors zone, maximale pour delta < 0.10 ou > 0.90.
    """
    d = df["delta_abs"].clip(0, 1)
    score = pd.Series(0.0, index=df.index)
    optimal = (d >= 0.35) & (d <= 0.65)
    too_low  = d < 0.35
    too_high = d > 0.65
    score[optimal] = 1.0
    score[too_low]  = (d[too_low]  - 0.10) / 0.25    # 0 @ 0.10, 1 @ 0.35
    score[too_high] = (0.90 - d[too_high]) / 0.25    # 1 @ 0.65, 0 @ 0.90
    return score.clip(0, 1)


def _score_theta(df: pd.DataFrame) -> pd.Series:
    """Theta le plus faible (en % du prix) = meilleur score."""
    t = df["theta_pct_jour"].fillna(df["theta_pct_jour"].max())
    mx = t.max()
    if mx == 0:
        return pd.Series(1.0, index=df.index)
    return (1 - t / mx).clip(0, 1)


def _score_levier(df: pd.DataFrame) -> pd.Series:
    """Levier normalisé — plus c'est élevé, mieux c'est (dans les limites)."""
    l = df["elasticite_calc"].fillna(0)
    # Plafonné à 40× pour éviter que les OTM extrêmes ne dominent
    l = l.clip(0, 40)
    mx = l.max()
    if mx == 0:
        return pd.Series(0.0, index=df.index)
    return (l / mx).clip(0, 1)


def _score_vega(df: pd.DataFrame) -> pd.Series:
    """Ratio vega/prix normalisé — sensibilité vola par euro investi."""
    v = df["ratio_vega_prix"].fillna(0)
    mx = v.max()
    if mx == 0:
        return pd.Series(0.0, index=df.index)
    return (v / mx).clip(0, 1)


def _score_spread(df: pd.DataFrame) -> pd.Series:
    """Spread le plus serré = meilleur score."""
    s = df["spread_pct"].fillna(df["spread_pct"].max())
    mx = s.max()
    if mx == 0:
        return pd.Series(1.0, index=df.index)
    return (1 - s / mx).clip(0, 1)


def get_rank_label(score: float) -> str:
    """Convertit un score numérique en label qualitatif."""
    if score >= 75:
        return "Excellent"
    if score >= 60:
        return "Bon"
    if score >= 45:
        return "Moyen"
    return "Faible"
