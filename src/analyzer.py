"""
Analyzer : enrichit le DataFrame normalisé avec des métriques dérivées.

Métriques calculées
───────────────────
- moneyness       : rapport cours_ss_jacent / strike (CALL) ou strike / cours (PUT)
- statut_moneyness: ITM / ATM / OTM
- delta_abs       : |delta| — pratique pour comparer CALL et PUT
- spread_pct      : (ask - bid) / ask × 100  → mesure de liquidité
- elasticite_calc : delta × (cours_ss_jacent / (parite × ask))  si non fournie
- theta_annuel    : theta × 365  → perte annuelle en % du prix ask
- ratio_vega_prix : vega / ask  → sensibilité vola par euro investi
- jours_echeance  : jours calendaires restants
- score_liquidite : note 0-100 basée sur volume + spread
"""

import numpy as np
import pandas as pd
from datetime import date


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute toutes les colonnes d'analyse au DataFrame."""
    df = df.copy()

    df = _moneyness(df)
    df = _spread(df)
    df = _elasticite(df)
    df = _theta_metrics(df)
    df = _vega_metrics(df)
    df = _time_metrics(df)
    df = _liquidity_score(df)
    return df


# ──────────────────────────────────────────────────────────────────────────────

def _moneyness(df: pd.DataFrame) -> pd.DataFrame:
    call_mask = df["type"] == "CALL"
    put_mask  = df["type"] == "PUT"

    df["moneyness"] = np.nan
    df.loc[call_mask, "moneyness"] = (
        df.loc[call_mask, "cours_ss_jacent"] / df.loc[call_mask, "strike"]
    )
    df.loc[put_mask, "moneyness"] = (
        df.loc[put_mask, "strike"] / df.loc[put_mask, "cours_ss_jacent"]
    )

    def _statut(row):
        m = row["moneyness"]
        if pd.isna(m):
            return "N/A"
        if m > 1.02:
            return "ITM"
        if m < 0.98:
            return "OTM"
        return "ATM"

    df["statut_moneyness"] = df.apply(_statut, axis=1)
    df["delta_abs"] = df["delta"].abs()
    return df


def _spread(df: pd.DataFrame) -> pd.DataFrame:
    df["spread_pct"] = ((df["ask"] - df["bid"]) / df["ask"] * 100).round(2)
    return df


def _elasticite(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule l'élasticité si absente ou nulle dans le CSV."""
    missing = df["elasticite"].isna() | (df["elasticite"] == 0)
    calc = (
        df["delta_abs"] * df["cours_ss_jacent"] / (df["parite"] * df["ask"])
    )
    df["elasticite_calc"] = df["elasticite"].where(~missing, calc).round(2)
    return df


def _theta_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Theta est négatif dans les CSV (coût quotidien en €)
    df["theta_abs"] = df["theta"].abs()
    # % de la valeur du warrant perdu par jour
    df["theta_pct_jour"] = (df["theta_abs"] / df["ask"] * 100).round(3)
    return df


def _vega_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Sensibilité vola par euro investi
    df["ratio_vega_prix"] = (df["vega"] / df["ask"]).round(3)
    return df


def _time_metrics(df: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp(date.today())
    df["jours_echeance"] = (df["echeance"] - today).dt.days
    return df


def _liquidity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score de liquidité 0-100 basé sur volume (70%) + spread (30%)."""
    vol = df["volume"].fillna(0)
    spread = df["spread_pct"].fillna(100)

    vol_norm   = _minmax(vol)   # 1 = meilleur volume
    spread_norm = 1 - _minmax(spread)   # 1 = spread le plus serré

    df["score_liquidite"] = (0.7 * vol_norm + 0.3 * spread_norm) * 100
    df["score_liquidite"] = df["score_liquidite"].round(1)
    return df


def _minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)
