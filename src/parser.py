"""
Parser : normalise les CSV de Société Générale et BNP Paribas
vers un schéma commun exploitable par l'analyzer.
"""
import pandas as pd
from pathlib import Path

# Colonnes cibles du schéma commun
SCHEMA = [
    "isin", "libelle", "emetteur", "sous_jacent", "type",
    "strike", "echeance", "parite", "bid", "ask",
    "cours_ss_jacent", "delta", "gamma", "theta", "vega",
    "elasticite", "volume", "encours",
]

# Mappings colonne source → colonne cible pour chaque format
_SG_MAP = {
    "isin":             "isin",
    "mnemonique":       "libelle",
    "emetteur":         "emetteur",
    "sous_jacent":      "sous_jacent",
    "type":             "type",
    "strike":           "strike",
    "echeance":         "echeance",
    "parite":           "parite",
    "cours_bid":        "bid",
    "cours_ask":        "ask",
    "cours_ss_jacent":  "cours_ss_jacent",
    "delta":            "delta",
    "gamma":            "gamma",
    "theta":            "theta",
    "vega":             "vega",
    "elasticite":       "elasticite",
    "volume":           "volume",
    "encours":          "encours",
}

_BNP_MAP = {
    "code_isin":        "isin",
    "libelle":          "libelle",
    "emetteur":         "emetteur",
    "sous_jacent":      "sous_jacent",
    "sens":             "type",
    "prix_exercice":    "strike",
    "date_echeance":    "echeance",
    "ratio":            "parite",
    "bid":              "bid",
    "ask":              "ask",
    "cours_reference":  "cours_ss_jacent",
    "delta":            "delta",
    "gamma":            "gamma",
    "theta_jour":       "theta",
    "vega":             "vega",
    "levier":           "elasticite",
    "volume_jour":      "volume",
    "open_interest":    "encours",
}


def _detect_format(columns: list[str]) -> str:
    """Détecte l'émetteur à partir des colonnes du CSV."""
    cols = set(c.lower() for c in columns)
    if "mnemonique" in cols:
        return "SG"
    if "code_isin" in cols or "sens" in cols:
        return "BNP"
    raise ValueError(
        f"Format CSV non reconnu. Colonnes détectées : {list(columns)}\n"
        "Formats supportés : Société Générale, BNP Paribas."
    )


def _normalise_type(val: str) -> str:
    """Normalise CALL/PUT indépendamment de la casse et du libellé."""
    v = str(val).strip().lower()
    if v in ("call", "c", "achat"):
        return "CALL"
    if v in ("put", "p", "vente"):
        return "PUT"
    return val.upper()


def _apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Renomme les colonnes selon le mapping et projette sur le schéma cible."""
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None
    return df[SCHEMA]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Charge et normalise un fichier CSV SG ou BNP."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    fmt = _detect_format(df.columns)
    mapping = _SG_MAP if fmt == "SG" else _BNP_MAP
    df = _apply_mapping(df, mapping)

    # Conversions de types
    df["echeance"] = pd.to_datetime(df["echeance"], dayfirst=False, errors="coerce")
    numeric_cols = ["strike", "parite", "bid", "ask", "cours_ss_jacent",
                    "delta", "gamma", "theta", "vega", "elasticite", "volume", "encours"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["type"] = df["type"].apply(_normalise_type)
    df["source_format"] = fmt
    return df


def load_multiple(paths: list[str | Path]) -> pd.DataFrame:
    """Charge et concatène plusieurs CSV (SG + BNP)."""
    frames = [load_csv(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["isin"])
