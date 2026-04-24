"""
Système Expert Warrants — Interface Streamlit
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import enrich
from src.explainer import explain_warrant
from src.parser import load_multiple
from src.scorer import DEFAULT_WEIGHTS, get_rank_label, score

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Expert Warrants",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Expert Warrants")
    st.caption("Aide à la décision — usage éducatif")

    st.divider()

    # Upload
    st.subheader("1. Données")
    uploaded = st.file_uploader(
        "Charger un ou plusieurs CSV (SG / BNP)",
        type=["csv"],
        accept_multiple_files=True,
    )
    use_sample = st.checkbox("Utiliser les données d'exemple", value=True)

    st.divider()

    # Filtres
    st.subheader("2. Filtres")

    type_warrant = st.radio("Type", ["CALL", "PUT", "Les deux"], horizontal=True)

    sous_jacent_options = ["Tous"]
    echeance_options    = ["Toutes"]

    st.divider()

    # Pondérations
    st.subheader("3. Pondérations (total = 100)")
    w_delta    = st.slider("Delta optimal",  0, 50, DEFAULT_WEIGHTS["delta"])
    w_theta    = st.slider("Theta faible",   0, 50, DEFAULT_WEIGHTS["theta"])
    w_levier   = st.slider("Levier",         0, 50, DEFAULT_WEIGHTS["levier"])
    w_liq      = st.slider("Liquidité",      0, 40, DEFAULT_WEIGHTS["liquidite"])
    w_vega     = st.slider("Vega/Prix",      0, 30, DEFAULT_WEIGHTS["vega"])
    w_spread   = st.slider("Spread",         0, 30, DEFAULT_WEIGHTS["spread"])

    total_w = w_delta + w_theta + w_levier + w_liq + w_vega + w_spread
    if total_w != 100:
        st.warning(f"Total : {total_w}/100 — ajustez les curseurs.")

    custom_weights = {
        "delta": w_delta, "theta": w_theta, "levier": w_levier,
        "liquidite": w_liq, "vega": w_vega, "spread": w_spread,
    }

    st.divider()
    st.caption(
        "⚖️ Outil éducatif uniquement. Pas un conseil en investissement "
        "(MiFID II). Risque de perte totale du capital. "
        "AMF : conseillers-investissement.amf-france.org"
    )

# ──────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_FILES = [DATA_DIR / "sample_sg.csv", DATA_DIR / "sample_bnp.csv"]


@st.cache_data(show_spinner="Chargement et analyse des données...")
def load_and_process(file_paths: list[str], weights: dict) -> pd.DataFrame:
    df = load_multiple(file_paths)
    df = enrich(df)
    df = score(df, weights)
    df["rank_label"] = df["score_total"].apply(get_rank_label)
    return df


def get_file_paths() -> list[str]:
    if uploaded:
        paths = []
        for f in uploaded:
            tmp = DATA_DIR / f.name
            tmp.write_bytes(f.read())
            paths.append(str(tmp))
        return paths
    if use_sample:
        return [str(p) for p in SAMPLE_FILES]
    return []


file_paths = get_file_paths()

if not file_paths:
    st.info("Chargez un CSV en sidebar ou cochez « Utiliser les données d'exemple ».")
    st.stop()

try:
    df_full = load_and_process(tuple(file_paths), tuple(sorted(custom_weights.items())))
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# FILTRES DYNAMIQUES (après chargement)
# ──────────────────────────────────────────────────────────────────────────────

sous_jacents = ["Tous"] + sorted(df_full["sous_jacent"].dropna().unique().tolist())
echeances    = ["Toutes"] + sorted(
    df_full["echeance"].dropna().dt.strftime("%d/%m/%Y").unique().tolist()
)

with st.sidebar:
    sous_jacent_sel = st.selectbox("Sous-jacent", sous_jacents)
    echeance_sel    = st.selectbox("Échéance",    echeances)
    emetteur_opts   = ["Tous"] + sorted(df_full["emetteur"].dropna().unique().tolist())
    emetteur_sel    = st.selectbox("Émetteur",    emetteur_opts)

df = df_full.copy()

if type_warrant != "Les deux":
    df = df[df["type"] == type_warrant]
if sous_jacent_sel != "Tous":
    df = df[df["sous_jacent"] == sous_jacent_sel]
if echeance_sel != "Toutes":
    df = df[df["echeance"].dt.strftime("%d/%m/%Y") == echeance_sel]
if emetteur_sel != "Tous":
    df = df[df["emetteur"] == emetteur_sel]

if df.empty:
    st.warning("Aucun warrant ne correspond aux filtres sélectionnés.")
    st.stop()

# Re-scorer sur le sous-ensemble filtré pour que la normalisation soit relative
df = score(df, custom_weights)
df["rank_label"] = df["score_total"].apply(get_rank_label)

# ──────────────────────────────────────────────────────────────────────────────
# VUE PRINCIPALE
# ──────────────────────────────────────────────────────────────────────────────

tab_liste, tab_detail, tab_compare, tab_glossaire = st.tabs(
    ["Classement", "Fiche détaillée", "Comparaison", "Glossaire"]
)

# ─── TAB 1 : CLASSEMENT ───────────────────────────────────────────────────────

with tab_liste:
    st.subheader(f"Classement — {len(df)} warrants")

    # Graphique score global
    fig_bar = go.Figure()
    colors = df["score_total"].apply(
        lambda s: "#2ecc71" if s >= 70 else ("#f39c12" if s >= 50 else "#e74c3c")
    )
    fig_bar.add_trace(go.Bar(
        x=df["libelle"],
        y=df["score_total"],
        marker_color=list(colors),
        text=df["score_total"],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Score : %{y}/100<br>"
            "<extra></extra>"
        ),
    ))
    fig_bar.update_layout(
        height=350,
        xaxis_title="",
        yaxis_title="Score /100",
        yaxis_range=[0, 105],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=80),
    )
    fig_bar.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Tableau
    display_cols = {
        "libelle":         "Warrant",
        "emetteur":        "Émetteur",
        "type":            "Type",
        "sous_jacent":     "Sous-jacent",
        "strike":          "Strike",
        "echeance":        "Échéance",
        "parite":          "Parité",
        "ask":             "Prix (ask)",
        "delta":           "Delta",
        "elasticite_calc": "Levier×",
        "theta_pct_jour":  "Theta %/j",
        "spread_pct":      "Spread %",
        "score_total":     "Score /100",
        "rank_label":      "Niveau",
        "statut_moneyness":"Moneyness",
        "jours_echeance":  "Jours",
    }

    df_display = df[list(display_cols.keys())].rename(columns=display_cols).copy()
    df_display["Échéance"] = pd.to_datetime(df_display["Échéance"]).dt.strftime("%d/%m/%Y")

    def color_score(val):
        if isinstance(val, (int, float)):
            if val >= 70:
                return "background-color: #d5f5e3; color: #1a5632"
            if val >= 50:
                return "background-color: #fef9e7; color: #7d6608"
            return "background-color: #fdedec; color: #922b21"
        return ""

    styled = df_display.style.applymap(color_score, subset=["Score /100"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ─── TAB 2 : FICHE DÉTAILLÉE ──────────────────────────────────────────────────

with tab_detail:
    libelles = df["libelle"].tolist()
    selected = st.selectbox("Sélectionner un warrant", libelles)

    row = df[df["libelle"] == selected].iloc[0]
    explanations = explain_warrant(row)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score global", f"{row['score_total']}/100", row["rank_label"])
    col2.metric("Delta", f"{row['delta']:.2f}", row["statut_moneyness"])
    col3.metric("Levier", f"{row['elasticite_calc']:.1f}×")
    col4.metric("Échéance dans", f"{int(row['jours_echeance'])} j" if pd.notna(row['jours_echeance']) else "N/A")

    st.divider()

    # Radar des sous-scores
    categories = ["Delta", "Theta", "Levier", "Liquidité", "Vega", "Spread"]
    score_cols  = ["score_delta", "score_theta", "score_levier",
                   "score_liquidite_w", "score_vega", "score_spread"]
    weights_list = [
        custom_weights["delta"], custom_weights["theta"], custom_weights["levier"],
        custom_weights["liquidite"], custom_weights["vega"], custom_weights["spread"],
    ]
    # score_cols are in range 0..weight → normalize back to 0..100
    vals = [
        (row[c] / w * 100) if w > 0 else 0
        for c, w in zip(score_cols, weights_list)
    ]
    vals_closed = vals + [vals[0]]
    cats_closed = categories + [categories[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(52,152,219,0.2)",
        line_color="rgba(52,152,219,0.8)",
        name=selected,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=350,
        margin=dict(t=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Synthèse
    st.markdown("### Synthèse")
    st.markdown(explanations["synthese"])

    # Sections didactiques en accordéon
    with st.expander("Delta — Réactivité au sous-jacent", expanded=True):
        st.markdown(explanations["delta"])

    with st.expander("Theta — Coût du temps"):
        st.markdown(explanations["theta"])

    with st.expander("Levier / Élasticité"):
        st.markdown(explanations["levier"])

    with st.expander("Moneyness — Position par rapport au strike"):
        st.markdown(explanations["moneyness"])

    with st.expander("Vega — Sensibilité à la volatilité"):
        st.markdown(explanations["vega"])

    with st.expander("Liquidité"):
        st.markdown(explanations["liquidite"])

    st.divider()
    st.markdown("### Points de vigilance")
    st.markdown(explanations["risques"])

    st.divider()
    st.markdown("### Verdict")
    st.markdown(explanations["verdict"])

# ─── TAB 3 : COMPARAISON ──────────────────────────────────────────────────────

with tab_compare:
    st.subheader("Comparer 2 ou 3 warrants côte à côte")
    libelles_all = df["libelle"].tolist()
    sel_comp = st.multiselect(
        "Choisir des warrants (2 à 3)",
        libelles_all,
        default=libelles_all[:min(2, len(libelles_all))],
        max_selections=3,
    )

    if len(sel_comp) < 2:
        st.info("Sélectionnez au moins 2 warrants.")
    else:
        rows = [df[df["libelle"] == s].iloc[0] for s in sel_comp]

        compare_metrics = {
            "Score /100":      "score_total",
            "Type":            "type",
            "Strike":          "strike",
            "Moneyness":       "statut_moneyness",
            "Delta":           "delta",
            "|Delta|":         "delta_abs",
            "Levier×":         "elasticite_calc",
            "Theta %/jour":    "theta_pct_jour",
            "Vega":            "vega",
            "Spread %":        "spread_pct",
            "Liquidité /100":  "score_liquidite",
            "Jours échéance":  "jours_echeance",
            "Prix ask (€)":    "ask",
            "Parité":          "parite",
        }

        compare_data = {}
        for metric, col in compare_metrics.items():
            compare_data[metric] = []
            for r in rows:
                val = r[col]
                if isinstance(val, float):
                    compare_data[metric].append(round(val, 3))
                else:
                    compare_data[metric].append(val)

        df_comp = pd.DataFrame(compare_data, index=sel_comp).T
        st.dataframe(df_comp, use_container_width=True)

        # Radar comparatif
        fig_cmp = go.Figure()
        colors_cmp = ["#3498db", "#e74c3c", "#2ecc71"]
        for i, (r, name) in enumerate(zip(rows, sel_comp)):
            vals_c = [
                (r[c] / w * 100) if w > 0 else 0
                for c, w in zip(score_cols, weights_list)
            ]
            vals_c_closed = vals_c + [vals_c[0]]
            fig_cmp.add_trace(go.Scatterpolar(
                r=vals_c_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor=f"rgba({int(colors_cmp[i][1:3],16)},{int(colors_cmp[i][3:5],16)},{int(colors_cmp[i][5:7],16)},0.15)",
                line_color=colors_cmp[i],
                name=name,
            ))
        fig_cmp.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

# ─── TAB 4 : GLOSSAIRE ────────────────────────────────────────────────────────

with tab_glossaire:
    st.subheader("Glossaire des Warrants")

    glossaire = {
        "CALL": (
            "Un warrant **CALL** donne le droit (non l'obligation) d'acheter le sous-jacent "
            "au prix d'exercice (strike). On l'achète quand on anticipe une **hausse** du sous-jacent. "
            "Sa valeur augmente quand le cours monte."
        ),
        "PUT": (
            "Un warrant **PUT** donne le droit de vendre le sous-jacent au strike. "
            "On l'achète quand on anticipe une **baisse**. "
            "Il sert aussi à protéger (hedger) un portefeuille d'actions."
        ),
        "Strike (Prix d'exercice)": (
            "Le prix auquel vous avez le droit d'acheter (CALL) ou de vendre (PUT) "
            "le sous-jacent. C'est le niveau pivot de votre pari."
        ),
        "Parité": (
            "Nombre de warrants nécessaires pour représenter 1 unité du sous-jacent. "
            "Parité 100 = 100 warrants ≡ 1 action (ou 1 indice). "
            "La parité divise le prix du warrant par rapport à l'option équivalente."
        ),
        "Delta (Δ)": (
            "Mesure la variation du prix du warrant pour une variation de 1 unité du sous-jacent. "
            "**CALL** : Delta entre 0 et +1. **PUT** : entre -1 et 0. "
            "Delta = 0.50 ≡ ATM. Delta proche de 1 ≡ ITM profond."
        ),
        "Gamma (Γ)": (
            "Vitesse à laquelle le delta change. Gamma élevé = le delta s'accélère rapidement "
            "quand le sous-jacent bouge. Maximal pour les warrants ATM proches de l'échéance."
        ),
        "Theta (Θ)": (
            "Décroissance quotidienne de la valeur du warrant due au passage du temps. "
            "Toujours négatif pour l'acheteur. Theta s'accélère dans les 30 derniers jours "
            "avant l'échéance (décroissance non linéaire)."
        ),
        "Vega (V)": (
            "Variation du prix du warrant pour une hausse de 1% de la volatilité implicite. "
            "Toujours positif pour l'acheteur. Critique autour des événements (résultats, "
            "banques centrales) : le 'crush de vola' post-événement peut effacer les gains."
        ),
        "ITM (In The Money)": (
            "**CALL ITM** : cours > strike. **PUT ITM** : cours < strike. "
            "Le warrant a une valeur intrinsèque positive. Delta > 0.50 (en valeur absolue)."
        ),
        "ATM (At The Money)": (
            "Cours ≈ strike. Delta ≈ 0.50. Maximum de valeur temps. "
            "Zone de sensibilité maximale au sous-jacent par rapport au coût."
        ),
        "OTM (Out of The Money)": (
            "**CALL OTM** : cours < strike. **PUT OTM** : cours > strike. "
            "Pas de valeur intrinsèque — uniquement de la valeur temps. "
            "Levier élevé mais risque de perte totale si le mouvement n'est pas assez fort."
        ),
        "Élasticité / Levier": (
            "Amplification : si levier = 15×, une hausse de 1% du sous-jacent → +15% du warrant. "
            "Calcul : Delta × (Cours sous-jacent) / (Parité × Prix warrant). "
            "L'effet est symétrique : les pertes sont également amplifiées."
        ),
        "Valeur intrinsèque": (
            "Ce que vaut le warrant si on l'exerçait maintenant. "
            "CALL : max(0, cours - strike) / parité. "
            "PUT : max(0, strike - cours) / parité."
        ),
        "Valeur temps": (
            "Prix du warrant - Valeur intrinsèque. "
            "C'est la prime payée pour la probabilité que le warrant se retrouve ITM. "
            "Elle s'érode avec le temps (theta) et augmente avec la volatilité (vega)."
        ),
        "Spread bid/ask": (
            "Différence entre le prix de vente (ask) et d'achat (bid) teneur de marché. "
            "C'est votre coût de transaction immédiat. "
            "Un spread de 3% signifie que vous perdez 3% dès l'achat."
        ),
        "Volatilité implicite": (
            "La volatilité 'anticipée' par le marché, extraite du prix du warrant. "
            "Elle monte avant les événements (résultats, élections) et chute après. "
            "Un achat sur volatilité haute paie une prime plus élevée."
        ),
    }

    for terme, definition in glossaire.items():
        with st.expander(f"**{terme}**"):
            st.markdown(definition)
