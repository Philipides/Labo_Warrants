"""
Explainer : génère des textes pédagogiques sur mesure pour chaque warrant.

Chaque fonction retourne une chaîne Markdown prête à l'affichage Streamlit.
L'objectif est d'expliquer le POURQUOI du score, pas de répéter des chiffres.
"""

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal

def explain_warrant(row: pd.Series) -> dict[str, str]:
    """
    Retourne un dictionnaire de sections explicatives pour un warrant.

    Clés :
        synthese, delta, theta, levier, liquidite, vega, moneyness,
        risques, verdict
    """
    t = row["type"]
    return {
        "synthese":   _synthese(row),
        "delta":      _explain_delta(row),
        "theta":      _explain_theta(row),
        "levier":     _explain_levier(row),
        "liquidite":  _explain_liquidite(row),
        "vega":       _explain_vega(row),
        "moneyness":  _explain_moneyness(row),
        "risques":    _risques(row),
        "verdict":    _verdict(row),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sections

def _synthese(row: pd.Series) -> str:
    typ    = row["type"]
    ss_jac = row["sous_jacent"]
    strike = row["strike"]
    emit   = row["emetteur"]
    score  = row["score_total"]
    label  = row.get("rank_label", "")
    echeance = row["echeance"].strftime("%d/%m/%Y") if pd.notna(row["echeance"]) else "N/A"

    sens = "hausse" if typ == "CALL" else "baisse"

    return (
        f"Ce warrant **{typ}** émis par **{emit}** vous expose à la **{sens}** "
        f"de **{ss_jac}** avec un strike à **{strike}** et une échéance au **{echeance}**. "
        f"Score global : **{score}/100** ({label})."
    )


def _explain_delta(row: pd.Series) -> str:
    delta     = row["delta"]
    delta_abs = row["delta_abs"]
    typ       = row["type"]
    statut    = row["statut_moneyness"]

    if typ == "CALL":
        direction = "hausse"
        signe_txt = f"**+{delta:.2f}**"
    else:
        direction = "baisse"
        signe_txt = f"**{delta:.2f}**"

    # Interprétation qualitative
    if delta_abs >= 0.65:
        qualite = (
            f"Le delta {signe_txt} est **élevé** : pour chaque point de variation du sous-jacent, "
            f"le warrant suit à {delta_abs*100:.0f}%. "
            f"Avantage : forte réactivité. Inconvénient : warrant plus cher (souvent ITM). "
            f"Profil adapté à un investisseur **convaincu et avec horizon court**."
        )
    elif delta_abs >= 0.35:
        qualite = (
            f"Le delta {signe_txt} est dans la **zone équilibrée** (0.35–0.65). "
            f"C'est le sweet spot : le warrant offre un bon compromis entre "
            f"sensibilité au sous-jacent et coût d'achat. "
            f"Adapté à la majorité des stratégies directionnelles."
        )
    else:
        qualite = (
            f"Le delta {signe_txt} est **faible** : le warrant est peu sensible "
            f"aux mouvements du sous-jacent. C'est souvent un warrant très OTM — "
            f"le levier est théoriquement élevé mais la probabilité d'expirer "
            f"dans la monnaie est faible. **Profil spéculatif, risque de perte totale élevé.**"
        )

    parity_note = (
        f"\n\n> *Delta par warrant = {delta_abs:.2f} / {int(row['parite'])} "
        f"= **{delta_abs/row['parite']:.4f}** "
        f"(variation du prix du warrant pour 1 point du sous-jacent).*"
    )

    return qualite + parity_note


def _explain_theta(row: pd.Series) -> str:
    theta     = row["theta"]
    theta_pct = row["theta_pct_jour"]
    jours     = row["jours_echeance"]

    perte_totale_estimee = theta_pct * jours if pd.notna(jours) else None

    txt = (
        f"**Theta = {theta:.4f} €/jour** — c'est le coût de la durée. "
        f"Chaque jour qui passe, ce warrant perd environ **{theta_pct:.2f}%** de sa valeur "
        f"*si le sous-jacent reste immobile*.\n\n"
    )

    if jours and jours > 0:
        txt += (
            f"Il reste **{int(jours)} jours** avant l'échéance. "
        )
        if perte_totale_estimee and perte_totale_estimee > 50:
            txt += (
                f"Attention : avec ce theta, la valeur temps totale à absorber représente "
                f"environ **{perte_totale_estimee:.0f}%** du prix actuel si le sous-jacent stagne — "
                f"ce warrant nécessite un mouvement **rapide et significatif** en votre faveur."
            )
        elif perte_totale_estimee:
            txt += (
                f"Le coût temps total restant est modéré (**{perte_totale_estimee:.0f}%**). "
                f"L'horizon est confortable pour attendre un mouvement favorable."
            )

    if jours and jours < 30:
        txt += (
            f"\n\n> ⚠️ *Moins de 30 jours à l'échéance : la décroissance temporelle "
            f"s'accélère de façon non linéaire (effet gamma du theta). "
            f"La prudence s'impose si le sous-jacent n'a pas encore bougé.*"
        )

    return txt


def _explain_levier(row: pd.Series) -> str:
    levier = row["elasticite_calc"]
    typ    = row["type"]
    cours  = row["cours_ss_jacent"]
    ask    = row["ask"]
    parite = row["parite"]

    sens = "hausse" if typ == "CALL" else "baisse"

    txt = (
        f"**Effet de levier = {levier:.1f}×** — si le sous-jacent progresse de 1% "
        f"dans le sens favorable ({sens}), le prix du warrant progresse théoriquement "
        f"de **{levier:.1f}%**.\n\n"
        f"*Calcul : Delta ({row['delta_abs']:.2f}) × Cours ({cours}) "
        f"/ (Parité ({int(parite)}) × Prix ask ({ask})) = {levier:.2f}×*\n\n"
    )

    if levier > 25:
        txt += (
            "Le levier est **très élevé** : les gains potentiels sont importants, "
            "mais les pertes le sont tout autant. Un mouvement de -10% du sous-jacent "
            f"peut entraîner une perte de **{min(levier*10, 100):.0f}%** du capital investi. "
            "Réservé aux investisseurs avertis avec une conviction forte."
        )
    elif levier > 12:
        txt += (
            "Le levier est **modéré à élevé** — bon équilibre entre amplification "
            "des gains et risque gérable. C'est la zone de confort pour la plupart "
            "des stratégies sur warrants."
        )
    else:
        txt += (
            "Le levier est **faible** pour un warrant. Cela signifie généralement "
            "un warrant très ITM — comportement proche d'un achat direct du sous-jacent "
            "avec moins de risque de perte totale, mais aussi moins d'amplification."
        )

    return txt


def _explain_liquidite(row: pd.Series) -> str:
    score_liq = row["score_liquidite"]
    spread    = row["spread_pct"]
    volume    = row["volume"]

    if score_liq >= 70:
        qualite = "**excellente**"
        conseil = "Vous pouvez entrer et sortir de cette position avec un faible impact sur le prix."
    elif score_liq >= 40:
        qualite = "**correcte**"
        conseil = "Restez vigilant : passez des ordres limités plutôt qu'au marché."
    else:
        qualite = "**faible**"
        conseil = (
            "⚠️ Liquidité insuffisante : risque de ne pas pouvoir sortir rapidement. "
            "Évitez les grosses positions ou utilisez impérativement des ordres limités."
        )

    return (
        f"Liquidité {qualite} (score {score_liq:.0f}/100).\n\n"
        f"- Spread bid/ask : **{spread:.2f}%** (coût de transaction immédiat à l'achat)\n"
        f"- Volume échangé : **{int(volume):,}** warrants\n\n"
        f"{conseil}"
    )


def _explain_vega(row: pd.Series) -> str:
    vega      = row["vega"]
    ratio_v   = row["ratio_vega_prix"]
    typ       = row["type"]

    txt = (
        f"**Vega = {vega:.4f} €** — pour chaque point de pourcentage de hausse "
        f"de la volatilité implicite, ce warrant gagne {vega:.4f} € par warrant.\n\n"
        f"Ratio Vega/Prix : **{ratio_v:.3f}** — mesure la sensibilité vola par euro investi.\n\n"
    )

    txt += (
        "**Quand le Vega compte-t-il ?**\n"
        "- Avant une publication de résultats ou un événement macro important : "
        "la volatilité implicite monte, les warrants s'apprécient (même si le sous-jacent stagne).\n"
        "- Après un événement : la volatilité s'effondre (**crush de vola**), "
        "les warrants perdent de la valeur même si la direction était bonne.\n\n"
        f"> *Conseil : si vous achetez ce warrant avant un événement catalyseur, "
        f"assurez-vous que le mouvement attendu ({'+' if typ=='CALL' else '-'}X%) "
        f"compense la perte de vola post-événement.*"
    )

    return txt


def _explain_moneyness(row: pd.Series) -> str:
    statut    = row["statut_moneyness"]
    moneyness = row["moneyness"]
    strike    = row["strike"]
    cours     = row["cours_ss_jacent"]
    typ       = row["type"]

    labels = {
        "ITM": ("Dans la monnaie", "a une valeur intrinsèque positive"),
        "ATM": ("À la monnaie",    "est au niveau du marché actuel"),
        "OTM": ("Hors de la monnaie", "n'a pas encore de valeur intrinsèque"),
    }
    nom, desc = labels.get(statut, ("N/A", ""))

    if typ == "CALL":
        distance = cours - strike
        sens_dist = "le sous-jacent doit encore progresser de" if statut == "OTM" else "le sous-jacent est déjà au-dessus du strike de"
    else:
        distance = strike - cours
        sens_dist = "le sous-jacent doit encore baisser de" if statut == "OTM" else "le sous-jacent est déjà en dessous du strike de"

    txt = (
        f"**Statut : {statut} — {nom}**\n\n"
        f"Ce warrant {desc}. "
        f"Strike = {strike}, Cours actuel = {cours}. "
    )

    if statut == "OTM":
        txt += (
            f"Pour que ce warrant expire dans la monnaie, "
            f"{sens_dist} **{abs(distance):.2f} points/€** ({abs(distance/cours*100):.1f}%). "
            "Risque de perte totale si ce niveau n'est pas atteint avant l'échéance."
        )
    elif statut == "ITM":
        txt += (
            f"La valeur intrinsèque est de **{abs(distance):.2f} points/€**. "
            "Ce warrant a déjà de la valeur réelle — moins de risque de perte totale, "
            "mais prix d'achat plus élevé."
        )
    else:
        txt += "Situation équilibrée — delta proche de 0.50, maximum de valeur temps."

    return txt


def _risques(row: pd.Series) -> str:
    risques = []
    jours   = row["jours_echeance"]
    spread  = row["spread_pct"]
    delta   = row["delta_abs"]
    levier  = row["elasticite_calc"]
    statut  = row["statut_moneyness"]

    if jours and jours < 30:
        risques.append("⏳ **Risque temps** : échéance proche, la décroissance temporelle est maximale.")
    if spread > 5:
        risques.append(f"💸 **Risque liquidité** : spread de {spread:.1f}% — coût de transaction élevé.")
    if delta < 0.20:
        risques.append("🎲 **Risque de perte totale** : delta très faible, warrant très OTM.")
    if levier > 30:
        risques.append(f"⚡ **Risque levier excessif** : {levier:.0f}× — amplification symétrique des gains ET des pertes.")
    if statut == "OTM" and jours and jours < 45:
        risques.append("🔴 **Double risque** : OTM + échéance courte = probabilité faible d'expirer ITM.")

    if not risques:
        risques.append("✅ Aucun risque rédhibitoire identifié sur ce warrant.")

    return "\n".join(risques)


def _verdict(row: pd.Series) -> str:
    score  = row["score_total"]
    typ    = row["type"]
    statut = row["statut_moneyness"]
    delta  = row["delta_abs"]
    jours  = row["jours_echeance"]

    if score >= 70:
        opening = "✅ **Warrant recommandé**"
        body = (
            f"Ce warrant présente un profil équilibré : delta dans la zone optimale, "
            f"theta contenu, levier attractif et bonne liquidité. "
            f"Il constitue un bon vecteur pour jouer la {'hausse' if typ == 'CALL' else 'baisse'} "
            f"du sous-jacent avec un rapport risque/rendement favorable."
        )
    elif score >= 50:
        opening = "🟡 **Warrant acceptable avec réserves**"
        body = (
            "Ce warrant peut convenir mais présente quelques faiblesses. "
            "Vérifiez les points signalés dans la section Risques avant de vous positionner. "
            "Limitez la taille de position."
        )
    else:
        opening = "🔴 **Warrant déconseillé**"
        body = (
            "Ce warrant cumule plusieurs handicaps (delta inadapté, theta élevé, "
            "levier excessif ou liquidité insuffisante). "
            "Il existe probablement de meilleurs candidats dans la liste filtrée."
        )

    disclaimer = (
        "\n\n---\n"
        "> ⚖️ *Information à caractère éducatif — pas un conseil en investissement "
        "au sens de la directive MiFID II. Les warrants sont des instruments financiers "
        "complexes pouvant entraîner la perte totale du capital investi. "
        "Consultez un conseiller financier agréé AMF pour toute décision personnelle.*"
    )

    return f"{opening}\n\n{body}{disclaimer}"
