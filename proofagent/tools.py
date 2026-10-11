"""Outils statistiques : les SEULS endroits où des chiffres sont produits.

Chaque outil renvoie un dict "résultat" avec :
- facts    : les chiffres numérotés F1, F2... que le LLM a le droit de citer (et rien d'autre)
- test     : le test utilisé, sa p-value et la taille d'effet
- significant / negligible : utilisés par l'auditeur
- warnings : limites à mentionner obligatoirement (exposition, causalité...)
- table    : données pour le graphique du dashboard
"""
import json
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "usagers.parquet"
ALPHA = 0.05
SEED = 42

# Variables que l'agent peut utiliser (colonne -> description)
VARIABLES = {
    "sexe": "sexe de l'usager",
    "tranche_age": "tranche d'âge de l'usager",
    "role": "conducteur, passager ou piéton",
    "vehicule": "type de véhicule de l'usager",
    "luminosite": "conditions de lumière",
    "agglomeration": "en ou hors agglomération",
    "meteo": "conditions météo",
    "etat_route": "état de la chaussée",
    "type_route": "type de route",
    "collision": "type de collision",
    "trajet": "motif du trajet",
    "equipement": "équipement de sécurité principal",
    "annee": "année (2019 à 2024)",
    "dep": "département (code, ex: 75, 13, 2A)",
}
TARGETS = {
    "grave": "usager tué ou hospitalisé",
    "tue": "usager tué",
}
TRAP_WARNING_EXPOSURE = ("Ce sont des effectifs, pas un risque : la base ne contient pas l'exposition "
                         "(kilomètres parcourus, nombre d'usagers sur la route).")
WARNING_CAUSAL = "Données observationnelles : on mesure une association, pas une causalité."


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    df = duckdb.sql(f"SELECT * FROM '{DATA.as_posix()}'").df()
    df["annee"] = df["annee"].astype(str)
    return df


def possible_values(var: str) -> list[str]:
    values = load()[var].dropna().unique().tolist()
    return sorted(map(str, values))


# --------------------------------------------------------------------------- helpers

def _filter(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    for var, value in (filters or {}).items():
        if var not in VARIABLES:
            raise ValueError(f"Variable de filtre inconnue : {var}")
        values = value if isinstance(value, list) else [value]
        df = df[df[var].astype(str).isin([str(v) for v in values])]
    if df.empty:
        raise ValueError(f"Aucune donnée pour le filtre {filters}")
    return df


def _check(target: str, *variables: str):
    if target not in TARGETS:
        raise ValueError(f"Cible inconnue : {target} (choix : {list(TARGETS)})")
    for v in variables:
        if v not in VARIABLES:
            raise ValueError(f"Variable inconnue : {v} (choix : {list(VARIABLES)})")


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f} %".replace(".", ",")


def fmt_int(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", " ")


def fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def fmt_num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


class Facts:
    """Petit registre qui numérote les chiffres : F1, F2, ..."""

    def __init__(self):
        self.items = {}

    def add(self, label: str, text: str, value: float | None = None) -> str:
        key = f"F{len(self.items) + 1}"
        self.items[key] = {"label": label, "text": text, "value": value}
        return key


def _result(tool, args, facts, test, significant, negligible, warnings, table):
    return {
        "tool": tool, "args": args, "facts": facts.items, "test": test,
        "significant": bool(significant), "negligible": bool(negligible),
        "warnings": warnings, "table": table,
    }


def _effect_label(v: float) -> str:
    return "négligeable" if v < 0.05 else "faible" if v < 0.1 else "moyen" if v < 0.3 else "fort"


# --------------------------------------------------------------------------- outils

def compare_groups(target: str, by: str, filters: dict | None = None) -> dict:
    """Taux de `target` par modalité de `by` + test du khi-deux + V de Cramér."""
    _check(target, by)
    # erreur fréquente du LLM : filtrer sur la variable qu'on compare -> on ignore ce filtre
    filters = {k: v for k, v in (filters or {}).items() if k != by}
    df = _filter(load(), filters).dropna(subset=[by])
    g = df.groupby(by)[target].agg(["sum", "count"]).sort_values("count", ascending=False).head(10)
    if len(g) < 2:
        raise ValueError(f"Il faut au moins 2 groupes à comparer pour la variable '{by}'.")
    low, high = proportion_confint(g["sum"], g["count"], alpha=ALPHA, method="wilson")
    g["taux"], g["ic_bas"], g["ic_haut"] = g["sum"] / g["count"], low, high

    contingency = np.array([g["sum"], g["count"] - g["sum"]]).T
    chi2, p, _, _ = chi2_contingency(contingency)
    n = int(g["count"].sum())
    cramer_v = float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1))))

    facts = Facts()
    for level, row in g.iterrows():
        facts.add(f"Taux « {TARGETS[target]} » – {by} = {level}",
                  f"{fmt_pct(row.taux)} (IC95 {fmt_pct(row.ic_bas)} à {fmt_pct(row.ic_haut)}, n = {fmt_int(row['count'])})",
                  row.taux)
    facts.add("p-value du khi-deux", fmt_p(p), p)
    facts.add("V de Cramér (taille d'effet)", f"{fmt_num(cramer_v, 3)} ({_effect_label(cramer_v)})", cramer_v)
    facts.add("Nombre d'usagers analysés", fmt_int(n), n)

    warnings = [WARNING_CAUSAL]
    if (g["count"] < 30).any():
        warnings.append("Certains groupes ont moins de 30 usagers : résultats fragiles.")
    table = g.reset_index().rename(columns={by: "groupe"})[["groupe", "taux", "ic_bas", "ic_haut", "count"]]
    return _result("compare_groups", {"target": target, "by": by, "filters": filters or {}}, facts,
                   {"name": "khi-deux d'indépendance", "p_value": p, "effect": cramer_v, "n": n},
                   p < ALPHA, cramer_v < 0.05, warnings, table.to_dict("records"))


def bootstrap_difference(target: str, by: str, group_a: str, group_b: str,
                         filters: dict | None = None, n_boot: int = 2000) -> dict:
    """Différence de taux entre deux groupes, intervalle de confiance par bootstrap."""
    _check(target, by)
    df = _filter(load(), filters)
    a = df.loc[df[by].astype(str) == str(group_a), target].to_numpy(dtype=float)
    b = df.loc[df[by].astype(str) == str(group_b), target].to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"Groupe vide : {group_a if len(a) == 0 else group_b} (valeurs : {possible_values(by)})")

    rng = np.random.default_rng(SEED)
    # rééchantillonner une variable 0/1 revient à tirer dans une binomiale
    boot = rng.binomial(len(a), a.mean(), n_boot) / len(a) - rng.binomial(len(b), b.mean(), n_boot) / len(b)
    diff = a.mean() - b.mean()
    low, high = np.percentile(boot, [2.5, 97.5])
    significant = not (low <= 0 <= high)

    facts = Facts()
    facts.add(f"Taux – {group_a}", f"{fmt_pct(a.mean())} (n = {fmt_int(len(a))})", a.mean())
    facts.add(f"Taux – {group_b}", f"{fmt_pct(b.mean())} (n = {fmt_int(len(b))})", b.mean())
    facts.add("Écart (points de %)", f"{fmt_num(100 * diff, 1)} points", diff)
    facts.add("IC95 bootstrap de l'écart", f"{fmt_num(100 * low, 1)} à {fmt_num(100 * high, 1)} points", None)

    warnings = [WARNING_CAUSAL]
    if min(len(a), len(b)) < 30:
        warnings.append("Moins de 30 usagers dans un des groupes : résultat fragile.")
    table = [{"groupe": str(group_a), "taux": a.mean(), "count": len(a)},
             {"groupe": str(group_b), "taux": b.mean(), "count": len(b)}]
    return _result("bootstrap_difference", {"target": target, "by": by, "group_a": group_a,
                                            "group_b": group_b, "filters": filters or {}}, facts,
                   {"name": f"bootstrap ({n_boot} tirages)", "p_value": None, "effect": diff, "n": len(a) + len(b)},
                   significant, abs(diff) < 0.01, warnings, table)


def yearly_trend(target: str, filters: dict | None = None) -> dict:
    """Évolution 2019-2024 : effectifs et taux de `target`, test de tendance (régression logistique sur l'année)."""
    _check(target)
    df = _filter(load(), filters)
    g = df.groupby("annee")[target].agg(["sum", "count"]).sort_index()
    g["taux"] = g["sum"] / g["count"]

    agg = g.reset_index().assign(year=lambda d: d["annee"].astype(int) - 2019, non=lambda d: d["count"] - d["sum"])
    model = smf.glm("sum + non ~ year", data=agg, family=sm.families.Binomial()).fit()
    p, slope = float(model.pvalues["year"]), float(model.params["year"])

    facts = Facts()
    for year, row in g.iterrows():
        facts.add(f"{year} : usagers concernés / taux", f"{fmt_int(row['sum'])} ({fmt_pct(row.taux)})", row["sum"])
    facts.add("p-value de la tendance", fmt_p(p), p)
    facts.add("Évolution annuelle du risque (odds ratio)", fmt_num(np.exp(slope), 3), np.exp(slope))

    warnings = [WARNING_CAUSAL, "2020 est atypique (confinements, moins de trafic).", TRAP_WARNING_EXPOSURE]
    table = g.reset_index().rename(columns={"annee": "groupe", "sum": "effectif"})[["groupe", "effectif", "taux"]]
    return _result("yearly_trend", {"target": target, "filters": filters or {}}, facts,
                   {"name": "régression logistique sur l'année", "p_value": p, "effect": slope, "n": int(g["count"].sum())},
                   p < ALPHA, abs(np.exp(slope) - 1) < 0.01, warnings, table.to_dict("records"))


def count(filters: dict | None = None) -> dict:
    """Nombre d'accidents, d'usagers, de tués et de blessés graves correspondant au filtre."""
    df = _filter(load(), filters)
    facts = Facts()
    facts.add("Accidents", fmt_int(df["num_acc"].nunique()), df["num_acc"].nunique())
    facts.add("Usagers impliqués", fmt_int(len(df)), len(df))
    facts.add("Tués", fmt_int(df["tue"].sum()), df["tue"].sum())
    facts.add("Tués ou hospitalisés", fmt_int(df["grave"].sum()), df["grave"].sum())
    table = df.groupby("annee")["num_acc"].nunique().reset_index().rename(
        columns={"annee": "groupe", "num_acc": "effectif"})
    return _result("count", {"filters": filters or {}}, facts,
                   {"name": "comptage", "p_value": None, "effect": None, "n": len(df)},
                   False, False, [TRAP_WARNING_EXPOSURE], table.to_dict("records"))


def logistic_model(target: str, variables: list[str], filters: dict | None = None) -> dict:
    """Régression logistique : odds ratios ajustés (chaque facteur 'toutes choses égales par ailleurs')."""
    _check(target, *variables)
    df = _filter(load(), filters).dropna(subset=variables)
    df = df.sample(min(len(df), 150_000), random_state=SEED)
    # référence de chaque variable = sa modalité la plus fréquente (ex : Voiture, Homme...)
    refs = {v: df[v].astype(str).mode()[0] for v in variables}
    df = df.assign(**{v: df[v].astype(str) for v in variables})
    formula = f"{target} ~ " + " + ".join(f"C({v}, Treatment('{refs[v]}'))" for v in variables)
    model = smf.logit(formula, data=df.assign(**{target: df[target].astype(int)})).fit(disp=0)

    ci = model.conf_int()
    res = pd.DataFrame({"or": np.exp(model.params), "low": np.exp(ci[0]), "high": np.exp(ci[1]),
                        "p": model.pvalues}).drop("Intercept")
    res.index = [f"{i.split(',')[0][2:]} = {i.split('[T.')[1][:-1]} (vs {refs[i.split(',')[0][2:]]})" for i in res.index]
    res = res.reindex(res["or"].apply(lambda x: abs(np.log(x))).sort_values(ascending=False).index).head(8)

    facts = Facts()
    for name, row in res.iterrows():
        facts.add(f"Odds ratio {name}",
                  f"{fmt_num(row['or'])} (IC95 {fmt_num(row.low)} à {fmt_num(row.high)}, p {fmt_p(row.p)})", row["or"])
    facts.add("Nombre d'usagers dans le modèle", fmt_int(model.nobs), model.nobs)
    facts.add("Pseudo-R² de McFadden", fmt_num(model.prsquared, 3), model.prsquared)

    warnings = [WARNING_CAUSAL, "Ajusté uniquement sur les variables du modèle (facteurs cachés possibles).",
                "Les odds ratios se lisent par rapport à la modalité de référence de chaque variable."]
    table = res.reset_index().rename(columns={"index": "groupe", "or": "odds_ratio"})
    return _result("logistic_model", {"target": target, "variables": variables, "filters": filters or {}}, facts,
                   {"name": "régression logistique", "p_value": float(res["p"].min()), "effect": None, "n": int(model.nobs)},
                   (res["p"] < ALPHA).any(), False, warnings, table.to_dict("records"))


ML_FEATURES = ["sexe", "tranche_age", "role", "vehicule", "luminosite", "agglomeration",
               "meteo", "etat_route", "type_route", "collision", "trajet"]


@lru_cache(maxsize=8)
def _ml_importance(target: str, filters_key: str) -> dict:
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df = _filter(load(), json.loads(filters_key)).sample(frac=1, random_state=SEED).head(40_000)
    X = pd.get_dummies(df[ML_FEATURES].fillna("Inconnu").astype(str), dtype=float)
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=SEED).fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    shap_values = shap.TreeExplainer(model).shap_values(X_test.head(2000))
    by_column = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
    # on regroupe les colonnes one-hot par variable d'origine
    by_var = by_column.groupby(lambda c: next(f for f in ML_FEATURES if c.startswith(f + "_"))).sum()
    share = (by_var / by_var.sum()).sort_values(ascending=False)
    return {"auc": auc, "share": share.to_dict(), "n": len(df)}


def ml_importance(target: str, filters: dict | None = None) -> dict:
    """Gradient boosting + SHAP : quelles variables aident le plus à PRÉDIRE la gravité."""
    _check(target)
    out = _ml_importance(target, json.dumps(filters or {}, sort_keys=True))
    facts = Facts()
    facts.add("AUC du modèle (0,5 = hasard, 1 = parfait)", fmt_num(out["auc"], 3), out["auc"])
    for var, s in list(out["share"].items())[:6]:
        facts.add(f"Part de l'importance SHAP – {var}", fmt_pct(s), s)
    warnings = [WARNING_CAUSAL, "Importance prédictive ≠ effet causal : une variable peut prédire sans être la cause."]
    table = [{"groupe": k, "importance": v} for k, v in out["share"].items()]
    return _result("ml_importance", {"target": target, "filters": filters or {}}, facts,
                   {"name": "gradient boosting + SHAP", "p_value": None, "effect": out["auc"], "n": out["n"]},
                   out["auc"] > 0.6, out["auc"] < 0.6, warnings, table)


TOOLS = {
    "compare_groups": compare_groups,
    "bootstrap_difference": bootstrap_difference,
    "yearly_trend": yearly_trend,
    "count": count,
    "logistic_model": logistic_model,
    "ml_importance": ml_importance,
}

TOOLS_DOC = """
- compare_groups(target, by, filters) : compare le taux de gravité entre les modalités d'une variable (khi-deux).
- bootstrap_difference(target, by, group_a, group_b, filters) : écart de taux entre DEUX groupes précis avec intervalle de confiance.
- yearly_trend(target, filters) : évolution de 2019 à 2024.
- count(filters) : nombre d'accidents / usagers / tués pour un périmètre.
- logistic_model(target, variables, filters) : effet de plusieurs facteurs en même temps (toutes choses égales par ailleurs).
- ml_importance(target, filters) : quelles variables prédisent le mieux la gravité (gradient boosting + SHAP).
- hors_perimetre : la question ne peut pas être traitée avec ces données.
"""
