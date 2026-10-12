"""Interface Streamlit de ProofAgent.

Lancer : streamlit run app.py
"""
import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.quality import run_checks
from proofagent.agent import MODEL, ask
from proofagent.report import build_note

ROOT = Path(__file__).parent
REPO = "https://github.com/mouadelhaouzi/ProofAgent"

INK, MUTED, LINE, ACCENT = "#1c2126", "#6a6f76", "#dcd7cc", "#1f4e5f"
VERDICTS = {  # libellé affiché, classe CSS
    "validé": ("Validé", "ok"),
    "validé après correction": ("Validé après correction", "fix"),
    "rejeté": ("Rejeté", "ko"),
    "non audité": ("Non audité", "na"),
}
EXAMPLES = [
    "Les accidents sont-ils plus graves hors agglomération qu'en ville ?",
    "Prouvez que la pluie cause des accidents plus graves.",
    "En Lozère (48) en 2023, les conducteurs hommes sont-ils plus gravement blessés que les conductrices ?",
    "Les trottinettes sont-elles devenues beaucoup plus dangereuses depuis 2019 ?",
    "Quelle marque de voiture est impliquée dans le plus d'accidents mortels ?",
]
TOOL_NAMES = {
    "compare_groups": "Comparaison de groupes (khi-deux)",
    "bootstrap_difference": "Écart entre deux groupes (bootstrap)",
    "yearly_trend": "Tendance annuelle (régression logistique)",
    "count": "Comptage",
    "logistic_model": "Régression logistique multivariée",
    "ml_importance": "Gradient boosting et SHAP",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextArea textarea, .stSelectbox, button {{
    font-family: 'IBM Plex Sans', sans-serif;
}}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stHeader"] {{
    display: none !important;
}}
.block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }}

/* en-tête */
.masthead {{ display: flex; justify-content: space-between; align-items: flex-end;
            border-bottom: 2px solid {INK}; padding-bottom: 14px; margin-bottom: 6px; gap: 24px; flex-wrap: wrap; }}
.masthead h1 {{ font-family: 'Source Serif 4', serif; font-weight: 600; font-size: 2.1rem;
               letter-spacing: -0.01em; margin: 0; padding: 0; color: {INK}; }}
.masthead .sub {{ color: {MUTED}; font-size: 0.95rem; margin-top: 4px; }}
.masthead .meta {{ text-align: right; font-size: 0.8rem; color: {MUTED}; line-height: 1.6; }}
.masthead .meta code {{ font-family: 'IBM Plex Mono', monospace; background: none; color: {INK}; padding: 0; }}

/* onglets */
.stTabs [data-baseweb="tab-list"] {{ gap: 28px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 0; font-size: 0.92rem; color: {MUTED}; background: none; }}
.stTabs [aria-selected="true"] {{ color: {INK}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {INK}; height: 2px; }}

/* titres de section */
.label {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
         color: {MUTED}; margin: 26px 0 10px; }}
.label span {{ color: {ACCENT}; margin-right: 8px; font-family: 'IBM Plex Mono', monospace; }}

/* champs */
.stTextArea textarea {{ border-radius: 2px; border: 1px solid {LINE}; background: #fff; font-size: 1rem; }}
.stSelectbox [data-baseweb="select"] > div {{ border-radius: 2px; background: #fff; }}
.stButton button, .stDownloadButton button {{ border-radius: 2px; font-weight: 500; padding: 0.45rem 1.4rem; }}
.stButton button[kind="primary"] {{ background: {INK}; border-color: {INK}; }}
.stButton button[kind="primary"]:hover {{ background: {ACCENT}; border-color: {ACCENT}; }}
.stButton button[kind="primary"] p {{ color: #fff; }}
.stButton button[kind="primary"]:disabled {{ background: #d9d4c9; border-color: #d9d4c9; }}
.stButton button[kind="primary"]:disabled p {{ color: #8b8f94; }}

/* réponse */
.answer {{ font-family: 'Source Serif 4', serif; font-size: 1.22rem; line-height: 1.65; color: {INK};
          border-left: 3px solid {INK}; padding: 4px 0 4px 22px; margin: 6px 0 18px; }}
.stamp {{ display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
         text-transform: uppercase; padding: 4px 10px; border: 1.5px solid; border-radius: 2px; }}
.stamp.ok  {{ color: #2d6a4a; border-color: #2d6a4a; }}
.stamp.fix {{ color: #8a5d12; border-color: #8a5d12; }}
.stamp.ko  {{ color: #9c3324; border-color: #9c3324; }}
.stamp.na  {{ color: {MUTED}; border-color: {MUTED}; }}

/* panneau méthode */
dl.method {{ display: grid; grid-template-columns: 120px 1fr; gap: 8px 14px; margin: 0; font-size: 0.9rem; }}
dl.method dt {{ color: {MUTED}; }}
dl.method dd {{ margin: 0; color: {INK}; }}

/* tableaux */
table.grid {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; border: none; }}
table.grid th, table.grid td {{ border: none !important; background: none !important; }}
table.grid th {{ text-align: left; font-weight: 600; color: {MUTED}; font-size: 0.72rem; letter-spacing: 0.08em;
                text-transform: uppercase; border-bottom: 1px solid {INK} !important; padding: 8px 16px 8px 0; }}
table.grid td {{ border-bottom: 1px solid {LINE} !important; padding: 9px 16px 9px 0; vertical-align: top; color: {INK}; }}
table.grid td.code {{ font-family: 'IBM Plex Mono', monospace; color: {ACCENT}; width: 44px; }}
table.grid td.num {{ font-variant-numeric: tabular-nums; color: #3a4047; }}
table.grid td.status {{ white-space: nowrap; width: 110px; }}
.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }}
.dot.ok {{ background: #2d6a4a; }} .dot.ko {{ background: #9c3324; }}

.notes {{ font-size: 0.84rem; color: {MUTED}; margin: 14px 0 0; padding-left: 18px; }}
.notes li {{ margin-bottom: 4px; }}

/* journal */
.step {{ display: grid; grid-template-columns: 34px 1fr; gap: 12px; padding: 12px 0; border-bottom: 1px solid {LINE}; }}
.step .n {{ font-family: 'IBM Plex Mono', monospace; color: {ACCENT}; font-size: 0.85rem; }}
.step .t {{ font-weight: 600; font-size: 0.9rem; margin-bottom: 4px; }}
.step .c {{ font-size: 0.86rem; color: #3a4047; white-space: pre-wrap; }}
.step .c code {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; background: #ece8df; padding: 1px 4px; }}

/* chiffres clés */
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); border-top: 1px solid {INK}; }}
.kpi {{ padding: 16px 18px 16px 0; border-bottom: 1px solid {LINE}; }}
.kpi .v {{ font-family: 'Source Serif 4', serif; font-size: 2rem; color: {INK}; line-height: 1.1; }}
.kpi .k {{ font-size: 0.8rem; color: {MUTED}; margin-top: 4px; }}

.intro {{ color: #3a4047; font-size: 0.95rem; max-width: 760px; line-height: 1.6; }}
.footer {{ margin-top: 56px; padding-top: 14px; border-top: 1px solid {LINE}; font-size: 0.78rem; color: {MUTED};
          display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }}
.footer a {{ color: {MUTED}; }}
[data-testid="stExpander"] details {{ border-radius: 2px; border-color: {LINE}; background: #fff; }}
</style>
"""


def html(content: str):
    st.markdown(content, unsafe_allow_html=True)


def label(number: str, text: str):
    html(f'<div class="label"><span>{number}</span>{text}</div>')


def chart(result: dict):
    table = pd.DataFrame(result["table"])
    if table.empty:
        return None
    measure = next(c for c in ["taux", "odds_ratio", "importance", "effectif"] if c in table)
    axis_title = {"taux": "Part des usagers tués ou hospitalisés", "odds_ratio": "Odds ratio",
                  "importance": "Part de l'importance SHAP", "effectif": "Effectif"}[measure]
    horizontal = result["tool"] not in ("yearly_trend", "count")
    groups = table["groupe"].astype(str)

    error = None
    if measure == "taux" and "ic_haut" in table:
        error = dict(type="data", symmetric=False, array=table["ic_haut"] - table["taux"],
                     arrayminus=table["taux"] - table["ic_bas"], color=MUTED, thickness=1, width=4)
    bar = go.Bar(
        x=table[measure] if horizontal else groups, y=groups if horizontal else table[measure],
        orientation="h" if horizontal else "v", marker=dict(color=ACCENT, line=dict(width=0)),
        error_x=error if horizontal else None, error_y=None if horizontal else error,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>" if horizontal and measure in ("taux", "importance") else None,
    )
    fig = go.Figure(bar)
    value_axis = dict(title=dict(text=axis_title, font=dict(size=11, color=MUTED)), gridcolor="#e7e3da",
                      zeroline=False, tickfont=dict(size=11, color=MUTED),
                      tickformat=".0%" if measure in ("taux", "importance") else None)
    cat_axis = dict(tickfont=dict(size=12, color=INK), showgrid=False)
    fig.update_layout(
        height=max(260, 34 * len(table) + 80), margin=dict(l=0, r=16, t=8, b=8), bargap=0.45,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif"), showlegend=False,
        xaxis=value_axis if horizontal else cat_axis,
        yaxis=dict(cat_axis, autorange="reversed") if horizontal else value_axis,
    )
    return fig


def facts_table(facts: dict) -> str:
    rows = "".join(f'<tr><td class="code">{k}</td><td>{escape(f["label"])}</td>'
                   f'<td class="num">{escape(f["text"])}</td></tr>' for k, f in facts.items())
    return f'<table class="grid"><tr><th>Réf.</th><th>Indicateur</th><th>Valeur</th></tr>{rows}</table>'


def trace_html(trace: list) -> str:
    out = []
    for i, step in enumerate(trace, 1):
        content = step["content"]
        if step["step"] == "Planificateur" and isinstance(content, dict):
            body = (f"Outil retenu : <code>{escape(str(content.get('tool')))}</code>\n"
                    f"Paramètres : <code>{escape(json.dumps(content.get('args', {}), ensure_ascii=False))}</code>\n"
                    f"{escape(str(content.get('raison', '')))}")
        elif step["step"] == "Calculateur" and isinstance(content, dict):
            body = "\n".join(f"<code>{k}</code> {escape(v['label'])} : {escape(v['text'])}" for k, v in content.items())
        elif step["step"] == "Auditeur" and isinstance(content, dict):
            body = "Aucune anomalie." if content["ok"] else "\n".join(escape(p) for p in content["problems"])
        else:
            body = escape(str(content))
        out.append(f'<div class="step"><div class="n">{i:02d}</div><div><div class="t">{step["step"]}</div>'
                   f'<div class="c">{body}</div></div></div>')
    return "".join(out)


@st.cache_data(show_spinner=False)
def quality_checks():
    return run_checks()


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="ProofAgent", layout="wide")
html(CSS)
html(f"""
<div class="masthead">
  <div>
    <h1>ProofAgent</h1>
    <div class="sub">Analyse statistique auditée des accidents corporels de la route, France 2019 – 2024</div>
  </div>
  <div class="meta">Source : ONISR, data.gouv.fr<br>Modèle local : <code>{MODEL}</code></div>
</div>
""")

tab_ask, tab_eval, tab_quality = st.tabs(["Analyse", "Évaluation", "Qualité des données"])

with tab_ask:
    label("01", "Question")
    col_q, col_opt = st.columns([3, 1], gap="large")
    with col_opt:
        example = st.selectbox("Questions types", ["—"] + EXAMPLES)
        use_auditor = st.toggle("Contrôle par l'auditeur", value=True,
                                help="Désactiver pour voir la réponse brute, sans vérification.")
    with col_q:
        question = st.text_area("Question", value="" if example == "—" else example, height=110,
                                label_visibility="collapsed",
                                placeholder="Ex. : les accidents sont-ils plus graves la nuit ?")
        run = st.button("Lancer l'analyse", type="primary", disabled=not question.strip())

    if run:
        with st.spinner("Analyse en cours. Le modèle tourne en local : comptez de 30 secondes à 2 minutes."):
            st.session_state["state"] = ask(question, use_auditor=use_auditor)

    state = st.session_state.get("state")
    if state:
        verdict_label, verdict_cls = VERDICTS.get(state["verdict"], (state["verdict"], "na"))
        result = state.get("result")

        label("02", "Conclusion")
        html(f'<span class="stamp {verdict_cls}">{verdict_label}</span>'
             f'<div class="answer" style="margin-top:14px">{escape(state["final"])}</div>')

        if result:
            label("03", "Résultats")
            left, right = st.columns([3, 2], gap="large")
            with left:
                fig = chart(result)
                if fig:
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            with right:
                n_users = f"{result['test']['n']:,}".replace(",", " ")
                significant = "Oui, au seuil de 5 %" if result["significant"] else "Non, au seuil de 5 %"
                html(f"""<dl class="method">
                    <dt>Méthode</dt><dd>{TOOL_NAMES.get(result['tool'], result['tool'])}</dd>
                    <dt>Test</dt><dd>{escape(result['test']['name'])}</dd>
                    <dt>Significatif</dt><dd>{significant}</dd>
                    <dt>Effectif</dt><dd>{n_users} usagers</dd>
                </dl>""")
                html('<ul class="notes">' + "".join(f"<li>{escape(w)}</li>" for w in result["warnings"]) + "</ul>")

            label("04", "Chiffres cités")
            html(facts_table(result["facts"]))

        label("05" if result else "03", "Journal d'exécution")
        with st.expander("Afficher les étapes de l'analyse"):
            html(trace_html(state["trace"]))

        st.write("")
        st.download_button("Télécharger la note de synthèse (PDF)", build_note(state),
                           file_name="note_proofagent.pdf", mime="application/pdf")


with tab_eval:
    results_path = ROOT / "outputs" / "eval_results.csv"
    if not results_path.exists():
        html('<p class="intro">Aucun résultat. Lancer <code>python eval/run_eval.py</code>.</p>')
    else:
        df = pd.read_csv(results_path)
        summary = {r["mode"]: r for r in json.loads(
            (ROOT / "outputs" / "eval_summary.json").read_text(encoding="utf-8"))["summary"]}
        verdicts = json.loads((ROOT / "outputs" / "eval_summary.json").read_text(encoding="utf-8"))["verdicts"]
        label("01", "Protocole")
        html('<p class="intro">Vingt questions d\'un assureur fictif, dont huit pièges : causalité, échantillon '
             'trop petit, effet négligeable, absence de données d\'exposition, information absente de la base. '
             'Chaque réponse est comparée avant et après passage par l\'auditeur.</p>')
        label("02", "Résultats")
        without, with_ = summary.get("sans auditeur", {}), summary.get("avec auditeur", {})
        corrected = verdicts.get("validé après correction", 0)
        html(f"""<div class="kpis">
          <div class="kpi"><div class="v">{without.get('violations')}/{without.get('reponses')}</div>
               <div class="k">Réponses non conformes, sans auditeur</div></div>
          <div class="kpi"><div class="v">{with_.get('violations')}/{with_.get('reponses')}</div>
               <div class="k">Réponses non conformes, avec auditeur</div></div>
          <div class="kpi"><div class="v">{without.get('pieges_dejoues')} → {with_.get('pieges_dejoues')}</div>
               <div class="k">Pièges déjoués</div></div>
          <div class="kpi"><div class="v">{corrected}</div>
               <div class="k">Réponses corrigées par l'auditeur</div></div>
        </div>""")
        label("03", "Détail par question")
        view = df[df["mode"] == "avec auditeur"][["id", "piege", "outil", "verdict", "reponse"]].copy()
        view["piege"] = view["piege"].fillna("").str.replace("_", " ").replace("", "—")
        view["outil"] = view["outil"].map(TOOL_NAMES).fillna("Hors périmètre")
        view["verdict"] = view["verdict"].map(lambda v: VERDICTS.get(v, (v,))[0])
        st.dataframe(view, hide_index=True, width="stretch", height=460, column_config={
            "id": st.column_config.TextColumn("Réf.", width=60),
            "piege": st.column_config.TextColumn("Piège", width=120),
            "outil": st.column_config.TextColumn("Méthode", width=230),
            "verdict": st.column_config.TextColumn("Verdict", width=170),
            "reponse": st.column_config.TextColumn("Réponse livrée", width="large"),
        })


with tab_quality:
    checks = quality_checks()
    passed = sum(ok for _, ok, _ in checks)
    label("01", "Contrôles du pipeline DuckDB → Parquet")
    html(f'<p class="intro">{passed} contrôles réussis sur {len(checks)}, exécutés sur les fichiers Parquet produits '
         'à partir des 24 fichiers CSV publiés par l\'ONISR.</p>')
    rows = "".join(f'<tr><td class="status"><span class="dot {"ok" if ok else "ko"}"></span>{"Conforme" if ok else "Échec"}</td>'
                   f'<td>{escape(name)}</td><td class="num">{escape(detail)}</td></tr>' for name, ok, detail in checks)
    html(f'<table class="grid"><tr><th>Statut</th><th>Contrôle</th><th>Détail</th></tr>{rows}</table>')


html(f"""<div class="footer">
  <span>Données : Bases annuelles des accidents corporels de la circulation routière, ONISR — Licence Ouverte.</span>
  <span><a href="{REPO}" target="_blank">Code source</a></span>
</div>""")
