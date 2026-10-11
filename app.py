"""Dashboard Streamlit : poser une question, voir le raisonnement des agents, exporter la note.

Lancer : streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline.quality import run_checks
from proofagent.agent import MODEL, ask
from proofagent.report import build_note

ROOT = Path(__file__).parent
BLUE = "#2a6fdb"
VERDICT_STYLE = {"validé": ("✅", "success"), "validé après correction": ("🛠️", "success"),
                 "rejeté": ("⛔", "error"), "non audité": ("⚠️", "warning")}
EXAMPLES = [
    "Les accidents sont-ils plus graves hors agglomération qu'en ville ?",
    "Prouvez que la pluie cause des accidents plus graves.",
    "En Lozère (48) en 2023, les conducteurs hommes sont-ils plus gravement blessés que les conductrices ?",
    "Les trottinettes sont-elles devenues beaucoup plus dangereuses depuis 2019 ?",
    "Quelle marque de voiture est impliquée dans le plus d'accidents mortels ?",
]

st.set_page_config(page_title="ProofAgent", page_icon="🔎", layout="wide")
st.title("🔎 ProofAgent")
st.caption("Un analyste IA qui doit prouver ce qu'il affirme — accidents de la route en France, 2019-2024 "
           f"(data.gouv.fr) · LLM local : `{MODEL}`")

tab_ask, tab_eval, tab_quality = st.tabs(["Poser une question", "Évaluation", "Qualité des données"])


def chart(result: dict):
    table = pd.DataFrame(result["table"])
    if table.empty:
        return None
    measure = next(c for c in ["taux", "odds_ratio", "importance", "effectif"] if c in table)
    labels = {"taux": "Taux (tué ou hospitalisé)", "odds_ratio": "Odds ratio", "importance": "Part de l'importance SHAP",
              "effectif": "Effectif", "groupe": ""}
    error = {}
    if measure == "taux" and "ic_haut" in table:
        error = {"error_x": table["ic_haut"] - table["taux"], "error_x_minus": table["taux"] - table["ic_bas"]}
    horizontal = result["tool"] != "yearly_trend" and result["tool"] != "count"
    fig = px.bar(table, x=measure if horizontal else "groupe", y="groupe" if horizontal else measure,
                 orientation="h" if horizontal else "v", labels=labels, **error)
    fig.update_traces(marker_color=BLUE, marker_line_width=0)
    if measure in ("taux", "importance"):
        fig.update_layout(xaxis_tickformat=".0%" if horizontal else None, yaxis_tickformat=None if horizontal else ".0%")
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=320, margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


with tab_ask:
    col_q, col_opt = st.columns([4, 1])
    with col_opt:
        use_auditor = st.toggle("Auditeur activé", value=True,
                                help="Désactive-le pour voir ce que le LLM aurait répondu sans contrôle.")
        example = st.selectbox("Exemples", [""] + EXAMPLES)
    with col_q:
        question = st.text_area("Question de l'assureur", value=example, height=90)
        go = st.button("Analyser", type="primary", disabled=not question.strip())

    if go:
        with st.spinner("Les agents travaillent… (LLM local sur CPU : 30 s à 2 min)"):
            st.session_state["state"] = ask(question, use_auditor=use_auditor)

    state = st.session_state.get("state")
    if state:
        icon, kind = VERDICT_STYLE.get(state["verdict"], ("", "info"))
        getattr(st, kind)(f"{icon} Verdict de l'auditeur : **{state['verdict']}**")
        st.markdown(f"> {state['final']}")

        result = state.get("result")
        if result:
            left, right = st.columns([3, 2])
            with left:
                fig = chart(result)
                if fig:
                    st.plotly_chart(fig, width="stretch")
            with right:
                st.markdown(f"**Méthode :** {result['test']['name']}  \n"
                            f"**Significatif :** {'oui' if result['significant'] else 'non'}")
                st.dataframe(pd.DataFrame([{"code": k, "chiffre": f["label"], "valeur": f["text"]}
                                           for k, f in result["facts"].items()]),
                             hide_index=True, width="stretch")
                for w in result["warnings"]:
                    st.caption(f"⚠️ {w}")

        with st.expander("Voir le raisonnement des agents"):
            for step in state["trace"]:
                st.markdown(f"**{step['step']}**")
                content = step["content"]
                if isinstance(content, dict) and all(k.startswith("F") for k in content):
                    content = {k: f"{v['label']} : {v['text']}" for k, v in content.items()}
                st.write(content)

        st.download_button("📄 Exporter la note de restitution (PDF)", build_note(state),
                           file_name="note_proofagent.pdf", mime="application/pdf")


with tab_eval:
    results_path = ROOT / "outputs" / "eval_results.csv"
    if not results_path.exists():
        st.info("Lance d'abord `python eval/run_eval.py`.")
    else:
        df = pd.read_csv(results_path)
        summary = json.loads((ROOT / "outputs" / "eval_summary.json").read_text(encoding="utf-8"))
        st.subheader("20 questions, dont 8 pièges")
        cols = st.columns(len(summary["summary"]))
        for col, row in zip(cols, summary["summary"]):
            col.metric(f"{row['mode'].capitalize()} — réponses avec violation",
                       f"{row['violations']}/{row['reponses']}")
            col.metric(f"{row['mode'].capitalize()} — pièges déjoués", row["pieges_dejoues"])
        st.caption("Verdicts de l'auditeur : " + ", ".join(f"{k} : {v}" for k, v in summary["verdicts"].items()))
        st.dataframe(df[["id", "piege", "mode", "outil", "verdict", "regles", "reponse"]],
                     hide_index=True, width="stretch")


with tab_quality:
    st.subheader("Tests de qualité du pipeline DuckDB → Parquet")
    checks = run_checks()
    st.metric("Checks réussis", f"{sum(ok for _, ok, _ in checks)}/{len(checks)}")
    st.dataframe(pd.DataFrame([{"": "✅" if ok else "❌", "test": name, "détail": d} for name, ok, d in checks]),
                 hide_index=True, width="stretch")
