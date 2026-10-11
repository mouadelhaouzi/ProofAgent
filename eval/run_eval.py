"""Évaluation : chaque question passe dans le graphe, on compare SANS et AVEC auditeur.

Astuce : "sans auditeur" = la 1re version du Rédacteur (draft). Avec temperature=0 c'est exactement
ce que produirait le graphe sans le nœud Auditeur, donc on ne fait tourner le LLM qu'une fois par question.

Ce qu'on mesure, sur le texte réellement livré au client :
- violations : la réponse livrée enfreint au moins une règle de preuve (chiffre inventé, causalité...)
- piège déjoué : pour les 8 questions pièges, la réponse livrée ne tombe pas dans le piège
  (pour hors_perimetre : le planificateur a bien refusé de calculer)
"""
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from proofagent.agent import ask, fill_facts
from proofagent.audit import audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def evaluate(q: dict) -> list[dict]:
    start = time.time()
    state = ask(q["question"], use_auditor=True)
    result = state.get("result")
    draft_audit = audit(state["draft"], result, q["question"])

    rows = []
    for mode, text, problems in [
        ("sans auditeur", fill_facts(state["draft"], result), draft_audit["problems"]),
        # avec auditeur, une réponse rejetée n'est pas livrée : pas de violation côté client
        ("avec auditeur", state["final"], [] if state["verdict"] == "rejeté" else state["audit"]["problems"]),
    ]:
        trap_ok = None
        if q["piege"] == "hors_perimetre":
            trap_ok = result is None and not problems
        elif q["piege"]:
            trap_ok = not problems
        rows.append({
            "id": q["id"], "piege": q["piege"] or "", "mode": mode, "outil": state["plan"].get("tool"),
            "verdict": state["verdict"] if mode == "avec auditeur" else "non audité",
            "violation": bool(problems), "regles": " ".join(p.split(" ")[0] for p in problems),
            "piege_dejoue": trap_ok, "reponse": text, "question": q["question"],
            "secondes": round(time.time() - start),
        })
    return rows


def main():
    questions = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    rows = []
    for q in questions:
        print(f"{q['id']} {q['question']}", flush=True)
        try:
            rows += evaluate(q)
        except Exception as e:  # une question qui plante ne doit pas arrêter toute l'évaluation
            print(f"   ERREUR : {e}")
            rows.append({"id": q["id"], "piege": q["piege"] or "", "mode": "erreur", "reponse": str(e),
                         "question": q["question"]})
        print(f"   -> {rows[-1].get('verdict')} ({rows[-1].get('secondes')} s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "eval_results.csv", index=False, encoding="utf-8")

    summary = df[df["mode"] != "erreur"].groupby("mode").agg(
        reponses=("id", "count"),
        violations=("violation", "sum"),
        pieges_dejoues=("piege_dejoue", lambda s: f"{int(s.dropna().sum())}/{s.notna().sum()}"),
    )
    verdicts = df[df["mode"] == "avec auditeur"]["verdict"].value_counts().to_dict()
    (OUT / "eval_summary.json").write_text(json.dumps({"summary": summary.reset_index().to_dict("records"),
                                                       "verdicts": verdicts}, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\n", summary, "\n", verdicts)


if __name__ == "__main__":
    main()
