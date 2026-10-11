"""Le graphe d'agents (LangGraph).

    question ─► Planificateur (LLM) ─► Calculateur (outils stats) ─► Rédacteur (LLM) ─► Auditeur (règles)
                      ▲                        │                          ▲                    │
                      └── erreur de paramètres ┘                          └──── corrections ───┘

- Le Planificateur choisit l'outil et ses paramètres (JSON) ; si l'outil plante, il corrige son plan.
- Le Calculateur exécute l'outil : c'est lui seul qui produit les chiffres.
- Le Rédacteur écrit la réponse en citant les chiffres par leur code [F1], [F2]...
- L'Auditeur vérifie la réponse ; si elle est refusée, le Rédacteur corrige (2 essais max).
"""
import json
import os
import re
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from proofagent import tools
from proofagent.audit import audit

MODEL = os.getenv("PROOFAGENT_MODEL", "qwen3:4b-instruct")
MAX_RETRIES = 2


class State(TypedDict, total=False):
    question: str
    use_auditor: bool
    plan: dict
    plan_attempts: int
    result: dict | None
    error: str | None
    draft: str            # 1re version du Rédacteur (= ce qu'on livrerait sans auditeur)
    answer: str           # version courante
    audit: dict
    attempts: int
    verdict: str
    final: str            # texte livré au client, chiffres remplacés
    trace: list


def llm(json_mode=False):
    return ChatOllama(model=MODEL, temperature=0, format="json" if json_mode else "", num_ctx=4096)


def _log(state, step, content):
    return state.get("trace", []) + [{"step": step, "content": content}]


# --------------------------------------------------------------------------- nœuds

def _variables_doc():
    lines = []
    for var, desc in tools.VARIABLES.items():
        values = tools.possible_values(var)
        shown = ", ".join(values[:12]) + (" ..." if len(values) > 12 else "")
        lines.append(f"- {var} ({desc}) : {shown}")
    return "\n".join(lines)


PLANNER_PROMPT = """Tu es le planificateur d'un analyste de données. Base : accidents corporels de la route en France,
2019-2024, une ligne par usager impliqué. Tu ne réponds PAS à la question : tu choisis UN outil.

Outils :
{tools}
Cibles (target) possibles : grave (tué ou hospitalisé), tue (tué).
Variables utilisables (by, variables, filters) et leurs valeurs exactes :
{variables}

La base NE contient PAS : alcool, vitesse réelle, marque/modèle de voiture, revenus, sinistres d'assurance,
kilomètres parcourus, permis, antécédents. Si la question en a besoin -> "hors_perimetre".

Conseils : pour « X est-il plus grave que le reste ? », utilise compare_groups avec by = la variable de X.
Pour « A comparé à B », utilise bootstrap_difference. N'utilise "filters" que pour restreindre le périmètre
(une année, un département, un type d'usager...).

Réponds uniquement en JSON : {{"tool": "...", "args": {{...}}, "raison": "une phrase"}}
Exemple : {{"tool": "compare_groups", "args": {{"target": "grave", "by": "meteo", "filters": {{"annee": "2024"}}}}, "raison": "..."}}
"""


def planner(state: State) -> State:
    messages = [("system", PLANNER_PROMPT.format(tools=tools.TOOLS_DOC, variables=_variables_doc())),
                ("user", state["question"])]
    if state.get("error"):  # le plan précédent a échoué : on renvoie l'erreur au planificateur
        messages += [("assistant", json.dumps(state["plan"], ensure_ascii=False)),
                     ("user", f"L'outil a renvoyé une erreur : {state['error']}. Corrige le plan.")]
    plan = json.loads(llm(json_mode=True).invoke(messages).content)
    return {"plan": plan, "plan_attempts": state.get("plan_attempts", 0) + 1,
            "trace": _log(state, "Planificateur", plan)}


def calculator(state: State) -> State:
    plan = state["plan"]
    if plan.get("tool") not in tools.TOOLS:
        return {"result": None, "error": None, "trace": _log(state, "Calculateur", "Hors périmètre : aucun calcul")}
    try:
        result = tools.TOOLS[plan["tool"]](**plan.get("args", {}))
        return {"result": result, "error": None, "trace": _log(state, "Calculateur", result["facts"])}
    except Exception as e:  # paramètres invalides choisis par le LLM
        return {"result": None, "error": str(e), "trace": _log(state, "Calculateur", f"Erreur : {e}")}


WRITER_PROMPT = """Tu es analyste pour un assureur. Tu réponds en français, en 3 à 5 phrases simples.

RÈGLES :
- N'écris AUCUN chiffre toi-même (ni nombre, ni pourcentage, ni p-value). Pour citer un chiffre, écris
  uniquement son code entre crochets, par exemple [F1]. Il sera remplacé automatiquement.
- Appuie-toi sur au moins deux chiffres [F..].
- Ce sont des données d'observation : parle d'association, jamais de cause.
- Termine par une limite de l'analyse.
"""


def _context(state: State) -> str:
    r = state.get("result")
    if r is None:
        reason = state.get("error") or state["plan"].get("raison", "")
        return f"Aucun calcul possible avec ces données ({reason}). Explique-le poliment, sans chiffre."
    facts = "\n".join(f"[{k}] {f['label']} : {f['text']}" for k, f in r["facts"].items())
    sig = "OUI" if r["significant"] else "NON"
    return (f"Méthode : {r['test']['name']}\nChiffres disponibles :\n{facts}\n"
            f"Résultat statistiquement significatif : {sig}\nLimites : {' '.join(r['warnings'])}")


def writer(state: State) -> State:
    messages = [("system", WRITER_PROMPT),
                ("user", f"Question du client : {state['question']}\n\n{_context(state)}")]
    if state.get("audit") and not state["audit"]["ok"]:
        messages += [("assistant", state["answer"]),
                     ("user", "L'auditeur a refusé ta réponse :\n- " + "\n- ".join(state["audit"]["problems"])
                      + "\nRéécris la réponse complète en corrigeant ces points.")]
    answer = llm().invoke(messages).content.strip()
    update = {"answer": answer, "attempts": state.get("attempts", 0) + 1,
              "trace": _log(state, "Rédacteur", answer)}
    if "draft" not in state:
        update["draft"] = answer
    return update


def auditor(state: State) -> State:
    result = audit(state["answer"], state.get("result"), state["question"])
    return {"audit": result, "trace": _log(state, "Auditeur", result)}


def finalize(state: State) -> State:
    if not state.get("use_auditor", True):
        verdict = "non audité"
    elif state["audit"]["ok"]:
        verdict = "validé" if state["attempts"] == 1 else "validé après correction"
    else:
        verdict = "rejeté"
    if verdict == "rejeté":
        final = ("L'auditeur a bloqué cette réponse : elle ne respectait pas les règles de preuve "
                 "(" + "; ".join(p.split(" – ")[0] for p in state["audit"]["problems"]) + ").")
    else:
        final = fill_facts(state["answer"], state.get("result"))
    return {"verdict": verdict, "final": final}


def fill_facts(text: str, result: dict | None) -> str:
    facts = (result or {}).get("facts", {})
    return re.sub(r"\[(F\d+)\]", lambda m: facts.get(m.group(1), {}).get("text", m.group(0)), text)


# --------------------------------------------------------------------------- graphe

def _after_calculator(state: State) -> str:
    return "planner" if state.get("error") and state["plan_attempts"] < 2 else "writer"


def _after_writer(state: State) -> str:
    return "auditor" if state.get("use_auditor", True) else "finalize"


def _after_audit(state: State) -> str:
    if state["audit"]["ok"] or state["attempts"] > MAX_RETRIES:
        return "finalize"
    return "writer"


def build_graph():
    g = StateGraph(State)
    for name, fn in [("planner", planner), ("calculator", calculator), ("writer", writer),
                     ("auditor", auditor), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.set_entry_point("planner")
    g.add_edge("planner", "calculator")
    g.add_conditional_edges("calculator", _after_calculator, ["planner", "writer"])
    g.add_conditional_edges("writer", _after_writer, ["auditor", "finalize"])
    g.add_conditional_edges("auditor", _after_audit, ["writer", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


GRAPH = build_graph()


def ask(question: str, use_auditor: bool = True) -> State:
    return GRAPH.invoke({"question": question, "use_auditor": use_auditor, "trace": []})


if __name__ == "__main__":
    import sys

    out = ask(" ".join(sys.argv[1:]) or "Les accidents sont-ils plus graves hors agglomération ?")
    for step in out["trace"]:
        print(f"\n--- {step['step']}\n{step['content']}")
    print(f"\n=== Verdict : {out['verdict']}\n{out['final']}")
