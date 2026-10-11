"""Note de restitution d'une page (PDF) à partir d'une réponse de l'agent."""
from datetime import date

from fpdf import FPDF

# les polices de base du PDF ne gèrent que le latin-1 : on remplace les quelques caractères en dehors
REPLACEMENTS = {"’": "'", "«": '"', "»": '"', "–": "-", "—": "-", "≠": "!=", "œ": "oe", "…": "...", " ": " "}


def _clean(text) -> str:
    text = str(text)
    for a, b in REPLACEMENTS.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_note(state: dict) -> bytes:
    result = state.get("result") or {}
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=15)
    w = pdf.epw

    def write(text, size=10, style="", h=5, color=(30, 30, 30)):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.multi_cell(w, h, _clean(text), new_x="LMARGIN", new_y="NEXT")

    write("ProofAgent - Note de restitution", 16, "B", 8)
    write(f"Données : accidents corporels de la route, data.gouv.fr (2019-2024)  |  {date.today():%d/%m/%Y}",
          8, color=(110, 110, 110))
    pdf.ln(4)

    write("Question", 11, "B")
    write(state["question"])
    pdf.ln(3)

    write("Réponse", 11, "B")
    write(state["final"])
    pdf.ln(3)

    if result:
        write("Chiffres utilisés (tous calculés par les outils statistiques)", 11, "B")
        for key, fact in result["facts"].items():
            write(f"[{key}] {fact['label']} : {fact['text']}", 9, h=4.5)
        pdf.ln(3)
        write("Méthode", 11, "B")
        write(f"{result['test']['name']} - résultat significatif : {'oui' if result['significant'] else 'non'}", 9)
        pdf.ln(3)
        write("Limites", 11, "B")
        for warning in result["warnings"]:
            write(f"- {warning}", 9, h=4.5)
        pdf.ln(3)

    verdict = state.get("verdict", "")
    color = (20, 120, 60) if verdict.startswith("validé") else (180, 40, 40)
    write(f"Verdict de l'auditeur : {verdict.upper()}", 11, "B", color=color)
    if state.get("audit") and state["audit"]["problems"]:
        for p in state["audit"]["problems"]:
            write(f"- {p}", 8, h=4)
    return bytes(pdf.output())
