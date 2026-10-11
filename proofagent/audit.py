"""L'auditeur : des règles simples et explicables, pas un 2e LLM.

Il relit la réponse rédigée par le LLM et la rejette si :
  R1  elle contient un chiffre qui ne vient pas d'un outil (tout chiffre doit être cité via [F1], [F2]...).
      Seuls les nombres déjà présents dans la question ou dans les libellés (ex : « 18-24 ans ») sont tolérés.
  R2  elle présente comme réel un écart non significatif
  R3  elle ne précise pas qu'un effet significatif est en réalité négligeable
  R4  elle affirme une causalité (données observationnelles)
  R5  elle oublie de signaler l'absence de données d'exposition quand c'est nécessaire
  R6  elle ne cite aucun chiffre alors qu'un calcul a été fait (une affirmation doit être prouvée)
"""
import re

FACT_TAG = re.compile(r"\[F\d+\]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

CAUSAL = re.compile(
    r"\b(caus\w*|provoqu\w*|entra[iî]n\w*|à cause|responsables?|parce qu|dû|due?s? à|"
    r"conduit à|mène à|rend\w* (les|plus|moins)|explique\w*|prouv\w*|démontr\w*|effet de)\b",
    re.IGNORECASE,
)
# une phrase causale avec une négation (« ne permet pas de conclure à une cause ») est une mise en garde, pas une affirmation
CAUSAL_DISCLAIMER = re.compile(r"(\bne\s+\w+\s+pas\b|\bn'\w+\s+pas\b|\bpas\s+(de|d'|une?)\b|\bsans\b|"
                               r"association|corrélation)", re.IGNORECASE)
NOT_SIGNIFICANT = re.compile(r"(pas|non|aucun\w*)\s+(\w+\s+){0,3}significati", re.IGNORECASE)
NEGLIGIBLE = re.compile(r"(négligeable|faible|modeste|minime|limité|légère|léger)", re.IGNORECASE)
EXPOSURE = re.compile(r"(exposition|kilom|km parcourus|trafic|nombre d'usagers sur la route)", re.IGNORECASE)


def _sentences_with(pattern: re.Pattern, text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if pattern.search(s)]


def audit(draft: str, result: dict | None, question: str = "") -> dict:
    """Renvoie {"ok": bool, "problems": [...]} pour une réponse brouillon."""
    problems = []

    labels = " ".join(f["label"] for f in (result or {}).get("facts", {}).values())
    allowed = set(NUMBER.findall(question + " " + labels))
    unsourced = [n for n in NUMBER.findall(FACT_TAG.sub("", draft)) if n not in allowed]
    if unsourced:
        problems.append(f"R1 – Chiffre non sourcé ({unsourced[0]}) : n'écris aucun chiffre toi-même, "
                        "cite uniquement [F1], [F2]...")

    if result is None:  # question hors périmètre : seule la règle des chiffres s'applique
        return {"ok": not problems, "problems": problems}

    if not FACT_TAG.search(draft):
        problems.append("R6 – Aucun chiffre cité : appuie ta réponse sur les chiffres [F1], [F2]... fournis.")

    # p-value non significative, ou intervalle bootstrap qui contient 0
    if not result["significant"] and result["tool"] != "count" and not NOT_SIGNIFICANT.search(draft):
        problems.append("R2 – Le résultat n'est PAS statistiquement significatif : dis-le explicitement "
                        "et ne conclus pas à une différence.")

    if result["significant"] and result["negligible"] and not NEGLIGIBLE.search(draft):
        problems.append("R3 – L'effet est significatif mais négligeable en taille : précise-le.")

    causal = [s for s in _sentences_with(CAUSAL, draft) if not CAUSAL_DISCLAIMER.search(s)]
    if causal:
        problems.append(f"R4 – Formulation causale interdite (« {causal[0].strip()[:80]} ») : "
                        "parle d'association ou de lien statistique, pas de cause.")

    needs_exposure = any("exposition" in w for w in result["warnings"])
    if needs_exposure and not EXPOSURE.search(draft):
        problems.append("R5 – Précise que ces chiffres ne tiennent pas compte de l'exposition "
                        "(kilomètres parcourus), donc qu'ils ne mesurent pas un risque.")

    return {"ok": not problems, "problems": problems}
