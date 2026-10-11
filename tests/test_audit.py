"""L'auditeur est du code déterministe : on peut le tester sans LLM."""
from proofagent.audit import audit

RESULT = {"tool": "compare_groups", "significant": True, "negligible": False,
          "warnings": ["Données observationnelles : association, pas causalité."],
          "facts": {"F1": {"label": "Taux – tranche_age = 18-24"}, "F2": {"label": "Taux – tranche_age = 65+"}}}


def problems(text, result=RESULT, question=""):
    return [p[:2] for p in audit(text, result, question)["problems"]]


def test_good_answer_passes():
    assert problems("Le taux est de [F1] chez les 18-24 ans contre [F2] chez les 65+. "
                    "C'est une association, pas une preuve de cause.") == []


def test_invented_number_is_rejected():
    assert "R1" in problems("Le taux est de [F1], soit environ 20 % de plus.")


def test_numbers_from_question_are_allowed():
    assert "R1" not in problems("Chez les 18 à 24 ans le taux est [F1].", question="Les 18-24 ans ?")


def test_causal_claim_is_rejected():
    assert "R4" in problems("Le taux est [F1] : l'âge provoque des accidents plus graves.")


def test_causal_disclaimer_is_accepted():
    assert "R4" not in problems("Le taux est [F1]. Les données ne permettent pas de conclure à une cause.")


def test_non_significant_must_be_said():
    result = {**RESULT, "significant": False}
    assert "R2" in problems("Les hommes ont un taux de [F1], plus élevé.", result)
    assert "R2" not in problems("L'écart [F1] n'est pas statistiquement significatif.", result)


def test_negligible_effect_must_be_said():
    result = {**RESULT, "negligible": True}
    assert "R3" in problems("La météo est liée à la gravité [F1].", result)


def test_exposure_warning():
    result = {**RESULT, "warnings": ["Pas de données d'exposition."]}
    assert "R5" in problems("Il y a eu [F1] accidents.", result)
    assert "R5" not in problems("Il y a eu [F1] accidents, sans tenir compte de l'exposition.", result)


def test_answer_without_any_fact_is_rejected():
    assert "R6" in problems("Les jeunes sont plus souvent gravement blessés.")


def test_out_of_scope_only_checks_numbers():
    assert audit("Cette donnée n'est pas dans la base.", None)["ok"]
