"""Tests de qualité des données, lancés après le build.

Chaque check renvoie (nom, ok, détail). `python pipeline/quality.py` affiche le rapport,
et `pytest` les lance aussi (tests/test_quality.py).
"""
from pathlib import Path

import duckdb

DATA = (Path(__file__).resolve().parents[1] / "data" / "processed").as_posix()
ACC = f"'{DATA}/accidents.parquet'"
USA = f"'{DATA}/usagers.parquet'"


def _one(con, sql):
    return con.execute(sql).fetchone()[0]


def run_checks(con=None):
    con = con or duckdb.connect()
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    n_dup = _one(con, f"SELECT count(*) - count(DISTINCT num_acc) FROM {ACC}")
    check("Identifiant accident unique", n_dup == 0, f"{n_dup} doublons")

    years = [r[0] for r in con.execute(f"SELECT DISTINCT annee FROM {ACC} ORDER BY 1").fetchall()]
    check("Les 6 années 2019-2024 sont présentes", years == list(range(2019, 2025)), str(years))

    per_year = con.execute(f"SELECT annee, count(*) FROM {ACC} GROUP BY 1").fetchall()
    # ordre de grandeur connu : ~45 000 à 60 000 accidents corporels par an
    check("Volume annuel plausible", all(40_000 < n < 65_000 for _, n in per_year), str(dict(per_year)))

    orphans = _one(con, f"SELECT count(*) FROM {USA} WHERE num_acc NOT IN (SELECT num_acc FROM {ACC})")
    check("Aucun usager orphelin", orphans == 0, f"{orphans} usagers sans accident")

    bad_grav = _one(con, f"SELECT count(*) FROM {USA} WHERE gravite IS NULL")
    check("Gravité toujours renseignée", bad_grav == 0, f"{bad_grav} valeurs manquantes")

    deaths = con.execute(f"SELECT annee, sum(tue::int) FROM {USA} GROUP BY 1").fetchall()
    # chiffres officiels ONISR : ~3 200 à 3 600 tués par an, 2 780 en 2020 (confinements)
    check("Nombre de tués cohérent avec l'ONISR", all(2_500 < n < 3_800 for _, n in deaths), str(dict(deaths)))

    bad_age = _one(con, f"SELECT count(*) FROM {USA} WHERE age < 0 OR age > 110")
    check("Âges entre 0 et 110 ans", bad_age == 0, f"{bad_age} âges aberrants")

    na_age = _one(con, f"SELECT avg((age IS NULL)::int) FROM {USA}")
    check("Moins de 5 % d'âges manquants", na_age < 0.05, f"{na_age:.1%}")

    bad_geo = _one(con, f"SELECT count(*) FROM {ACC} WHERE lat NOT BETWEEN -25 AND 52 OR lon NOT BETWEEN -180 AND 180")
    check("Coordonnées GPS valides (métropole + DOM)", bad_geo == 0, f"{bad_geo} points hors limites")

    na_vehicle = _one(con, f"SELECT avg((vehicule = 'Autre')::int) FROM {USA}")
    check("Moins de 5 % de véhicules non classés", na_vehicle < 0.05, f"{na_vehicle:.1%}")
    return checks


if __name__ == "__main__":
    results = run_checks()
    for name, ok, detail in results:
        print(f"[{'OK' if ok else 'KO'}] {name} ({detail})")
    print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} checks OK")
