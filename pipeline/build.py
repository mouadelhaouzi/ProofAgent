"""CSV bruts -> 2 fichiers Parquet propres avec DuckDB.

- accidents.parquet : 1 ligne par accident (caractéristiques + lieu)
- usagers.parquet   : 1 ligne par personne impliquée, enrichie avec l'accident et le véhicule

C'est la table usagers qu'utilisent les outils statistiques.
Les codes numériques de la base sont traduits en libellés lisibles (cf. docs ONISR).
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "data" / "raw").as_posix()
OUT = ROOT / "data" / "processed"


def read(table: str) -> str:
    # all_varchar : les formats changent selon les années, on type nous-mêmes après
    return (f"read_csv('{RAW}/{table}_*.csv', delim=';', header=true, all_varchar=true, "
            f"union_by_name=true, filename=true)")


SQL_ACCIDENTS = f"""
CREATE OR REPLACE TABLE accidents AS
WITH caract AS (
    SELECT coalesce(Num_Acc, Accident_Id) AS num_acc, * EXCLUDE (Num_Acc, Accident_Id)
    FROM {read('caract')}
),
lieux AS (
    -- quelques accidents ont plusieurs lignes "lieux" : on garde la première
    SELECT * FROM {read('lieux')}
    QUALIFY row_number() OVER (PARTITION BY Num_Acc) = 1
)
SELECT
    c.num_acc,
    CAST(c.an AS INT)                                        AS annee,
    CAST(c.mois AS INT)                                      AS mois,
    make_date(CAST(c.an AS INT), CAST(c.mois AS INT), CAST(c.jour AS INT)) AS date,
    CAST(split_part(c.hrmn, ':', 1) AS INT)                  AS heure,
    trim(c.dep)                                              AS dep,
    -- quelques coordonnées sont hors limites (erreurs de saisie) : on les met à NULL
    CASE WHEN CAST(replace(c.lat, ',', '.') AS DOUBLE) BETWEEN -25 AND 52
         THEN CAST(replace(c.lat, ',', '.') AS DOUBLE) END   AS lat,
    CASE WHEN CAST(replace(c.long, ',', '.') AS DOUBLE) BETWEEN -180 AND 180
         THEN CAST(replace(c.long, ',', '.') AS DOUBLE) END  AS lon,
    CASE c.lum WHEN '1' THEN 'Jour' WHEN '2' THEN 'Aube/crépuscule'
               WHEN '3' THEN 'Nuit sans éclairage' WHEN '4' THEN 'Nuit sans éclairage'
               WHEN '5' THEN 'Nuit avec éclairage' END       AS luminosite,
    CASE c.agg WHEN '1' THEN 'Hors agglomération' WHEN '2' THEN 'En agglomération' END AS agglomeration,
    CASE trim(c.atm) WHEN '1' THEN 'Normale' WHEN '2' THEN 'Pluie' WHEN '3' THEN 'Pluie'
                     WHEN '4' THEN 'Neige/grêle' WHEN '5' THEN 'Brouillard' WHEN '6' THEN 'Vent fort'
                     WHEN '7' THEN 'Éblouissant' WHEN '8' THEN 'Couvert' WHEN '9' THEN 'Autre' END AS meteo,
    CASE trim(c.col) WHEN '1' THEN 'Frontale' WHEN '2' THEN 'Par l''arrière' WHEN '3' THEN 'Par le côté'
                     WHEN '4' THEN 'En chaîne' WHEN '5' THEN 'Multiple' WHEN '6' THEN 'Autre collision'
                     WHEN '7' THEN 'Sans collision' END      AS collision,
    CASE trim(l.catr) WHEN '1' THEN 'Autoroute' WHEN '2' THEN 'Nationale' WHEN '3' THEN 'Départementale'
                      WHEN '4' THEN 'Communale' ELSE 'Autre' END AS type_route,
    CASE trim(l.surf) WHEN '1' THEN 'Normale' WHEN '2' THEN 'Mouillée' WHEN '3' THEN 'Mouillée'
                      WHEN '4' THEN 'Mouillée' WHEN '5' THEN 'Enneigée' WHEN '7' THEN 'Verglacée'
                      WHEN '6' THEN 'Autre' WHEN '8' THEN 'Autre' WHEN '9' THEN 'Autre' END AS etat_route,
    TRY_CAST(l.vma AS INT)                                   AS vitesse_max
FROM caract c
LEFT JOIN lieux l ON l.Num_Acc = c.num_acc
"""

SQL_USAGERS = f"""
CREATE OR REPLACE TABLE usagers AS
WITH u AS (SELECT * FROM {read('usagers')}),
v AS (
    SELECT * FROM {read('vehicules')}
    QUALIFY row_number() OVER (PARTITION BY Num_Acc, id_vehicule) = 1
)
SELECT
    a.*,
    CASE u.catu WHEN '1' THEN 'Conducteur' WHEN '2' THEN 'Passager' WHEN '3' THEN 'Piéton' END AS role,
    CASE u.sexe WHEN '1' THEN 'Homme' WHEN '2' THEN 'Femme' END AS sexe,
    -- âge > 110 ans = année de naissance mal saisie -> NULL
    CASE WHEN a.annee - TRY_CAST(u.an_nais AS INT) BETWEEN 0 AND 110
         THEN a.annee - TRY_CAST(u.an_nais AS INT) END       AS age,
    CASE WHEN a.annee - TRY_CAST(u.an_nais AS INT) BETWEEN 0 AND 17 THEN '0-17'
         WHEN a.annee - TRY_CAST(u.an_nais AS INT) < 25 THEN '18-24'
         WHEN a.annee - TRY_CAST(u.an_nais AS INT) < 35 THEN '25-34'
         WHEN a.annee - TRY_CAST(u.an_nais AS INT) < 50 THEN '35-49'
         WHEN a.annee - TRY_CAST(u.an_nais AS INT) < 65 THEN '50-64'
         WHEN a.annee - TRY_CAST(u.an_nais AS INT) BETWEEN 65 AND 110 THEN '65+' END AS tranche_age,
    CASE trim(u.trajet) WHEN '1' THEN 'Domicile-travail' WHEN '2' THEN 'Domicile-école'
                        WHEN '3' THEN 'Courses' WHEN '4' THEN 'Professionnel'
                        WHEN '5' THEN 'Loisirs' WHEN '9' THEN 'Autre' END AS trajet,
    CASE trim(u.secu1) WHEN '0' THEN 'Aucun' WHEN '1' THEN 'Ceinture' WHEN '2' THEN 'Casque'
                       WHEN '3' THEN 'Dispositif enfant' WHEN '4' THEN 'Gilet'
                       WHEN '5' THEN 'Airbag' WHEN '6' THEN 'Gants' WHEN '7' THEN 'Gants' END AS equipement,
    CASE WHEN trim(v.catv) IN ('1', '80') THEN 'Vélo'
         WHEN trim(v.catv) IN ('50', '60') THEN 'Trottinette/EDP'
         WHEN trim(v.catv) IN ('2', '30', '31', '32', '33', '34', '41', '42', '43') THEN 'Deux-roues motorisé'
         WHEN trim(v.catv) = '7'  THEN 'Voiture'
         WHEN trim(v.catv) = '10' THEN 'Utilitaire'
         WHEN trim(v.catv) IN ('13', '14', '15', '16', '17') THEN 'Poids lourd'
         WHEN trim(v.catv) IN ('37', '38', '39', '40') THEN 'Transport en commun'
         ELSE 'Autre' END                                    AS vehicule,
    CAST(u.grav AS INT)                                      AS grav_code,
    CASE CAST(u.grav AS INT) WHEN 1 THEN 'Indemne' WHEN 2 THEN 'Tué'
                             WHEN 3 THEN 'Blessé hospitalisé' WHEN 4 THEN 'Blessé léger' END AS gravite,
    -- variable cible principale : tué ou hospitalisé
    CAST(u.grav AS INT) IN (2, 3)                            AS grave,
    CAST(u.grav AS INT) = 2                                  AS tue
FROM u
JOIN accidents a ON a.num_acc = u.Num_Acc
LEFT JOIN v ON v.Num_Acc = u.Num_Acc AND v.id_vehicule = u.id_vehicule
WHERE CAST(u.grav AS INT) IN (1, 2, 3, 4)
"""


def build(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute(SQL_ACCIDENTS)
    con.execute(SQL_USAGERS)
    return con


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = build()
    for table in ("accidents", "usagers"):
        path = (OUT / f"{table}.parquet").as_posix()
        con.execute(f"COPY {table} TO '{path}' (FORMAT parquet)")
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n:,} lignes -> {path}")


if __name__ == "__main__":
    main()
