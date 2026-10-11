"""Télécharge les fichiers BAAC (accidents corporels) 2019-2024 depuis data.gouv.fr."""
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATASET_API = "https://www.data.gouv.fr/api/1/datasets/53698f4ca3a729239d2036df/"
YEARS = range(2019, 2025)
TABLES = {  # nom de table -> préfixes possibles dans les titres data.gouv (ils varient selon les années)
    "caract": ("caracteristiques", "carcteristiques", "caract"),
    "lieux": ("lieux",),
    "vehicules": ("vehicules",),
    "usagers": ("usagers",),
}


def find_urls():
    resources = requests.get(DATASET_API, timeout=30).json()["resources"]
    urls = {}
    for r in resources:
        title = r["title"].lower()
        if "immatricul" in title or not title.endswith(".csv"):
            continue
        for year in YEARS:
            if str(year) not in title:
                continue
            for table, prefixes in TABLES.items():
                if title.startswith(prefixes):
                    urls.setdefault((table, year), r["url"])
    return urls


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    urls = find_urls()
    for (table, year), url in sorted(urls.items()):
        dest = RAW / f"{table}_{year}.csv"
        if dest.exists():
            continue
        print(f"- {dest.name}")
        dest.write_bytes(requests.get(url, timeout=120).content)
    missing = [(t, y) for t in TABLES for y in YEARS if (t, y) not in urls]
    if missing:
        print("Fichiers introuvables :", missing)
    print(f"{len(urls)} fichiers dans {RAW}")


if __name__ == "__main__":
    main()
