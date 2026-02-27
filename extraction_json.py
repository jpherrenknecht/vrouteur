


import json
import os
import csv

# IDs à parcourir
IDS = [2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 16, 18, 19, 20, 21]

BASE_PATH = "/home/jp/vrouteur/static/polars"
OUT_CSV = "polars_winch_pro.csv"

rows = []

for pid in IDS:
    fname = f"{BASE_PATH}/polar_{pid}.json"

    if not os.path.isfile(fname):
        print(f"[WARN] Fichier manquant : {fname}")
        continue

    with open(fname, "r", encoding="utf-8") as f:
        pol = json.load(f)

    try:
        rows.append({
            "id": pid,
            "bateau": pol["label"],

            "lws": pol["winch"]["lws"],
            "hws": pol["winch"]["hws"],

            # Sail change (pro)
            "sail_lw_timer": pol["winch"]["sailChange"]["pro"]["lw"]["timer"],
            "sail_lw_ratio": pol["winch"]["sailChange"]["pro"]["lw"]["ratio"],
            "sail_hw_timer": pol["winch"]["sailChange"]["pro"]["hw"]["timer"],
            "sail_hw_ratio": pol["winch"]["sailChange"]["pro"]["hw"]["ratio"],

            # Tack (pro)
            "tack_lw_timer": pol["winch"]["tack"]["pro"]["lw"]["timer"],
            "tack_lw_ratio": pol["winch"]["tack"]["pro"]["lw"]["ratio"],
            "tack_hw_timer": pol["winch"]["tack"]["pro"]["hw"]["timer"],
            "tack_hw_ratio": pol["winch"]["tack"]["pro"]["hw"]["ratio"],

            # Gybe (pro)
            "gybe_lw_timer": pol["winch"]["gybe"]["pro"]["lw"]["timer"],
            "gybe_lw_ratio": pol["winch"]["gybe"]["pro"]["lw"]["ratio"],
            "gybe_hw_timer": pol["winch"]["gybe"]["pro"]["hw"]["timer"],
            "gybe_hw_ratio": pol["winch"]["gybe"]["pro"]["hw"]["ratio"],
        })

    except KeyError as e:
        print(f"[ERROR] Clé manquante dans {fname}: {e}")

# ─────────────────────────────────────────────────────────────
# Écriture CSV
# ─────────────────────────────────────────────────────────────

if not rows:
    raise RuntimeError("Aucune donnée à exporter")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
        delimiter=";",          # Excel / LibreOffice FR
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] CSV généré : {OUT_CSV}")





























import json
import os
from pprint import pprint

# IDs à parcourir
IDS = [2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 16, 18, 19, 20, 21]

BASE_PATH = "/home/jp/vrouteur/static/polars"

rows = []

for pid in IDS:
    fname = f"{BASE_PATH}/polar_{pid}.json"

    if not os.path.isfile(fname):
        print(f"[WARN] Fichier manquant : {fname}")
        continue

    with open(fname, "r", encoding="utf-8") as f:
        pol = json.load(f)

    try:
        row = {
            "id": pid,
            "bateau": pol["label"],
            "lws": pol["winch"]["lws"],
            "hws": pol["winch"]["hws"],

            # Sail change (pro)
            "sail_lw_timer": pol["winch"]["sailChange"]["pro"]["lw"]["timer"],
            "sail_lw_ratio": pol["winch"]["sailChange"]["pro"]["lw"]["ratio"],
            "sail_hw_timer": pol["winch"]["sailChange"]["pro"]["hw"]["timer"],
            "sail_hw_ratio": pol["winch"]["sailChange"]["pro"]["hw"]["ratio"],

            # Tack (pro)
            "tack_lw_timer": pol["winch"]["tack"]["pro"]["lw"]["timer"],
            "tack_lw_ratio": pol["winch"]["tack"]["pro"]["lw"]["ratio"],
            "tack_hw_timer": pol["winch"]["tack"]["pro"]["hw"]["timer"],
            "tack_hw_ratio": pol["winch"]["tack"]["pro"]["hw"]["ratio"],

            # Gybe (pro)
            "gybe_lw_timer": pol["winch"]["gybe"]["pro"]["lw"]["timer"],
            "gybe_lw_ratio": pol["winch"]["gybe"]["pro"]["lw"]["ratio"],
            "gybe_hw_timer": pol["winch"]["gybe"]["pro"]["hw"]["timer"],
            "gybe_hw_ratio": pol["winch"]["gybe"]["pro"]["hw"]["ratio"],
        }

        rows.append(row)

    except KeyError as e:
        print(f"[ERROR] Clé manquante dans {fname}: {e}")

# ─────────────────────────────────────────────────────────────
# Affichage tableau
# ─────────────────────────────────────────────────────────────

if not rows:
    print("Aucune donnée chargée.")
else:
    headers = rows[0].keys()
    colw = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    def print_row(r):
        return " | ".join(str(r[h]).ljust(colw[h]) for h in headers)

    print(print_row({h: h for h in headers}))
    print("-+-".join("-" * colw[h] for h in headers))

    for r in rows:
        print(print_row(r))
