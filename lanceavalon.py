import subprocess
import time

AVALON_DIR = "/home/jp/AvalonVirtual-1.30"


def trouve_fenetre(nom):
    result = subprocess.run(
        ["xdotool", "search", "--name", nom],
        capture_output=True,
        text=True
    )

    ids = result.stdout.strip().splitlines()
    return ids[-1] if ids else None


# ============================================================
# 1. Vérifier si Avalon est déjà lancé
# ============================================================

wid = trouve_fenetre("Main menu")

if wid:
    print("Avalon est déjà lancé :", wid)

else:
    print("Lancement d'Avalon...")

    subprocess.Popen(
        ["./runavalon.sh"],
        cwd=AVALON_DIR
    )

    # Attente de la fenêtre principale
    for i in range(30):
        wid = trouve_fenetre("Main menu")

        if wid:
            break

        time.sleep(1)

    if not wid:
        raise RuntimeError("Impossible de démarrer Avalon")

print("Fenêtre principale :", wid)


# ============================================================
# 2. Vérifier si Wind est déjà ouverte
# ============================================================

wind_wid = trouve_fenetre("^Wind$")

if not wind_wid:

    print("Ouverture de la fenêtre Wind...")

    subprocess.run([
        "xdotool",
        "windowactivate",
        "--sync",
        wid
    ])

    time.sleep(0.5)

    # Coordonnées que tu viens de vérifier
    subprocess.run([
        "xdotool",
        "mousemove",
        "--window", wid,
        "500", "320",
        "click", "1"
    ])

    # Attente de Wind
    for i in range(20):

        wind_wid = trouve_fenetre("^Wind$")

        if wind_wid:
            break

        time.sleep(0.5)

    if not wind_wid:
        raise RuntimeError("La fenêtre Wind ne s'est pas ouverte")

else:
    print("La fenêtre Wind est déjà ouverte")

print("Fenêtre Wind :", wind_wid)


# ============================================================
# 3. Mettre Wind au premier plan
# ============================================================

subprocess.run([
    "xdotool",
    "windowactivate",
    "--sync",
    wind_wid
])

time.sleep(0.5)

# ============================================================
# 4. Lancer le chargement des vents 0.25°
# ============================================================

print("Lancement du téléchargement des vents 0.25°")

subprocess.run([
    "xdotool",
    "mousemove",
    "--window", wind_wid,
    "360", "270",
    "click", "1"
])


# ============================================================
# 5. Attendre la fenêtre "Loading forecast"
# ============================================================

print("Attente de la fenêtre Loading forecast...")

loading_wid = None

for i in range(30):

    loading_wid = trouve_fenetre("^Loading forecast$")

    if loading_wid:
        break

    time.sleep(0.5)

if not loading_wid:
    raise RuntimeError("Fenêtre Loading forecast introuvable")

print("Fenêtre Loading forecast :", loading_wid)


# ============================================================
# 6. Attendre la fin du téléchargement
# ============================================================

# TEMPORAIRE : on attend 10 secondes
# On améliorera ensuite cette partie.
time.sleep(10)


# ============================================================
# 7. Cliquer sur Retour
# ============================================================

subprocess.run([
    "xdotool",
    "windowactivate",
    "--sync",
    loading_wid
])

time.sleep(0.5)

subprocess.run([
    "xdotool",
    "mousemove",
    "--window", loading_wid,
    "300", "690",
    "click", "1"
])

print("Retour")



# ============================================================
# Après avoir cliqué sur Retour de "Loading forecast"
# ============================================================

print("Attente du retour à la fenêtre Wind...")

for i in range(20):

    wind_wid = trouve_fenetre("^Wind$")

    # Loading forecast doit avoir disparu
    loading = trouve_fenetre("^Loading forecast$")

    if wind_wid and not loading:
        break

    time.sleep(0.5)

if not wind_wid:
    raise RuntimeError("Fenêtre Wind introuvable")


# ============================================================
# Cliquer sur "Retour" dans Wind
# ============================================================

subprocess.run([
    "xdotool",
    "windowactivate",
    "--sync",
    wind_wid
])

time.sleep(0.5)

subprocess.run([
    "xdotool",
    "mousemove",
    "--window", wind_wid,
    "360", "303",
    "click", "1"
])

print("Retour vers le menu principal")
# ============================================================
# Cliquer sur "Calcul de la Route"
# ============================================================

print("Attente du menu principal...")

main_wid = None

for i in range(20):

    main_wid = trouve_fenetre("Main menu")

    if main_wid:
        break

    time.sleep(0.5)

if not main_wid:
    raise RuntimeError("Menu principal Avalon introuvable")


# Mettre la fenêtre au premier plan
subprocess.run([
    "xdotool",
    "windowactivate",
    "--sync",
    main_wid
])

time.sleep(0.5)


# Calcul de la Route
subprocess.run([
    "xdotool",
    "mousemove",
    "--window", main_wid,
    "500", "352",
    "click", "1"
])

print("Ouverture du calcul de route")