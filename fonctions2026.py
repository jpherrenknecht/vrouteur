import time
import os
import math
from datetime import datetime,timezone
from cachetools import TTLCache
from threading import Lock
import torch
import numpy as np
import json
import sqlite3 as sql
import zlib
import requests
import folium
import calendar
from scipy.signal import savgol_filter
from scipy.ndimage import label, generate_binary_structure
from urllib.request import urlretrieve
from collections import defaultdict, deque     # necessaire pour la carte re



from psycopg2 import pool
# utilitaires postgres 
pg_pool = pool.SimpleConnectionPool(
    1, 10,  # minconn, maxconn
    dbname="vrouteur",
    user="jp",
    password="Licois1000",
    host="localhost",                   # ou l'adresse IP/nom de domaine
    port="5432"                         # par défaut PostgreSQL
    )

conn = pg_pool.getconn()
cursor = conn.cursor()



Pi=math.pi
R_NM = 3440.065  # rayon moyen de la Terre en milles nautiques
KNOTS = 1.94384



tabCoeffboat = [
        {"_id":0      ,"name": "unknow",               "stamina": 1},
        {"_id":1      ,"name": "unknow",               "stamina": 1},
        {"_id":2      ,"name": "Figaro 3",             "stamina": 1},
        {"_id":3      ,"name": "Class 40 2021",        "stamina": 1},
        {"_id":4      ,"name": "Imoca",                "stamina": 1.2},
        {"_id":5      ,"name": "Mini 6.50",            "stamina": 1},
        {"_id":6      ,"name": "Ultim (Solo)",         "stamina": 1.5},
        {"_id":7      ,"name": "Volvo 65",             "stamina": 1.2},
        {"_id":8      ,"name": "unknow",               "stamina": 1},
        {"_id":9      ,"name": "Ultim (Crew)",         "stamina": 1.5},
        {"_id":10     ,"name": "Olympus",              "stamina": 1.5},
        {"_id":11     ,"name": "Ocean 50 (Multi 50)",  "stamina": 1},
        {"_id":12     ,"name": "unknow",               "stamina": 1},
        {"_id":13     ,"name": "Caravelle",            "stamina": 2},
        {"_id":14     ,"name": "Super Maxi 100",       "stamina": 1.5},
        {"_id":15     ,"name": "unknow",               "stamina": 1},
        {"_id":16     ,"name": "Tara",                 "stamina": 2},
        {"_id":17     ,"name": "unknow",               "stamina": 1},
        {"_id":18     ,"name": "OffShore Racer",       "stamina": 1},
        {"_id":19     ,"name": "Mod70",                "stamina": 1.2},
        {"_id":20     ,"name": "Cruiser Racer",        "stamina": 1.2},
        {"_id":21     ,"name": "Ultim BP XI",          "stamina": 1.5},
      ]

# hostname = socket.gethostname()
#print(f"Nom de l'ordinateur : {hostname}\n")
hostname='linux3'

if hostname=='linux0' :         # sur ordi linux1  (serveur)
    basedirnpy          = '/home/jp/static/npy/'
    basedirGribs025     = '/home/jp/gribs/gribs025/'
    basedirGribsVR32    = '/home/jp/gribs/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribs/gribsgfs32/'
    staticbd            = '/home/jp/static/bd/basededonnees.db'
    staticCommandes     = '/home/jp/static/bd/commandes.db'


if hostname=='linux1' :          # sur ordi linux1  (2eme ordi)
    basedirnpy          = '/home/jp/static/npy/'
    basedirGribs025     = '/home/jp/gribs/gribs025/'
    basedirGribsVR32    = '/home/jp/gribslocaux/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribslocaux/gribsgfs32/'
    staticbd            = '/home/jp/staticLocal/bd/basededonnees.db'
    staticCommandes     = '/home/jp/staticLocal/bd/commandes.db'


if hostname=='linux3' :          # sur ordi linux2  (ordi blanc)
    basedirnpy          = '/home/jp/staticLocal/npy/'
    basedirGribs025     = '/home/jp/gribslocaux/gribs025/'
    basedirGribsVR32    = '/home/jp/gribslocaux/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribslocaux/gribsgfs32/'
    # staticbd            = '/home/jp/static/bd/basededonnees.db'
    basedirGribsECMWF   = "/home/jp/gribslocaux/ecmwf/"
    basedirECMWF        = '/home/jp/gribslocaux/ecmwf/'
    # staticCommandes     = '/home/jp/staticLocal/bd/commandes.db'


if hostname=='portable' :          # sur ordi portable  (3eme ordi)
    basedirnpy          = '/home/jp/static/npy/'
    basedirGribs025     = '/home/jp/gribslocaux/gribs025/'
    basedirGribsVR32    = '/home/jp/gribslocaux/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribslocaux/gribsgfs32/'
    staticbd            = '/home/jp/staticLocal/bd/basededonnees.db'
    staticCommandes     = '/home/jp/staticLocal/bd/commandes.db'






def dateheure(filename):
    '''retourne la date et heure du fichier grib a partir du nom'''
    ''' necessaire pour charger le NOAA'''
    tic=time.time()
    ticstruct = time.localtime()
    utc = time.gmtime()
    decalage = ticstruct[3] - utc[3]
    x     = filename.split('.')[0]
    x     = x.split('/')[-1]
    heure = x.split('-')[1]
    date  = (x.split('-')[0]).split('_')[1]
    year  = int(date[0:4])
    month = int(date[4:6])
    day   = int(date[6:8])
    tigt=datetime(year,month,day,int(heure),0, 0)
    tig=time.mktime(tigt.timetuple()) +decalage*3600 # en secondes UTC
    return date,heure,tig 



##############################################################################
########### Recherche dans les tables postgresql  #############################
###############################################################################


def rechercheTableCoursesActives(username):
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT  user_id,coursesactives
            FROM coursesactives
            WHERE username = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (username,))
        resultats = cursor.fetchone()
        return resultats if resultats else (None, None)
    finally:
        cursor.close()
        pg_pool.putconn(conn)



def rechercheTableRacesinfos():
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT racesinfos 
        FROM racesinfos 
        ORDER BY timestamp DESC 
        LIMIT 1
        """)
        resultat = cursor.fetchone()
        return resultat[0] if resultat else None
    finally:
        cursor.close()
        pg_pool.putconn(conn)



def rechercheTableBoatInfos(user_id, course):
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT boatinfos
            FROM boatinfos
            WHERE user_id = %s AND course = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id, course))
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        cursor.close()
        pg_pool.putconn(conn)





def rechercheTablePersonalInfos(user_id, course):

    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT personalinfos
            FROM personalinfos
            WHERE user_id = %s AND course = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id, course))
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        cursor.close()
        pg_pool.putconn(conn)



def rechercheTableLegInfos(course):
  
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT leginfos
            FROM leginfos
            WHERE course = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (course,))
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        cursor.close()
        pg_pool.putconn(conn)


def rechercheTableProgsvr(user_id, course):
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, progsvr
            FROM progsvr
            WHERE user_id = %s AND course = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id, course))
        result = cursor.fetchone()
        print("Résultat brut de la requête :", result)
        return (result[0], result[1]) if result else (None, None)
    finally:
        cursor.close()
        pg_pool.putconn(conn)


# def rechercheTableProgsvr(user_id, course):
#     '''recherche retournant 2 resultats timestamp en premier '''
#     conn = pg_pool.getconn()
#     try:
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT timestamp, progsvr
#             FROM progsvr
#             WHERE user_id = %s AND course = %s
#             ORDER BY timestamp DESC
#             LIMIT 1
#         """, (user_id, course))
#         result = cursor.fetchone()
#         return (result[0], result[1]) if result else (None, None)
#     finally:
#         cursor.close()
#         pg_pool.putconn(conn)


def rechercheTablePolaires(polar_id):
    '''recherche retournant 2 resultats timestamp en premier '''
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT updated,_id,polaires
            FROM polaires
            WHERE _id = %s
            ORDER BY updated DESC
            LIMIT 1
        """, (polar_id,))
        result = cursor.fetchone()
       
        return (result[0],result[1] ,result[2]) if result else (None, None,None)
    finally:
        cursor.close()
        pg_pool.putconn(conn)



def enregistrerPolaireSiPlusRecent(timestamp, _id, message):
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        # Vérifier s'il existe déjà une entrée pour cet _id
        cursor.execute("""
            SELECT updated
            FROM polaires
            WHERE _id = %s
            ORDER BY updated DESC
            LIMIT 1
        """, (_id,))
        result = cursor.fetchone()
        
        # Enregistrer si pas d'entrée ou si la nouvelle est plus récente
        if result is None or timestamp > result[0]:
            print ('Les polaires ont besoin d etre mises a jour pour l id :',_id)

            cursor.execute("""
                INSERT INTO polaires (updated, _id, polaires)
                VALUES (%s, %s, %s)
            """, (timestamp, _id, json.dumps(message)))
            conn.commit()
            return True  # Enregistrement effectué
        else:
            print ('Les polaires sont les plus recentes pour l id ',_id)
            return False  # Enregistrement ignoré, trop ancien
    finally:
        cursor.close()
        pg_pool.putconn(conn)





def rechercheboatinfos(username,course):
    '''Recuperation sur le serveur '''
    url = 'http://vrouteur.com/rechercheboatinfos?username='+username+'&course='+course
    response = requests.get(url)
    # print (response)
    if response.status_code == 200:
        None
        # print('reponse bien reçue')
        # print("Réponse reçue :", response.json())
    else:
        print("Erreur :", response.status_code)
    return response.json() 


def rechercheleginfos (course):
    '''Recuperation sur le serveur '''
    url = 'http://vrouteur.com/rechercheleginfos?course='+course
    response = requests.get(url)
    if response.status_code == 200:
        None
        # print('reponse bien reçue')
        # print("Réponse reçue :", response.json())
    else:
        print("Erreur :", response.status_code)
    return response.json() 


    

#######################################################################################
##   Fonctions de Wrap 
#######################################################################################

def lon_to_360(lon):
    return lon % 360

def wrap360(x):
    return x % 360.0

def wrap180(x):
    return ((x + 180.0) % 360.0) - 180.0

def dlon_short(x2, x1):
    return wrap180(x2 - x1)

def wrap360_t(x):
    return torch.remainder(x, 360.0)

def wrap180_t(x):
    return torch.remainder(x + 180.0, 360.0) - 180.0

def dlon_short_t(lon2, lon1):
    return wrap180_t(lon2 - lon1)



#######################################################################################
##   Fonctions d'affichage 
#######################################################################################

def format_duration(seconds):
    ''' affiche une duree en heures mn secondes'''
    
    seconds = int(seconds)
    j = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    parts = []
    if j > 0:
        parts.append(f"{j} j")
    if h > 0 or j > 0:
        parts.append(f"{h} h")
    if m > 0 or h > 0 or j > 0:
        parts.append(f"{m} mn")
    parts.append(f"{s} s")

    return " ".join(parts)

#######################################################################################
##   Calculs caps et TWA
#######################################################################################


def ftwato(HDG,TWD):
    '''Twa orientee pour des float ou torch mais pas np '''
    return ((TWD-HDG+540)%360)-180


def ftwao(HDG, TWD):
    """
    TWA orientée (True Wind Angle)
    Compatible numpy et torch.
    Retourne un objet du même type que l'entrée : torch numpy ou python 
    """
    # --- Cas torch.Tensor ---
    if isinstance(HDG, torch.Tensor) or isinstance(TWD, torch.Tensor):
        # conversion en tensor si besoin
        HDG = HDG if isinstance(HDG, torch.Tensor) else torch.tensor(HDG, dtype=torch.float32)
        TWD = TWD if isinstance(TWD, torch.Tensor) else torch.tensor(TWD, dtype=torch.float32)
        return torch.remainder(TWD - HDG + 540, 360) - 180

    # --- Cas numpy.ndarray ---
    elif isinstance(HDG, np.ndarray) or isinstance(TWD, np.ndarray):
        return np.mod(TWD - HDG + 540, 360) - 180

    # --- Cas scalaires Python ---
    else:
        return ((TWD - HDG + 540) % 360) - 180



def ftwaos(cap, dvent):
    '''twa orientee simple avec des valeurs float'''
    twa=(cap-dvent+360)%360
    if twa<180:
        twa=-twa
    else:
        twa=360-twa
    return twa   

def fcap(twao,twd):
    ''' retourne le cap en fonction de la twaorientée et de la direction du vent '''
    cap=(360+twd-twao)%360
    return cap


def fcapto(TWAO, TWD) :
    """
    Retourne le cap (HDG) en fonction de la TWA orientée et de la direction du vent (TWD).
    Version optimisée 100% PyTorch.
    Entrées et sorties : torch.Tensor (en degrés).
    """
    return torch.remainder(360 + TWD - TWAO, 360)


def dist_mn(lat1, lon1, lat2, lon2):
    """Distance approximative entre 2 points en milles nautiques.
    lat/lon en radians.
    """
    R_NM = 3440.065
    dlon = (lon2 - lon1 + torch.pi) % (2 * torch.pi) - torch.pi
    dlat = lat2 - lat1
    x = dlon * torch.cos((lat1 + lat2) / 2)
    return torch.sqrt(x * x + dlat * dlat) * R_NM



def dist_mn_vec(lat1, lon1, lat2, lon2):
    """
    Distance approximative entre 2 ensembles de points en milles nautiques.
    lat/lon en radians.
    Compatible scalaires, vecteurs et tenseurs broadcastables.
    Gère le passage 0/360 et ±180 via repli de dlon dans [-pi, pi].
    """
    R_NM = 3440.065

    lat1 = torch.as_tensor(lat1)
    lon1 = torch.as_tensor(lon1)
    lat2 = torch.as_tensor(lat2)
    lon2 = torch.as_tensor(lon2)

    dlon = lon2 - lon1
    dlon = (dlon + torch.pi) % (2.0 * torch.pi) - torch.pi

    dlat = lat2 - lat1
    x = dlon * torch.cos((lat1 + lat2) * 0.5)

    return torch.sqrt(x * x + dlat * dlat) * R_NM


    
@torch.no_grad()
def dist_mn_t(lat1, lon1, lat2, lon2):
    """
    Distance orthodromique entre deux points (milles nautiques)

    lat1, lon1, lat2, lon2 : torch.Tensor (deg)
    tolère longitudes 0..360 ou -180..180
    fonctionne GPU
    """

    lat1r = torch.deg2rad(lat1)
    lat2r = torch.deg2rad(lat2)

    dlat = lat2r - lat1r
    dlon = torch.deg2rad(wrap180_t(lon2 - lon1))

    a = (
        torch.sin(dlat * 0.5)**2 +
        torch.cos(lat1r) * torch.cos(lat2r) *
        torch.sin(dlon * 0.5)**2
    )

    a = torch.clamp(a, 0.0, 1.0)
    c = 2.0 * torch.asin(torch.sqrt(a))

    return R_NM * c

def calcul_caps(tabpoints):
    """
    tabpoints : ndarray (N, 2) -> [[lat1, lon1], [lat2, lon2], ...]
                longitudes supposées en degrés, typiquement 0..360
    Retourne un tableau (N,) des caps initiaux en degrés [0..360),
    la dernière valeur étant 0.
    """
    lat1 = np.radians(tabpoints[:-1, 0])
    lon1 = tabpoints[:-1, 1]

    lat2 = np.radians(tabpoints[1:, 0])
    lon2 = tabpoints[1:, 1]

    # delta longitude court en degrés dans [-180, 180]
    dlon_deg = ((lon2 - lon1 + 180.0) % 360.0) - 180.0
    dlon = np.radians(dlon_deg)

    x = np.sin(dlon) * np.cos(lat2)
    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    caps = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    caps = np.append(caps, 0.0)

    return caps




def calcul_cap_loxodromique(lat0, lon0, lat1, lon1):
    """
    Calcule le cap constant (loxodromique) entre deux points.
    Les latitudes et longitudes sont en degrés.
    """
    # Conversion en radians
    lat0, lon0, lat1, lon1 = map(math.radians, [lat0, lon0, lat1, lon1])

    delta_lon = lon1 - lon0
    delta_phi = math.log(math.tan(math.pi / 4 + lat1 / 2) / math.tan(math.pi / 4 + lat0 / 2))

    # Gestion du cas où delta_phi est proche de 0 (évite une division par zéro)
    if abs(delta_phi) < 1e-10:
        cap = math.atan2(delta_lon, math.cos(lat0))
    else:
        cap = math.atan2(delta_lon, delta_phi)

    # Conversion en degrés et normalisation entre 0 et 360
    cap = (math.degrees(cap) + 360) % 360
    return cap




def calcul_cap_loxodromique_tensor(lat0, lon0, lat1, lon1):
    """
    Calcule le cap constant (loxodromique) entre deux points avec torch.
    Les latitudes et longitudes sont en degrés (torch.tensor).
    """
    # Conversion en radians
    lat0 = torch.deg2rad(lat0)
    lon0 = torch.deg2rad(lon0)
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)

    delta_lon = lon1 - lon0
    delta_phi = torch.log(torch.tan(torch.pi / 4 + lat1 / 2) / torch.tan(torch.pi / 4 + lat0 / 2))

    # Gestion du cas où delta_phi est proche de 0 (évite une division par zéro)
    near_zero = torch.abs(delta_phi) < 1e-10
    cap = torch.where(
        near_zero,
        torch.atan2(delta_lon, torch.cos(lat0)),
        torch.atan2(delta_lon, delta_phi)
    )

    # Conversion en degrés et normalisation entre 0 et 360
    cap = (torch.rad2deg(cap) + 360) % 360
    return cap



def distance_haversine(lat0, lon0, lat1, lon1):
    """
    Calcule la distance entre deux points en utilisant la formule de Haversine.
    """
    R = 6371000  # Rayon de la Terre en mètres
    lat0, lon0, lat1, lon1 = map(math.radians, [lat0, lon0, lat1, lon1])
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    a = math.sin(dlat / 2) ** 2 + math.cos(lat0) * math.cos(lat1) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_m= R * c
    return distance_m / 1852  # Conversion en miles nautiques



def distance_haversine_torch(lat0, lon0, lat1, lon1):
    """
    Calcule la distance entre deux points en utilisant la formule de Haversine,
    compatible avec torch.tensor.
    """
    R = 6371000  # Rayon de la Terre en mètres

    # Conversion en radians
    lat0 = torch.deg2rad(lat0)
    lon0 = torch.deg2rad(lon0)
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)

    dlat = lat1 - lat0
    dlon = lon1 - lon0

    a = torch.sin(dlat / 2)**2 + torch.cos(lat0) * torch.cos(lat1) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    distance_m = R * c
    return distance_m / 1852  # Conversion en miles nautiques




##################################################################################################################
##############    Constitution du tableau des intervalles de temps     ###########################################
##################################################################################################################



def construire_dt(seuils, taille=1000, dtype=torch.float32):
    """
    Génère un tableau dt en float32 à partir d'une liste de seuils.
    
    seuils : liste de [nombre_d'éléments, valeur_dt]
             ex : [[180, 60], [12, 600], [48, 1800]]
    taille : taille totale du vecteur (ex: 1000)
    dtype  : type du tenseur PyTorch (par défaut: float32)
    """
    dt = torch.full((taille,), 3600.0, dtype=dtype, device='cuda')  # Valeur par défaut

    index = 0
    for n, valeur in seuils:
        fin = index + n
        if fin > taille:
            fin = taille
        dt[index:fin] = float(valeur)
        index = fin
        if index >= taille:
            break

    return dt
#######################################################################################
##   Penalites en temps 
#######################################################################################

def peno_np(lwtimer, hwtimer, Tws, Stamina):
    '''calcul de la penalite avec tws et stamina sous forme de np.array'''
    ''' fonction testée exacte ITYC'''
    Cstamina = 2 - 0.015 * Stamina
    # loi de transition (10 < TWS < 30)
    ftws = 50 - np.cos((Tws - 10) * math.pi / 20) * 50
    Peno_mid = (((hwtimer - lwtimer) * ftws / 100) + lwtimer) 
    #Plateaux
    Peno = np.where(Tws <= 10, lwtimer, np.where( Tws >= 30, hwtimer,Peno_mid ))* Cstamina
    return Peno

def peno_torch(lwtimer, hwtimer, Tws, Stamina):
    """
    Calcul de pénalité (Torch).
    - lwtimer, hwtimer : scalaires ou tensors broadcastables
    - Tws, Stamina     : tensors (ou scalaires) broadcastables
    Retour : tensor
    """
    # Convertit tout en tensors, en alignant dtype/device sur Tws si possible
    Tws_t = Tws if torch.is_tensor(Tws) else torch.as_tensor(Tws)
    ref = Tws_t

    lwt = lwtimer if torch.is_tensor(lwtimer) else torch.as_tensor(lwtimer, device=ref.device, dtype=ref.dtype)
    hwt = hwtimer if torch.is_tensor(hwtimer) else torch.as_tensor(hwtimer, device=ref.device, dtype=ref.dtype)
    St  = Stamina if torch.is_tensor(Stamina) else torch.as_tensor(Stamina, device=ref.device, dtype=ref.dtype)
    Tws_t = Tws_t.to(device=ref.device, dtype=ref.dtype)

    # Coef stamina
    Cstamina = 2 - 0.015 * St

    # Transition (10 < TWS < 30)
    pi = torch.pi if hasattr(torch, "pi") else torch.tensor(math.pi, device=ref.device, dtype=ref.dtype)
    ftws = 50 - torch.cos((Tws_t - 10) * pi / 20) * 50
    Peno_mid = ((hwt - lwt) * ftws / 100) + lwt

    # Plateaux
    Peno_base = torch.where(
        Tws_t <= 10, lwt,
        torch.where(Tws_t >= 30, hwt, Peno_mid)
    )

    return Peno_base * Cstamina
#######################################################################################
##   Penalites en Stamina
#######################################################################################
 
def calc_perte_stamina_np(tws, TackGybe, Chgt,coeffboat, MF = 0.8):
    """
    Calcule la perte énergétique selon les conditions de vent, les changements de manœuvre et les coefficients.
    
    Args:
        tws : True Wind Speed. TackGybe : Tableau des virements . Chgt : Tableau des changements de voile.
        coeffboat (float): Coefficient global du bateau.
        MF (float): Coefficient multiplicateur pour Chgt.
    
    Returns:
        ndarray: La perte calculée pour chaque entrée.
    """
    m = np.zeros_like(tws)
    p = np.zeros_like(tws)

    # Remplissage de m
    m[(tws > 10) & (tws <= 20)] = 0.2
    m[(tws > 20) & (tws < 30)] = 0.6

    # Remplissage de p
    p[tws <= 10] = 10
    p[(tws > 10) & (tws <= 20)] = 8
    p[tws >= 30] = 18

    # Calcul final de la perte
    perte = ((m * tws + p) * (TackGybe + 2 * Chgt * MF)) * coeffboat
    
    return perte



def calc_perte_stamina_to(tws: torch.Tensor, TackGybe: torch.Tensor, Chgt: torch.Tensor,
                coeffboat: float ,MF: float = 0.8) -> torch.Tensor:
    """
    Calcule la perte énergétique selon les conditions de vent, les changements de manœuvre et les coefficients.
    Args:
        tws (Tensor): True Wind Speed.
        TackGybe (Tensor): Nombre de virement ou empannage.
        Chgt (Tensor): Nombre de changements de voile ou autre.
        MF (float): Coefficient multiplicateur pour Chgt.
        coeffboat (float): Coefficient global du bateau.

    Returns:
        Tensor: La perte calculée pour chaque entrée.
    """
    m = torch.zeros_like(tws)
    p = torch.zeros_like(tws)

    # Remplissage de m
    m[(tws > 10) & (tws <= 20)] = 0.2
    m[(tws > 20) & (tws < 30)] = 0.6

    # Remplissage de p
    p[tws <= 10] = 10
    p[(tws > 10) & (tws <= 20)] = 8
    p[tws >= 30] = 18

    # Calcul final de la perte
    perte = ((m * tws + p) * coeffboat*(TackGybe + 2 * Chgt * MF))  
    return perte



def frecupstaminato(dt: torch.Tensor, TWS: torch.Tensor, pouf=0.8) -> torch.Tensor:
    #dt  : tensor (durée, typiquement en minutes si c'est ton (L18-L17))
    #TWS : tensor (wind speed)
    #Retour : tensor des points récupérés sur l'intervalle dt.
    tws = TWS.clamp(0.0, 30.0)
    x = (1.0 + torch.cos(torch.pi * tws / 30.0)) * 0.5
    return ((0.10375 + 0.20875 * x.pow(2.15)) / 60.0) * dt * pouf

def frecupstamina_np(dt, TWS, pouf=0.8):
    """
    Version numpy optimisée.
    dt  : array ou scalaire (durée)
    TWS : array numpy (wind speed)
    retourne : array numpy
    """
    tws = np.clip(TWS, 0.0, 30.0)
    x = (1.0 + np.cos(np.pi * tws / 30.0)) * 0.5
    x **= 2.15
    return (0.10375 + 0.20875 * x) * (dt * pouf / 60.0)
    
#######################################################################################
##   Recuperation de la stamina
#######################################################################################
def _temps_pour_1pt_numpy(tws):
    tws = np.clip(tws, 0.0, 30.0)
    return 240.0 + 240.0 * (1.0 - np.cos(np.pi * tws / 30.0))

def _temps_pour_1pt_torch(tws):
    tws = torch.clamp(tws, 0.0, 30.0)
    return 240.0 + 240.0 * (1.0 - torch.cos(math.pi * tws / 30.0))


def temps_pour_1pt(tws):
    """
    tws : float | int | np.ndarray | torch.Tensor
    return : même type logique que l'entrée
    """
    if torch.is_tensor(tws):
        return _temps_pour_1pt_torch(tws)
    else:
        return _temps_pour_1pt_numpy(np.asarray(tws, dtype=float))


def pts_recuperes(dt, tws, pouf=0.8):
    """
    dt  : float | np.ndarray | torch.Tensor (secondes)
    tws : float | np.ndarray | torch.Tensor
    pouf : scalaire
    """
    if torch.is_tensor(tws) or torch.is_tensor(dt):
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, dtype=torch.float32, device=tws.device)
        return dt / (temps_pour_1pt(tws) * pouf)
    else:
        dt = np.asarray(dt, dtype=float)
        return dt / (temps_pour_1pt(tws) * pouf)






        
#######################################################################################
##   Fonctions de Routage 
#######################################################################################

def eta(dico):
    arrayroutage2 = dico['arrayroutage2']# on calcule l 'ETA 
    dernier = len(arrayroutage2) - 1
    ETA=(t0 + arrayroutage2[dernier][1]) 
    dureeEta = arrayroutage2[dernier][1]
    print('ETA :',time.strftime(" %d %b %H:%M ",time.localtime(ETA)))
    print('Durée',format_duration(dureeEta))
    return ETA,dureeEta
    



    
def lissagepoints_dt_twa(tabpoints, tabtwa, tabdt):
    tabres = np.copy(tabpoints)

    # --- 1. Ruptures de signe du TWA ---
    pos_twa = np.where(tabtwa[:-1] * tabtwa[1:] < 0)[0] + 1
    # print ( 'rupture twa ',pos_twa)
    
    # --- 2. Ruptures de pas de temps ---
    diffs = np.diff(tabdt)
    pos_dt = np.where(np.diff(diffs) != 0)[0] + 1
    # print ( 'rupture rythme ',pos_dt)
    # --- 3. Fusion et bornes ---
    pos_all = np.unique(np.concatenate(([0], pos_twa, pos_dt, [len(tabpoints)-1])))
    # print ( 'rupture twa ou rythme',pos_all)

    # --- 4. Lissage tronçon par tronçon ---
    for i in range(len(pos_all) - 1):
        i0, i1 = pos_all[i], pos_all[i+1]
        segment = tabpoints[i0:i1+1]

        if len(segment) > 6:
            y = segment[:, 0]
            x = segment[:, 1]
            Y = savgol_filter(y, 5, 2, mode='nearest')
            X = savgol_filter(x, 5, 2, mode='nearest')
        else:
            Y = segment[:, 0]
            X = segment[:, 1]

        # On conserve les extrémités intactes
        tabres[i0+1:i1, 0] = Y[1:-1]
        tabres[i0+1:i1, 1] = X[1:-1]
    return tabres
def smooth(tab, tol=1, window=1):
    """
    Lissage discret : si un point diffère légèrement (≤ tol)
    de ses voisins identiques, il est remplacé par leur valeur.

    tab : tableau numpy (ou liste)
    tol : écart maximal considéré comme aberration isolée (défaut 1)
    window : nombre de voisins de chaque côté (défaut 1)
    """
    tab = np.asarray(tab, dtype=float).copy()
    n = len(tab)
    if n < 3 or window < 1:
        return tab

    for i in range(window, n - window):
        gauche = tab[i - window]
        droite = tab[i + window]
        if abs(tab[i] - gauche) <= tol and abs(tab[i] - droite) <= tol and abs(gauche - droite) <= tol:
            # voisins quasi identiques → remplacement
            tab[i] = gauche # ou simplement gauche, car proches
    return tab
    



def smoothTo(t):
    """
    Smooth un tensor 1D en remplaçant toute valeur qui est
    différente de ses voisins mais entourée de deux valeurs identiques.
    """
    t = t.clone()  # pour ne pas modifier l'original

    # On compare les décalages
    left  = t[:-2]    # tous sauf les 2 derniers
    mid   = t[1:-1]   # tous sauf les extrémités
    right = t[2:]     # tous sauf les 2 premiers

    mask = (left == right) & (mid != left)   # condition de correction

    # on corrige uniquement au milieu
    t[1:-1][mask] = left[mask]

    return t    

#######################################################################################
##   Fonctions de cartographie 
#######################################################################################

filename='/home/jp/staticLocal/cumsum/sequenceglobale.pt'
sequences2 = torch.load(filename,map_location="cuda:0")                     # sequences2 fait 87 Mo imp
cumsumx = torch.cumsum(sequences2, dim=0)    
cache_cartes = TTLCache(maxsize=1000, ttl=5 * 24 * 60 * 60)
cache_lock = Lock()


# def get_carte(lat,lon, zoom_x=3, zoom_y=4):
#     '''Cherche si la carte a deja ete chargee dans le cache '''
#     ''' La charge depuis la carte ou depuis les fichiers  '''
#     lat, lon = int(lat, int(lon))


#     cle = (lat, lon)
    
    
#     with cache_lock:
#         if cle in cache_cartes:
#             return cache_cartes[cle]
#     # Si non trouvé (ou expiré), on recharge la carte
#     carte = fcarte(lat, lon, zoom_x, zoom_y)
#     with cache_lock:
#         cache_cartes[cle] = carte
#     return carte



def get_carte3 (lat, lon):
    #la cle est un multiple de 10
    cle = (lat, lon)
    with cache_lock:
        if cle in cache_cartes:
            return cache_cartes[cle]

    # Calcul ou chargement de la carte (taille fixe de 10x10)
    carte = fcarte3(lat, lon)       # dans fonctions2025

    with cache_lock:
        cache_cartes[cle] = carte

    return carte



def fcarte3(lat,lon):
    '''Charge la carte des offsets directement  
     transforme en coordonnees et renvoie le tableau de coordonnees'''   

    lat0=int(10*(lat//10) +10)
    lon0=int(10*(lon//10))
    # print (lat0,lon0)
    filename='maps2/carteoffset_'+str(lat0)+'_'+str(lon0)+'.npy'
    print('**********************************************')
    print ('filename ',filename)


    # filename='carteoffset_'+str(lat0)+'_'+str(lon0)+'.npy'
    # on recharge le numpy pour voir que la sauvegarde est correcte
    offsets = np.load(filename)                                # matrice des offsets

    pas = 1/730                                              # un pas = 1/730 °
    # on va transformer le fichier numpy d'offsets en coordonnees 
    coords = np.empty_like(offsets, dtype=np.float32)
    coords[..., 0] = lat0 - offsets[..., 0].astype(np.float32) * pas    # latitude diminue vers le sud
    coords[..., 1] = lon0 + offsets[..., 1].astype(np.float32) * pas    # longitude augmente vers l’est
    return coords 



def assemble_masques(lat_min, lat_max, lon_min, lon_max):
    ''' Assemble les masques entre les lats et lons min et max'''

    latitudes = range(math.floor(lat_max), math.ceil(lat_min), -1)
    longitudes = range(math.floor(lon_min), math.ceil(lon_max))  
    masques = []
    for lat in latitudes:
        ligne = []
        for lon in longitudes:
            try:
                mask = chargeMasque(lat, lon)
            except Exception as e:
                print(f"Erreur en chargeant masque {lat},{lon} : {e}")
                mask = np.zeros((730,730), dtype=np.uint8)
            ligne.append(mask)
        masques.append(np.concatenate(ligne, axis=1))
    masque_global = np.concatenate(masques, axis=0)
    
    return masque_global, latitudes[0], longitudes[0]


def contour_terre3(mask_np, lat0, lon0, resolution_lat, resolution_lon):
    '''Transforme les masques en polylignes '''
    mask = torch.tensor(mask_np, dtype=torch.uint8, device='cuda')
    padded = torch.nn.functional.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    top = padded[:-2, 1:-1]
    bottom = padded[2:, 1:-1]
    center = padded[1:-1, 1:-1]

    contours = []
    H, W = mask_np.shape
    dlat = resolution_lat / H
    dlon = resolution_lon / W

    # Bord gauche
    ys, xs = torch.nonzero((center != left), as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        lat1 = lat0 - y * dlat
        lon1 = lon0 + x * dlon
        contours.append([(lat1, lon1), (lat1 - dlat, lon1)])

    # Bord droit
    ys, xs = torch.nonzero((center != right), as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        lat1 = lat0 - y * dlat
        lon1 = lon0 + (x + 1) * dlon
        contours.append([(lat1, lon1), (lat1 - dlat, lon1)])

    # Bord haut
    ys, xs = torch.nonzero((center != top), as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        lat1 = lat0 - y * dlat
        lon1 = lon0 + x * dlon
        contours.append([(lat1, lon1), (lat1, lon1 + dlon)])

    # Bord bas
    ys, xs = torch.nonzero((center != bottom), as_tuple=True)
    for y, x in zip(ys.tolist(), xs.tolist()):
        lat1 = lat0 - (y + 1) * dlat
        lon1 = lon0 + x * dlon
        contours.append([(lat1, lon1), (lat1, lon1 + dlon)])

    return contours



def chaine_segments(segments):
    # Indexation des points : point → liste des segments connectés
    point_to_segments = defaultdict(list)
    for seg in segments:
        p1, p2 = tuple(seg[0]), tuple(seg[1])
        point_to_segments[p1].append((p1, p2))
        point_to_segments[p2].append((p2, p1))  # ajout dans l’autre sens aussi

    used = set()
    polylines = []

    def find_chain(start):
        chain = deque()
        chain.append(start)
        current = start
        while True:
            found = False
            for seg in point_to_segments[current]:
                if seg not in used:
                    used.add(seg)
                    used.add((seg[1], seg[0]))  # marquer les deux directions
                    chain.append(seg[1])
                    current = seg[1]
                    found = True
                    break
            if not found:
                break
        return list(chain)

    # Boucle sur tous les points de départ possibles
    for seg in segments:
        p1 = tuple(seg[0])
        p2 = tuple(seg[1])
        if (p1, p2) not in used and (p2, p1) not in used:
            used.add((p1, p2))
            used.add((p2, p1))
            poly = find_chain(p2)
            poly.insert(0, p1)
            polylines.append(poly)

    return polylines
    
#on transforme la carte en un tableau numpy
def segments_to_array(carte):
    return np.array([[seg[0][0], seg[0][1], seg[1][0], seg[1][1]] for seg in carte], dtype=np.float32)


def chargeMasque (y0,x0):
    resolution=1
    lat=math.ceil(y0)
    lng=math.floor(x0)
    pas=730
    file=str(resolution)+'_'+str(lng)+'_'+str(lat)+'.deg'
    # print(basedir)
    filename='/home/jp/staticLocal/carto/'+file
    with open(filename,'rb') as fid:
            header=fid.read(11)
            gzbuf=fid.read()
            databuf=zlib.decompress(gzbuf,-zlib.MAX_WBITS)
    data=np.frombuffer(databuf,dtype=np.int8)
    data.resize((730,730))
    mask=   ( data > -1) 
    return mask 


# def assemble_masques(lat_min, lat_max, lon_min, lon_max):
#     latitudes = range(math.floor(lat_max), math.ceil(lat_min), -1)
#     longitudes = range(math.floor(lon_min), math.ceil(lon_max))
    
#     masques = []
#     for lat in latitudes:
#         ligne = []
#         for lon in longitudes:
#             try:
#                 mask = chargeMasque(lat, lon)
#             except Exception as e:
#                 print(f"Erreur en chargeant masque {lat},{lon} : {e}")
#                 mask = np.zeros((730,730), dtype=np.uint8)
#             ligne.append(mask)
#         masques.append(np.concatenate(ligne, axis=1))
#     masque_global = np.concatenate(masques, axis=0)
    
#     return masque_global, latitudes[0], longitudes[0]


def prepare_segments(polygons):
    segments = []
    for poly in polygons.values():
        if poly.numel() == 0:
            continue  # ignore les polygones vides
        if not torch.all(poly[0] == poly[-1]):
            poly = torch.cat([poly, poly[0:1]], dim=0)
        segs = torch.stack([poly[:-1], poly[1:]], dim=1).reshape(-1, 4)  # [x1, y1, x2, y2]
        segments.append(segs)

    if segments:
        return torch.cat(segments, dim=0)
    else:
        return torch.empty((0, 4), dtype=torch.float32)


def terremer(lat,lon):
    ''' lat lon sont des tenseurs '''
    ''' Necessite que cumsumx soit charge'''
    ''' retourne 1 si mer 0 si terre  '''
    p=180*730
    lon = (lon + 360) % 360
    indicelat = torch.floor(730 * (90 - lat)).to(torch.int64)
    indicelon = torch.ceil(730 * lon).to(torch.int64)-1
    indice = indicelat + p * indicelon # Indice unique
    idx = torch.searchsorted(cumsumx, indice, right=True) - 1
    return idx%2
 
   
def decoupe_latlon(lat_lon, seuil=0.01):
   
    ''' permet de decouper les isochrones a la rencontre avec les terres ,plus le seuil est petit , plus on coupe au niveau des terres'''
    # Calcul des écarts successifs
    diffs = torch.abs(lat_lon[1:] - lat_lon[:-1])
    sauts = (diffs > seuil).any(dim=1).nonzero(as_tuple=False).flatten()

    # Indices de découpe
    split_indices = (sauts + 1).tolist()
    split_indices = [0] + split_indices + [lat_lon.size(0)]

    # Extraction + conversion en list of lists
    sub_arrays = [
        lat_lon[split_indices[i]:split_indices[i+1]].tolist()
        for i in range(len(split_indices) - 1)
    ]

    return sub_arrays


def construit_dico_isochrones_1sur6(isoglobal: torch.Tensor, seuil=0.01):
    dico_isochrones = {}
    if isoglobal.shape[0] == 0:
        return dico_isochrones

    iso_col = isoglobal[:, 0]
    idx_change = torch.where(iso_col[1:] != iso_col[:-1])[0] + 1

    bornes = torch.cat([
        torch.tensor([0], device=isoglobal.device),
        idx_change,
        torch.tensor([isoglobal.shape[0]], device=isoglobal.device)
    ])

    for k in range(len(bornes) - 1):
        start = int(bornes[k].item())
        end = int(bornes[k + 1].item())

        bloc = isoglobal[start:end]
        iso = int(bloc[0, 0].item())

        if iso % 6 != 0:
            continue

        lat_lon = bloc[:, [3, 4]]
        segs = decoupe_latlon_folium(lat_lon, seuil=seuil)

        if segs:
            dico_isochrones[iso] = segs

    return dico_isochrones




def decoupe_latlon_folium(points: torch.Tensor, seuil=0.01):
    """
    Retourne uniquement les segments de longueur >= 2,
    directement prêts pour folium.PolyLine(...)
    """
    n = points.shape[0]
    if n < 2:
        return []

    diffs = torch.abs(points[1:] - points[:-1])
    sauts = (diffs > seuil).any(dim=1)
    coupures = (torch.nonzero(sauts, as_tuple=False).flatten() + 1).tolist()

    res = []
    start = 0

    for end in coupures:
        if end - start >= 2:
            res.append(points[start:end].detach().cpu().tolist())
        start = end

    if n - start >= 2:
        res.append(points[start:n].detach().cpu().tolist())

    return res



def decoupe_latlon_fast(lat_lon, seuil=0.01):
    n = lat_lon.size(0)
    if n == 0:
        return []
    if n == 1:
        return [lat_lon.tolist()]

    # écart max entre deux points consécutifs sur lat/lon
    diffs = (lat_lon[1:] - lat_lon[:-1]).abs()
    sauts = torch.nonzero(diffs.amax(dim=1) > seuil, as_tuple=False).flatten()

    if sauts.numel() == 0:
        return [lat_lon.tolist()]

    split_indices = torch.cat([
        torch.tensor([0], device=lat_lon.device),
        sauts + 1,
        torch.tensor([n], device=lat_lon.device)
    ])

    return [
        lat_lon[int(split_indices[i].item()):int(split_indices[i+1].item())].tolist()
        for i in range(split_indices.numel() - 1)
    ]




def ajouter_croix(m, lat, lon, taille, couleur='red', tooltip=None, popup=None):
    # Croix formée de deux segments
    m.add_child(folium.PolyLine(
        locations=[[lat - taille, lon], [lat + taille, lon]],
        color=couleur, weight=2
    ))

    m.add_child(folium.PolyLine(
        locations=[[lat, lon - taille], [lat, lon + taille]],
        color=couleur, weight=2
    ))

    # Marqueur invisible au centre pour afficher le tooltip et/ou popup
    if tooltip or popup:
        folium.CircleMarker(
            location=[lat, lon],
            radius=1,  # très petit marqueur
            color='transparent',
            fill=True,
            fill_opacity=0.0,
            tooltip=tooltip,
            popup=popup
        ).add_to(m)
    



def points_in_any_polygon_vectorized(lat, lon, segments):
    # lat, lon: [N] (1D tensors)
    # segments: [S, 4] where each row is [x1, y1, x2, y2] i.e. [lon1, lat1, lon2, lat2]

    # reshape en colonnes pour broadcasting
    x = lon.view(-1, 1)  # [N, 1]
    y = lat.view(-1, 1)  # [N, 1]

    x1 = segments[:, 0].view(1, -1)  # [1, S]
    y1 = segments[:, 1].view(1, -1)
    x2 = segments[:, 2].view(1, -1)
    y2 = segments[:, 3].view(1, -1)

    # condition 1 : le rayon croise verticalement le segment
    cond1 = ((y1 > y) != (y2 > y))  # shape [N, S]

    # calcul de l'intersection horizontale
    slope = (x2 - x1) / (y2 - y1 + 1e-10)
    xinters = x1 + (y - y1) * slope  # shape [N, S]

    # condition 2 : le point est avant l’intersection
    cond2 = x < xinters

    # on compte les croisements et on applique la règle impair/pair
    mask = torch.sum((cond1 & cond2), dim=1) % 2 == 1  # [N]
    return mask





def detect_barrier_crossings(lat0, lon0, lat1, lon1, barriers):
    p = torch.stack((lon0, lat0), dim=1)  # [N, 2]
    r = torch.stack((lon1 - lon0, lat1 - lat0), dim=1)  # [N, 2]
    
    def cross(a, b):
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    
    intersects = torch.zeros(lat0.shape[0], dtype=torch.bool, device=lat0.device)

    for key, barrier in barriers.items():
        q = barrier[0][[1, 0]]  # [lon, lat]
        s = barrier[1][[1, 0]] - q  # direction vector

        q = q[None, :]  # [1, 2]
        s = s[None, :].expand_as(p)  # [N, 2]

        qp = q - p
        cross_r_s = cross(r, s)
        cross_qp_r = cross(qp, r)
        cross_qp_s = cross(qp, s)

        eps = 1e-10
        parallel = torch.abs(cross_r_s) < eps
        t = cross_qp_s / (cross_r_s + eps)
        u = cross_qp_r / (cross_r_s + eps)

        current_intersect = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        intersects |= current_intersect  # union logique

    return intersects  # [N] 

#######################################################################################
##   Fonctions d'affichage et d 'impression 
#######################################################################################

def format_time(ts):
    # conversion depuis timestamp Unix
    return time.strftime(" %d %b %H:%M ",time.localtime(ts))
 

def decaler_colonnes(tenseur, colonnes=[5, 6, 7, 8]):
    for col in colonnes:
        tenseur[:-1, col] = tenseur[1:, col]  # décalage vers le haut
        tenseur[-1, col] = 0                  # dernière ligne à zéro
    return tenseur



def impression_tensor15 (tab,titre='Calculs'):
    typeVoiles = ['jib', 'Spi', 'Staysail', 'LightJib', 'Code0', 'HeavyGnk', 'LightGnk']
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print (' 0:niso   1:i   2:PtMe   3:lat    4:lon      5:Voi  6:twam1 7:stam  8:Speno  9:tws  10:twd   11:tws10  12:Cap   13:CapR  14:twa   15:VVini 16:VVmax 17:BestV 18:Boost  19:voiledef    ')
    print ('_______________________________________________________________________________________________________________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f} {:8.0f} {:6.0f}   {:8.5f} {:8.5f}    {}   {:8.2f}  {:6.2f}  {:6.2f} {:6.2f} {:6.1f} {:8.2f} {:8.2f} {:8.3f}    {:8.1f}    {:6.2f}  {:6.2f}  {:6.2f}  {:10.3f}  {:6.2f}'.\
          format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),typeVoiles[int(tab[ i,5].item())],tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item()\
        ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),tab[ i,13].item(),tab[ i,14].item() ,tab[ i,15].item(),tab[ i,16].item(),tab[ i,17].item(),tab[ i,18].item(),tab[ i,19].item()))


def impression_routage (tab,t0,titre='Routage '):
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print (' 0:niso  1heure     2:lat   3:lon        4:date    5:twa     6:cap    7:vmgmin   8:vmgmax   9:speed   10:voile 11:boost  12:tws 13:twd    14:stamina  15:peno      ')
    print ('____________________________________________________________________________________________________________________________________________________________')
    print ( ) 
    # i=0
    # print(   'Etat VR: {:4.0f}     {:8.5f}  {:8.5f} {} {:7.2f}   {:8.2f}  {:6.2f}  {:8.2f}   {:8.2f}  {:6.1f}    {:8.3f} {:8.2f} {:8.2f}   {:8.1f}    {:6.2f}     '.\
    #         format( 0, tab[ 0,2].item(),tab[ 0,3].item(),time.strftime(" %d %b %H:%M ",time.localtime(t0)),tab[ i,5].item(),tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item()\
    #     ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),0,tab[ i,14].item() ,tab[ i,15].item()))
    # print ('____________________________________________________________________________________________________________________________________________________________')
          
    for i in range (len(tab)):
        print('{:4.0f} {:8.0f}     {:8.5f}  {:8.5f} {} {:7.0f}   {:8.0f}  {:6.2f}  {:8.2f}   {:8.2f}  {:6.1f} || {:8.3f} {:8.2f} {:8.2f}    {:8.1f}   {:6.2f}  '.\
          format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),time.strftime(" %d %b %H:%M ",time.localtime(tab[ i,1].item()+t0)),tab[ i,5].item(),tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item()\
        ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),tab[ i,13].item(),tab[ i,14].item() ,tab[ i,15].item()  ))

        