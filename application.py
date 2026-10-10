import csv
import json
import math
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import socket
import sys
import time
import copy
import ast
import urllib.parse  # pour decoder url
from   urllib.parse import unquote
import uuid
import webbrowser
import sqlite3 as sql
import psycopg2
from   psycopg2 import pool
import folium
import numpy as np
import torch
import requests
import xarray as xr
import copy
import logging

import gc

from datetime import datetime,timezone,timedelta
from matplotlib.path import Path
from flask import (Flask, flash, jsonify, redirect, render_template,make_response, request,session, url_for)
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_caching import Cache
from scipy.ndimage import label, generate_binary_structure

# from numba import njit
# from numba import prange
from pathlib import Path
from flask_cors import CORS
# from global_land_mask import globe
from shapely.geometry import (LineString, MultiLineString, MultiPoint, Point,Polygon)
from shapely.strtree import STRtree
from apscheduler.schedulers.background import BackgroundScheduler
from websocket import create_connection
from scipy.signal import savgol_filter
from cachetools import TTLCache
from threading import Lock
from collections import defaultdict, OrderedDict, deque     # necessaire pour la carte 


from fonctions2026 import *


Pi=math.pi
R_NM = 3440.065  # rayon moyen de la Terre en milles nautiques
KNOTS = 1.94384
device='cuda'

basedirGribs025 = '/home/jp/gribslocaux/gribs025/'
fileMeta        = basedirGribs025+'meta.txt'
META_URL        = "https://vro-api-client.prod.virtualregatta.com/dl?t=fwr"
OUTDIR          = "/home/jp/gribslocaux/vr"
TMPDIR          = os.path.join(OUTDIR, "tmp")
NY, NX          = 721, 1440
NMAX            = 128
GRGFS = None
tigGFS = None
indiceGR = -1
fileName=None





torch.cuda.empty_cache()
torch.cuda.synchronize()

hostname = socket.gethostname()
print(f"Nom de l'ordinateur : {hostname}\n")

if hostname=='linux0' :         # sur ordi linux1  (serveur)
    basedirnpy          = '/home/jp/static/npy/'
    basedirGribs025     = '/home/jp/gribs/gribs025/'
    basedirECMWF        = '/home/jp/gribs/ecmwf/'
    basedirGribsVR32    = '/home/jp/gribs/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribs/gribsgfs32/'
    staticbd            = '/home/jp/static/bd/basededonnees.db'
    staticCommandes     = '/home/jp/static/bd/commandes.db'


if hostname=='linux1' :          # sur ordi linux1  (2eme ordi)
    basedirnpy          = '/home/jp/static/npy/'
    basedirGribs025     = '/home/jp/gribslocaux/gribs025/'
    basedirECMWF        = '/home/jp/gribslocaux/ecmwf/'
    basedirGribsECMWF   = "/home/jp/gribslocaux/ecmwf/"
    basedirGribsVR32    = '/home/jp/gribslocaux/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribslocaux/gribsgfs32/'
    staticbd            = '/home/jp/staticLocal/bd/basededonnees.db'
    staticCommandes     = '/home/jp/staticLocal/bd/commandes.db'


if hostname=='linux3' :          # sur ordi linux1  (2eme ordi)
    basedirnpy          = '/home/jp/staticLocal/npy/'
    basedirGribs025     = '/home/jp/gribslocaux/gribs025/'
    basedirECMWF        = '/home/jp/gribslocaux/ecmwf/'
    basedirGribsECMWF   = "/home/jp/gribslocaux/ecmwf/"
    basedirGribsVR32    = '/home/jp/gribslocaux/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribslocaux/gribsgfs32/'
    staticbd            = '/home/jp/static/bd/basededonnees.db'      # plus d actualite avec mabase postgre 
    staticCommandes     = '/home/jp/static/bd/commandes.db'         # plus d actualite avec mabase postgre


if hostname=='portable' :          # sur ordi portable  (3eme ordi)
    basedirnpy          = '/home/jp/static/npy/'
    basedirGribs025     = '/home/jp/gribslocaux/gribs025/'
    basedirGribsVR32    = '/home/jp/gribslocaux/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribslocaux/gribsgfs32/'
    staticbd            = '/home/jp/staticLocal/bd/basededonnees.db'
    staticCommandes     = '/home/jp/staticLocal/bd/commandes.db'


# for debug cuda 
# print()
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.version.git_version)
# print(torch.cuda.is_available())
from scheduler_cleaner import start_scheduler
start_scheduler()   # Lancer le scheduler une seule fois au démarrage

app         = Flask(__name__)
cors        = CORS(app)                             # necessaire pour les appels crossorigin a voir si pas en doublon avec ligne suivante
CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, resources={r"/vrouteur.com/": {"origins": "*"}})
bloqueroutage=0

logging.basicConfig()
logging.getLogger('psycopg2').setLevel(logging.DEBUG)                         # pour la base de donnees postgresql



################################################################################################
################      Table des matieres       #################################################
################################################################################################


# 167 Gestion du websocket 
# 243 Gestion du grib
# 337 Interpolateurs
# 817 Cache pour les fichiers 
# 848 Cache pour les cartes 
# 913 Base de donnees et tables 
# 380 class CpuGpuCache:


socketioOn=True
# if socketioOn:
#socketio = SocketIO(app, logger=True, engineio_logger=True)
IS_PRODUCTION = os.environ.get("FLASK_ENV") == "production"
hostname = socket.gethostname()


ECM_actif=True

print ('is PRODUCTION',IS_PRODUCTION)
print("ENV =", os.getenv("VROUTEUR_ENV"))
################################################################################################
# 167 Gestion du websocket     #################################################
################################################################################################


socketio = SocketIO(app, async_mode="threading", logger=False, engineio_logger=False)

# socketio = SocketIO(app, cors_allowed_origins="*",logger=True, engineio_logger=True,async_mode='threading')   # ou 'gevent'
clients = {}                                            # dictionnaire gardant les clients en memoire 
users   = {}

@socketio.on('connect')
def handle_connect():
    print(f"Nouveau client connecté (SID: {request.sid})")

@socketio.on('disconnect')
def handle_disconnect():
    # Nettoyer si le client était enregistré
    for client_id, sid in list(clients.items()):
        if sid == request.sid:
            leave_room(client_id)
            del clients[client_id]
            print(f"Client {client_id} déconnecté")
            break


@socketio.on('register')
def handle_register(data):
    client_id = data.get('client_id')
    current_sid = request.sid

    if not client_id:
        emit('registered', {'status': 'error', 'message': 'client_id manquant'})
        return

    old_sid = clients.get(client_id)
    if old_sid and old_sid != current_sid:
        print(f"Client {client_id} déjà enregistré avec une autre session. Mise à jour...")
        leave_room(client_id, sid=old_sid)

    clients[client_id] = current_sid
    join_room(client_id)

    print(f"Client {client_id} enregistré avec SID {current_sid}")
    emit('registered', {'status': 'ok', 'client_id': client_id})


@socketio.on('request_test_update')
def handle_test(data):
    client_id = data.get('client_id')
    send_update_to_client(client_id, f"Hello {client_id}, test OK")



    # Fonction pour envoyer une mise à jour personnalisée
def send_update_to_client(client_id, message):
    print('ligne 154 clients connectes : ',clients)
    print( 'message envoye au client  ',message)
    if client_id in clients:
        socketio.emit('update', {'message': message}, room=client_id, namespace='/')
    else:
        print(f"\nClient {client_id} non connecté\n")    


#exemple 
#send_update_to_client( "user1234", 'Coucou 1234 ')


################################################################################################
###############      Gestion du grib          #################################################
#################################################################################################


#________________________________________________________________________
#  Chargement GFS
#________________________________________________________________________

GRGFS = None
GR_t = None
tigGFS = None
indiceGR = -1
fileName = None

def charge_grib(basedirGribs025):
    global GRGFS, GR_t, tigGFS, indiceGR, fileName

    fileName, tig_file, fileNameP, tigP = gribFileName(basedirGribs025)


    GRGFS, tig_loaded, indice_loaded = load_npz(fileName)
    GR_t = torch.as_tensor(GRGFS, device=device, dtype=torch.float32)

    tigGFS = tig_loaded
    indiceGR = indice_loaded
    print ('*****************************************************************************')
    print ('*****************************************************************************')
    print("Grib chargé en base :", fileName, "tigP", tigP,"soit ------", time.strftime(" %d %b %H:%M ",time.gmtime(tigP))," -----indice", indiceGR)
    print("Grib chargé en base :", fileName, "tig", tigGFS,"soit ------", time.strftime(" %d %b %H:%M ",time.gmtime(tigGFS))," -----indice", indiceGR)

    print ('*****************************************************************************')
    print ('*****************************************************************************')



def majgrib(basedirGribs025):
    global GRGFS, GR_t, tigGFS, indiceGR, fileName

    newFileName, tig_file, fileNameP, tigP = gribFileName(basedirGribs025)
    tigmeta, indicemeta = load_meta(fileMeta)

    # changement de run
    if tigGFS is None or tigmeta > tigGFS:
        GRGFS, tig_loaded, indice_loaded = load_npz(newFileName)
        GR_t = torch.as_tensor(GRGFS, device=device, dtype=torch.float32)

        fileName = newFileName
        tigGFS = tig_loaded
        indiceGR = indice_loaded
        print ('*****************************************************************************')
        print ('*****************************************************************************')
        print("Nouveau grib chargé :", fileName, "tigGFS", tigGFS, "indice", indiceGR)
        return

    # même run, mais plus d'indices chargés
    if tigmeta == tigGFS and indicemeta > indiceGR:
        GRGFS, tig_loaded, indice_loaded = load_npz(newFileName)
        GR_t = torch.as_tensor(GRGFS, device=device, dtype=torch.float32)

        fileName = newFileName
        tigGFS   = tig_loaded
        indiceGR = indice_loaded
        print ('*****************************************************************************')
        print ('*****************************************************************************')
        print("Grib mis à jour par majgrib :", fileName, "tig", tigGFS, "soit *****", time.strftime(" %d %b %H:%M ",time.gmtime(tigGFS))," ***** indice", indiceGR)
        return

    print("Grib déjà à jour")





def gribFileName(basedir):
    """Cherche le dernier grib complet disponible et le précédent."""
    temps_secondes = time.time()
    date_tuple = time.gmtime(temps_secondes)
    date_formatcourt = time.strftime("%Y%m%d", date_tuple)
    dateveille_formatcourt = time.strftime("%Y%m%d", time.gmtime(temps_secondes - 86400) )
    mn_jour_utc = date_tuple.tm_hour * 60 + date_tuple.tm_min

    if mn_jour_utc < 3*60 + 40:
        filename = basedir + "gfsvr_" + dateveille_formatcourt + "-18.npz"
    elif mn_jour_utc < 9*60 + 40:
        filename = basedir + "gfsvr_" + date_formatcourt + "-00.npz"
    elif mn_jour_utc < 15*60 + 40:
        filename = basedir + "gfsvr_" + date_formatcourt + "-06.npz"
    elif mn_jour_utc < 21*60 + 40:
        filename = basedir + "gfsvr_" + date_formatcourt + "-12.npz"
    else:
        filename = basedir + "gfsvr_" + date_formatcourt + "-18.npz"
    tig = dateheure(filename)[2]
    tigP = tig - 6 * 3600
    fileNameP = basedir + "gfsvr_" + time.strftime("%Y%m%d-%H", time.gmtime(tigP) ) + ".npz"
    return filename, tig, fileNameP, tigP


charge_grib(basedirGribs025)
majgrib(basedirGribs025)



#################################################################################################
###############      Interpolateur              #################################################
#################################################################################################



def gfs_interpolateur_np(dtig, lat, lon):
    ''' fonction d'interpolation '''
    global GRGFS
    nt = GRGFS.shape[0]
    # print()
    # print ('On est dans gfs interpolateur_np')
    # --- temps ---
    h = (dtig-60) / 3600.0
    ft = h / 3.0

    k0 = int(np.floor(ft))
    k0 = max(0, min(k0, nt - 2))
    k1 = k0 + 1

    wt = np.float32(ft - k0)

    # --- indices espace ---
    lon360 = lon % 360.0

    fi = (90.0 - lat) / 0.25
    fj = lon360 / 0.25

    i0 = int(np.floor(fi))
    i0 = max(0, min(i0, 719))
    i1 = i0 + 1

    jfloor = np.floor(fj)
    j0 = int(jfloor) % 1440
    j1 = (j0 + 1) % 1440

    wi = np.float32(fi - i0)
    wj = np.float32(fj - jfloor)
    
    # print ('dtig',dtig)
    # print ('dtig /3600/3',dtig/3600/3)
    # print ('indices de base ',k0,i0,j0)

    # --- extraction des 4 points aux deux temps ---
    u00_0 = GRGFS[k0, i0, j0, 0]
    u10_0 = GRGFS[k0, i1, j0, 0]
    u01_0 = GRGFS[k0, i0, j1, 0]
    u11_0 = GRGFS[k0, i1, j1, 0]

    v00_0 = GRGFS[k0, i0, j0, 1]
    v10_0 = GRGFS[k0, i1, j0, 1]
    v01_0 = GRGFS[k0, i0, j1, 1]
    v11_0 = GRGFS[k0, i1, j1, 1]

    u00_1 = GRGFS[k1, i0, j0, 0]
    u10_1 = GRGFS[k1, i1, j0, 0]
    u01_1 = GRGFS[k1, i0, j1, 0]
    u11_1 = GRGFS[k1, i1, j1, 0]

    v00_1 = GRGFS[k1, i0, j0, 1]
    v10_1 = GRGFS[k1, i1, j0, 1]
    v01_1 = GRGFS[k1, i0, j1, 1]
    v11_1 = GRGFS[k1, i1, j1, 1]

    # --- temps d'abord sur les 4 coins ---
    u00 = u00_0 * (1.0 - wt) + u00_1 * wt
    u10 = u10_0 * (1.0 - wt) + u10_1 * wt
    u01 = u01_0 * (1.0 - wt) + u01_1 * wt
    u11 = u11_0 * (1.0 - wt) + u11_1 * wt

    v00 = v00_0 * (1.0 - wt) + v00_1 * wt
    v10 = v10_0 * (1.0 - wt) + v10_1 * wt
    v01 = v01_0 * (1.0 - wt) + v01_1 * wt
    v11 = v11_0 * (1.0 - wt) + v11_1 * wt

    # --- bilinéaire espace ---
    u0 = u00 * (1.0 - wi) + u10 * wi
    u1 = u01 * (1.0 - wi) + u11 * wi
    u = u0 * (1.0 - wj) + u1 * wj

    v0 = v00 * (1.0 - wi) + v10 * wi
    v1 = v01 * (1.0 - wi) + v11 * wi
    v = v0 * (1.0 - wj) + v1 * wj

    tws = np.sqrt(u*u + v*v) * 1.94384449

    angle_to = np.degrees(np.arctan2(v, u))
    twd = (270.0 - angle_to) % 360.0
    # print ('twd ',twd)
    return float(np.clip(tws, 1.0, 70.0)), float(twd)





def gfs_interpolateur_to(dtig, lat, lon):
    """
    Version torch scalaire de gfs_interpolateur_np.
    GR_t doit être un tensor torch de shape (nt, 721, 1440, 2)
    avec comp 0 = u, comp 1 = v.
    """
    global GR_t

    nt = GR_t.shape[0]
    device = GR_t.device
    dtype = GR_t.dtype

    # --- conversion en tensors ---
    dtig = torch.as_tensor(dtig, device=device, dtype=dtype)-60
    lat  = torch.as_tensor(lat,  device=device, dtype=dtype)
    lon  = torch.as_tensor(lon,  device=device, dtype=dtype)

    # --- temps ---
    h = dtig / 3600.0
    ft = h / 3.0

    k0 = torch.floor(ft).long()
    k0 = torch.clamp(k0, 0, nt - 2)
    k1 = k0 + 1

    wt = (ft - k0.to(dtype)).to(dtype)

    # --- indices espace ---
    lon360 = torch.remainder(lon, 360.0)

    fi = (90.0 - lat) / 0.25
    fj = lon360 / 0.25

    i0 = torch.floor(fi).long()
    i0 = torch.clamp(i0, 0, 719)
    i1 = i0 + 1

    jfloor = torch.floor(fj)
    j0 = torch.remainder(jfloor.long(), 1440)
    j1 = torch.remainder(j0 + 1, 1440)

    wi = (fi - i0.to(dtype)).to(dtype)
    wj = (fj - jfloor).to(dtype)

    # --- extraction des 4 points aux deux temps ---
    u00_0 = GR_t[k0, i0, j0, 0]
    u10_0 = GR_t[k0, i1, j0, 0]
    u01_0 = GR_t[k0, i0, j1, 0]
    u11_0 = GR_t[k0, i1, j1, 0]

    v00_0 = GR_t[k0, i0, j0, 1]
    v10_0 = GR_t[k0, i1, j0, 1]
    v01_0 = GR_t[k0, i0, j1, 1]
    v11_0 = GR_t[k0, i1, j1, 1]

    u00_1 = GR_t[k1, i0, j0, 0]
    u10_1 = GR_t[k1, i1, j0, 0]
    u01_1 = GR_t[k1, i0, j1, 0]
    u11_1 = GR_t[k1, i1, j1, 0]

    v00_1 = GR_t[k1, i0, j0, 1]
    v10_1 = GR_t[k1, i1, j0, 1]
    v01_1 = GR_t[k1, i0, j1, 1]
    v11_1 = GR_t[k1, i1, j1, 1]

    # --- interpolation temps d'abord ---
    u00 = u00_0 * (1.0 - wt) + u00_1 * wt
    u10 = u10_0 * (1.0 - wt) + u10_1 * wt
    u01 = u01_0 * (1.0 - wt) + u01_1 * wt
    u11 = u11_0 * (1.0 - wt) + u11_1 * wt

    v00 = v00_0 * (1.0 - wt) + v00_1 * wt
    v10 = v10_0 * (1.0 - wt) + v10_1 * wt
    v01 = v01_0 * (1.0 - wt) + v01_1 * wt
    v11 = v11_0 * (1.0 - wt) + v11_1 * wt

    # --- bilinéaire espace ---
    u0 = u00 * (1.0 - wi) + u10 * wi
    u1 = u01 * (1.0 - wi) + u11 * wi
    u = u0 * (1.0 - wj) + u1 * wj

    v0 = v00 * (1.0 - wi) + v10 * wi
    v1 = v01 * (1.0 - wi) + v11 * wi
    v = v0 * (1.0 - wj) + v1 * wj

    # --- vent ---
    tws = torch.sqrt(u * u + v * v) * 1.94384449
    tws = torch.clamp(tws, 1.0, 70.0)

    angle_to = torch.rad2deg(torch.atan2(v, u))
    twd = torch.remainder(270.0 - angle_to, 360.0)

    return tws,twd






#__________________________________________________________________________________
# Chargement ECM
#_________________________________________________________________________________
def gribFileNameECM(basedir: str, now_utc: float | None = None):
    """
    Choisit le dernier run ECMWF supposé 'disponible' en fonction de l'heure UTC,
    et construit le nom de npz + renvoie (date, heure_run).

    Retour:
      filenameC: str  ex: ".../EcmC_20260228-00.npz"
      date: int       yyyymmdd (ex: 20260228)
      heure_run: int  0 ou 12
      tig_run: float  epoch seconds UTC du run (pratique pour meta)
    """
    if now_utc is None:
        now_utc = time.time()

    dt = datetime.fromtimestamp(now_utc, tz=timezone.utc)
    mn_jour_utc = dt.hour * 60 + dt.minute

   # Seuils "safe" (tu as mis 07:40 et 19:40 UTC)
    # Avant 07:40 -> on prend le 12Z de la veille
    # E ntre 07:40 et 19:40 -> on prend le 00Z du jour
    # Après 19:40 -> on prend le 12Z du jour
    if mn_jour_utc < (7 * 60 + 30):
        run_hour = 12
        run_date = (dt.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)).date()
    elif mn_jour_utc < (19 * 60 + 30):
        run_hour = 0
        run_date = dt.date()
    else:
        run_hour = 12
        run_date = dt.date()


    # print ('run_hour',run_hour)
    date_int = int(run_date.strftime("%Y%m%d"))
    filenameC = f"{basedir}EcmC_{date_int}-{run_hour:02d}.npz"

    # tig du run (UTC)
    tig_run = datetime(run_date.year, run_date.month, run_date.day, run_hour, 0, 0, tzinfo=timezone.utc).timestamp()

    return filenameC, date_int, run_hour, tig_run




def load_ecmwf_npz_i16(path_npz: str):
    z = np.load(path_npz, allow_pickle=False)

    Uq = z["Uq"]
    Vq = z["Vq"]
    scale = float(z["scale"])
    tig = float(z["tig"])
    run_hour = int(z["run_hour"])

    lat0 = float(z.get("lat0", np.nan))
    lon0 = float(z.get("lon0", np.nan))
    dlat = float(z.get("dlat", np.nan))
    dlon = float(z.get("dlon", np.nan))

    # steps_h peut manquer dans les anciens packs
    if "steps_h" in z:
        steps_h = z["steps_h"].astype(np.int32, copy=False)
    else:
        # fallback: reconstruire depuis T si tu sais que c'est 0..144 par 3h
        T = Uq.shape[0]
        steps_h = (np.arange(T, dtype=np.int32) * 3)

    return Uq, Vq, steps_h, scale, tig, run_hour, (lat0, lon0, dlat, dlon)




# def dequantize_uv_int16_fixed(Uq: np.ndarray, Vq: np.ndarray, scale: float) -> np.ndarray:
#     """
#     Reconstitue float32 (T,Y,X,2) mais pas necessaire si on envoie directement au gpu
#     """
#     U = Uq.astype(np.float32) * np.float32(scale)
#     V = Vq.astype(np.float32) * np.float32(scale)
#     return np.stack([U, V], axis=-1)


def wrap_lon_m180_180_torch(lon):
    """
    lon: torch tensor (deg)
    """
    return torch.remainder(lon + 180.0, 360.0) - 180.0



class ECMWFInterpGPU_I16:
    """
    Interpolateur ECMWF 0.25°:
      - lat décroissante (90 -> -90)
      - lon dans [-180, 180)
      - steps irréguliers (0..144/3h, puis 6h)
    Entrées :
      Uq, Vq : int16 (T, ny, nx)  quantifiés
      scale  : float32 (m/s par pas int16)
      steps_h: int32 (T,) heures depuis run
    """
    def __init__(self, Uq_i16: torch.Tensor, Vq_i16: torch.Tensor,
                 scale: float, steps_h: torch.Tensor,
                 lat0: float = 90.0, lon0: float = -180.0,
                 dlat: float = 0.25, dlon: float = 0.25,
                 device="cuda"):
        self.device = torch.device(device)
        self.Uq = Uq_i16.to(self.device, non_blocking=True)
        self.Vq = Vq_i16.to(self.device, non_blocking=True)
        self.scale = float(scale)
        self.steps_h = steps_h.to(self.device, non_blocking=True).to(torch.int32)

        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.dlat = float(dlat)
        self.dlon = float(dlon)

        # dimensions
        self.T, self.ny, self.nx = self.Uq.shape

        # bornes indexables
        self.max_i = self.ny - 2
        self.max_j = self.nx - 2

    @torch.no_grad()
    def __call__(self, dtig: float | torch.Tensor, lat0, lon0, return_uv=False):
        """
        dtig: secondes depuis tig_run (float ou tensor scalaire)
        lat0/lon0: scalaires ou arrays -> renvoie arrays (N,)
        """
        dev = self.device

        lat = torch.as_tensor(lat0, dtype=torch.float32, device=dev).reshape(-1)
        lon = torch.as_tensor(lon0, dtype=torch.float32, device=dev).reshape(-1)
        lon = wrap_lon_m180_180_torch(lon)
        
        # --- indices spatiaux (lat décroissante)
        fi = (self.lat0 - lat) / self.dlat
        fj = (lon - self.lon0) / self.dlon

        i0 = torch.floor(fi).to(torch.int64)
        j0 = torch.floor(fj).to(torch.int64)

        wi = (fi - i0.to(torch.float32)).clamp(0.0, 1.0)
        wj = (fj - j0.to(torch.float32)).clamp(0.0, 1.0)

        i0 = i0.clamp(0, self.max_i)
        j0 = j0.clamp(0, self.max_j)
        i1 = i0 + 1
        j1 = j0 + 1

        # --- temps
        if isinstance(dtig, torch.Tensor):
            h = (dtig.to(torch.float32) / 3600.0).item()
        else:
            h = float(dtig) / 3600.0

        # clamp dans la plage
        h = max(float(self.steps_h[0].item()), min(h, float(self.steps_h[-1].item()) - 1e-6))

        steps = self.steps_h.to(torch.float32)
        # idx = premier > h  => k = idx-1
        idx = int(torch.searchsorted(steps, torch.tensor(h, device=dev), right=False).item())
        k = max(0, min(idx - 1, self.T - 2))

        h0 = float(steps[k].item())
        h1 = float(steps[k + 1].item())
        a = (h - h0) / (h1 - h0)  # 0..1
        a = float(max(0.0, min(1.0, a)))

        # --- lecture bilinéaire pour un temps donné t
        def sample_uv_at_t(t: int):
            U = self.Uq[t].to(torch.float32) * self.scale
            V = self.Vq[t].to(torch.float32) * self.scale

            U00 = U[i0, j0]; U01 = U[i0, j1]; U10 = U[i1, j0]; U11 = U[i1, j1]
            V00 = V[i0, j0]; V01 = V[i0, j1]; V10 = V[i1, j0]; V11 = V[i1, j1]

            # bilinéaire
            U0 = U00 * (1 - wj) + U01 * wj
            U1 = U10 * (1 - wj) + U11 * wj
            UU = U0 * (1 - wi) + U1 * wi

            V0 = V00 * (1 - wj) + V01 * wj
            V1 = V10 * (1 - wj) + V11 * wj
            VV = V0 * (1 - wi) + V1 * wi

            return UU, VV

        u0, v0 = sample_uv_at_t(k)
        u1, v1 = sample_uv_at_t(k + 1)

        u = u0 * (1.0 - a) + u1 * a
        v = v0 * (1.0 - a) + v1 * a

        if return_uv:
            return u, v

        # option : vitesse + direction "from" (comme souvent)

        #   vit_kn = torch.sqrt(u*u + v*v) * KNOTS
        # vit_kn = torch.clamp(vit_kn, min=self.clamp_min, max=self.clamp_max)


        
        # ang_math = torch.atan2(v, u) * self._rad2deg               # 0=Est, CCW
        # ang_tow  = torch.remainder(270.0 - ang_math, 360.0)        # 0=N, CW (towards)
        # ang_from = torch.remainder(ang_tow + 180.0, 360.0)         # from

        ang_math = torch.atan2(v, u) * (180.0 / math.pi)
        ang_from = torch.remainder(270.0 - ang_math, 360.0)


        
       
        spd = torch.sqrt(u*u + v*v)* KNOTS
        # ang_from = (torch.rad2deg(torch.atan2(-u, -v)) + 360.0) % 360.0
        return spd, ang_from
    
    def build(self, sizes=(1024, 512), h=0.0, lat_range=(-60.0, 60.0), lon_range=(-180.0, 180.0)):
        dev = torch.device(self.device) if isinstance(self.device, str) else self.device

        with torch.no_grad():
            for n in sizes:
                lat = torch.linspace(lat_range[0], lat_range[1], n, device=dev, dtype=torch.float32)
                lon = torch.linspace(lon_range[0], lon_range[1], n, device=dev, dtype=torch.float32)
                _ = self(h, lat, lon)
                _ = self(h, lat, lon)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        return self


def drop_old_interp_ecm():
    global ecm_interpolateur
    ecm_interpolateur = None
    gc.collect()
    torch.cuda.empty_cache()


# mise a jour ECM
# on va seulement comparer le nom du tigECM avec le dernier disponible 
def majgribECM():
    global tigECM,ecm_interpolateur
    fileNameECM, date_int, run_hour, dernierTigECM =gribFileNameECM(basedirGribsECM)
    if dernierTigECM>tigECM:
        # dans ce cas on recharge completement 
        Uq, Vq, steps_h, scale, tigECM, run_hour, (lat0, lon0, dlat, dlon) = load_ecmwf_npz_i16(fileNameECM)
        steps_t = torch.from_numpy(steps_h)  # int32
        Uq_t = torch.from_numpy(Uq)  # (T,ny,nx) int16
        Vq_t = torch.from_numpy(Vq)
        drop_old_interp_ecm()
        ecm_interpolateur = ECMWFInterpGPU_I16(Uq_t, Vq_t, scale=0.01, steps_h=steps_t, device="cuda").build()
        print ('Nouvelles valeurs tigECM:{} '.format(time.strftime(" %d %b %H:%M ",time.gmtime(tigECM)))) 
    else :
        print('Le grib ECM du {} est à jour \n'.format(time.strftime(" %d %b %H:%M ",time.gmtime(tigECM))))
try:        
#Chargement initial
    basedirGribsECM='/home/jp/gribslocaux/ecmwf/'
    fileNameECM, date_int, run_hour, tigECM =gribFileNameECM(basedirGribsECM)
    # print ('filenameECM',fileNameECM)
    # print ('tigECM : ',time.strftime(" %d %b %H:%M ",time.gmtime(tigECM)))
    Uq, Vq, steps_h, scale, tigECM, run_hour, (lat0, lon0, dlat, dlon) = load_ecmwf_npz_i16(fileNameECM)
    steps_t = torch.from_numpy(steps_h)  # int32
    Uq_t = torch.from_numpy(Uq)  # (T,ny,nx) int16
    Vq_t = torch.from_numpy(Vq)
    ecm_interpolateur = ECMWFInterpGPU_I16(Uq_t, Vq_t, scale=0.01, steps_h=steps_t, device="cuda").build()
except:
    None
try:
    majgribECM()
except:
    None

#####################################################################################
# Cache pour les fichiers 
#####################################################################################

app.config['CACHE_TYPE'] = 'simple'  # 'filesystem' pour stockage sur disque
app.config['CACHE_DEFAULT_TIMEOUT'] = 86400  # 24 heures

cache = Cache(app)
cache.init_app(app)  # Initialisation

def get_fichier(filename):
    """ Charge un fichier numpy en cache si ce n'est pas déjà fait """
    cached_data = cache.get(filename)
    
    if cached_data is None:
        try:
            with open(filename, 'rb') as f:
                cached_data = np.load(f)
            cache.set(filename, cached_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Erreur : Fichier {filename} introuvable")

    return cached_data


#####################################################################################
# Cache pour les cartes 
#####################################################################################


# Cache partagé, avec TTL de 5 jours (en secondes)
cache_cartes = TTLCache(maxsize=1000, ttl=5 * 24 * 60 * 60)
cache_lock = Lock()

def retirer_lacs(masque):
    """
    Enlève les lacs intérieurs d’un masque booléen (True=mer, False=terre),
    en gardant uniquement la mer connectée au bord.
    """
    # Connexité 4 (segments H/V seulement, pas diagonales)
    masque = ~masque 
    structure = generate_binary_structure(2, 1)

    # Labelisation des zones d'eau (True = mer)
    labels, num_labels = label(masque, structure=structure)

    # On récupère les labels présents sur les bords du masque
    bords = np.concatenate([
        labels[0, :],           # bord haut
        labels[-1, :],          # bord bas
        labels[:, 0],           # bord gauche
        labels[:, -1]           # bord droit
    ])
    labels_mer_connectee = np.unique(bords)
    labels_mer_connectee = labels_mer_connectee[labels_mer_connectee != 0]

    # On crée un nouveau masque avec uniquement les zones d'eau connectées au bord
    masque_sans_lacs = np.isin(labels, labels_mer_connectee)

    return masque_sans_lacs



def get_carte(lat, lon):
    # Centre sur les multiples de 5 (carte 10x10)
    lat_centre = (int(lat) // 5) * 5
    lon_centre = (int(lon) // 5) * 5

    cle = (lat_centre, lon_centre)

    with cache_lock:
        if cle in cache_cartes:
            return cache_cartes[cle]

    # Calcul ou chargement de la carte (taille fixe de 10x10)
    carte = fcarte(lat_centre, lon_centre)

    with cache_lock:
        cache_cartes[cle] = carte

    return carte




def updateUsers(users, username, user_id):
    users[user_id] = username
    #print ('************************',users,'/n****************************************')


###############################################################################"
# 913 Base de donnees et tables 
###############################################################################"


pg_pool = pool.SimpleConnectionPool(
                                        1, 10,  # minconn, maxconn
                                        dbname="vrouteur",
                                        user="jp",
                                        password="Licois1000",
                                        host="localhost",  # ou l'adresse IP/nom de domaine
                                        port="5432"        # par défaut PostgreSQL
                                    )


# sera a mettre apres creation des bases dans les fonctions de recherche
conn = pg_pool.getconn()
cursor = conn.cursor()





# def start_scheduler():
#     scheduler = BackgroundScheduler()
#     scheduler.add_job(delete_old_records, 'interval', days=1)  # Exécute tous les jours
#     scheduler.start()

# delete_old_records()


###############################################################################"
# creation d'une classe pour enregistrements des donnees course en cache 
###############################################################################"


#         if value == request.sid:
#             client_id = key
#             break
#     if client_id:
#         del clients[client_id]  # Retirer le client déconnecté
#         leave_room(client_id)
#     print(f"Client {client_id} déconnecté")

# @socketio.on('message')
# def handle_message(data):
#     print(f"Message reçu : {data}")   



# #print ('clients',clients )


#     # Fonction pour envoyer une mise à jour personnalisée
# def send_update_to_client(client_id, message):
#     print('ligne 97 ',clients)
#     if client_id in clients:
#         socketio.emit('update', {'message': message}, room=client_id)
#     else:
#         print(f"Client {client_id} non connecté")    






#send_update_to_client( "user1234", 'Coucou 1234 ')


            
################################################################################################
################      Gestion des sessions     #################################################
################################################################################################




app.secret_key="3777cf54e0080367185bfec476f58c4f6060c6587f677ea4538d8fff6daa4ef7"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"


################################################################################################
################      Gestion des repertoires  #################################################
################################################################################################




# pg_pool = pool.SimpleConnectionPool(
#                                         1, 10,  # minconn, maxconn
#                                         dbname="vrouteur",
#                                         user="jp",
#                                         password="Licois1000",
#                                         host="localhost",  # ou l'adresse IP/nom de domaine
#                                         port="5432"        # par défaut PostgreSQL
#                                     )


# # sera a mettre apres creation des bases dans les fonctions de recherche
# conn = pg_pool.getconn()
# cursor = conn.cursor()

# # connection a la base postgres locale 
# conn = psycopg2.connect(
#     host="localhost",
#     database="databaselinux3",
#     user="jp",
#     password="Licois1000"
# )
# cursor = conn.cursor()





# def start_scheduler():
#     scheduler = BackgroundScheduler()
#     scheduler.add_job(delete_old_records, 'interval', days=1)  # Exécute tous les jours
#     scheduler.start()

# delete_old_records()


###############################################################################"
# creation d'une classe pour enregistrements des donnees course en cache 
###############################################################################"


class CpuGpuCache:
    def __init__(self, maxsize=10, ttl=3600):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl  # secondes

    def get(self, key):
        self.cleanup()
        if key in self.cache:
            value, _ = self.cache.pop(key)
            self.cache[key] = (value, time.time())  # mise à jour du TTL
            return value
        return None

    def set(self, key, cpu_data, gpu_data):
        self.cleanup()
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)  # LRU
        self.cache[key] = ((cpu_data, gpu_data), time.time())

    def cleanup(self):
        now = time.time()
        expired = [k for k, (_, t) in self.cache.items() if now - t > self.ttl]
        for k in expired:
            del self.cache[k]

    def clear(self):
        self.cache.clear()





###############################################################################"
# creation des tables postgresql a faire une fois uniquement 
###############################################################################"

# actions et tables

# table                API                                  Fonction recherche                                  url                                         Javascript
# boatinfos         boatinfos                               rechercheTableBoatInfos(user_id, course)           @app.route('/rechercheboatinfos'         chercheboatinfos(username, course)
# coursesactives    Accountdetailrequest                    rechercheTableCoursesActives(username)             @app.route('/recherchecoursesuser'       cherchecoursesuser(username)
# racesinfos        getracesinfos                           rechercheTableRacesinfos()                         @app.route('/rechercheracesinfos'        chercheracesinfos()
# personalinfos     ne vient pas de l api personalinfos     rechercheTablePersonalInfos(user_id, course):      @app.route('/recherchepersonalinfos'     recherchepersonalinfos(username, user_id, course) 
# leginfos          leginfos                                rechercheTableLegInfos(course)                     @app.route('/rechercheleginfos',         chercheleginfos(course)
# progsvr           boatactions                             rechercheTableProgsvr(user_id, course)             @app.route('/rechercheprogsvr',          chercheprogsvr(user_id, course, t0routage)    
# polaires          polaires                                               (polar_id)                          @app.route("/parametres"
# fleetinfos        fleetinfos                              rechercheTableFleetinfos(user_id, course)          @app.route('/rechercheflotte',           afficheFlotte()
# teammembers       teammembers                             rechercheTableTeammembers(teamname)                @app.route('/rechercheteam'                 chercheTeamNames(teamname)


# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS boatinfos
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  timestamp REAL,
#                                  username TEXT,
#                                  user_id TEXT,
#                                  course TEXT,
#                                  boatinfos TEXT                          
#                              )          
#                                  """)
# conn.commit()
# print ('la table boatinfos      a ete creee si elle n existait pas ')


# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS boatactions
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  timestamp REAL,
#                                  username TEXT,
#                                  user_id TEXT,
#                                  course TEXT,
#                                  boatactions TEXT                          
#                              )          
#                                  """)
# conn.commit()
# print ('la table boatactions    a ete creee si elle n existait pas ')





# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS coursesactives
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  timestamp REAL,
#                                  username TEXT,
#                                  user_id TEXT,
#                                  coursesactives TEXT                          
#                              )          
#                                  """)
# conn.commit()
# print ('la table coursesactives a ete creee si elle n existait pas ')



# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS racesinfos
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  timestamp REAL,
#                                  racesinfos TEXT                          
#                              )          
#                                  """)
# conn.commit()
# print ('la table racesinfos     a ete creee si elle n existait pas ')



# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS personalinfos
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  timestamp REAL,
#                                  username  TEXT,
#                                  user_id   TEXT,
#                                  course    TEXT, 
#                                  personalinfos TEXT                          
#                              )          
#                                  """)
# conn.commit()
# print ('la table personalinfos  a ete creee si elle n existait pas ')


# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS leginfos
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  timestamp REAL,
#                                  course    TEXT, 
#                                  leginfos TEXT                          
#                              )          
#                                  """)
# conn.commit()
# print ('la table leginfos       a ete creee si elle n existait pas ')




# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS progsvr
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  timestamp      REAL,
#                                  user_id        TEXT,
#                                  course         TEXT, 
#                                  progsvr        TEXT                          
#                              )          
#                                  """)
# conn.commit()
# print ('la table progsvr        a ete creee si elle n existait pas ')




# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS polaires
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  _id            INTEGER,
#                                  updated        INTEGER,
#                                  polaires       TEXT
                                                           
#                              )          
#                                  """)
# conn.commit()
# print ('la table polaires       a ete creee si elle n existait pas ')



# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS fleetinfos
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  timestamp      REAL,
#                                  username       TEXT,   
#                                  user_id        TEXT,
#                                  course         TEXT,
#                                  fleetinfos     TEXT
                                                           
#                              )          
#                                  """)
# conn.commit()
# print ('la table fleetinfos     a ete creee si elle n existait pas ')





# cursor.execute("""
#                          CREATE TABLE IF NOT EXISTS teammembers
#                              (
#                                  id SERIAL PRIMARY KEY  ,
#                                  timestamp      REAL,
#                                  username       TEXT,   
#                                  user_id        TEXT,
#                                  teamname       TEXT, 
#                                  team_id        TEXT,
#                                  teammembers    TEXT
                                                           
#                              )          
#                                  """)
# conn.commit()
# print ('la table teammembers    a ete creee si elle n existait pas ')








# cursor.execute("DROP TABLE IF EXISTS progsvr")
# conn.commit()
# print("Ancienne table 'progsvr' supprimée.")

# Créer la nouvelle table avec les bons types
# cursor.execute("""
#     CREATE TABLE IF NOT EXISTS progsvr (
#         id SERIAL PRIMARY KEY,
#         timestamp TIMESTAMP WITH TIME ZONE,
#         user_id TEXT,
#         course TEXT,
#         progsvr JSONB
#     )
# """)
# conn.commit()
# print("Nouvelle table 'progsvr' créée.")


# cursor.execute("""

# CREATE TABLE IF NOT EXISTS historoutages (
#     id SERIAL PRIMARY KEY,
#     timestamp REAL,
#     username TEXT,
#     user_id TEXT,
#     status TEXT,
#     heuredepart REAL,
#     lat REAL,
#     lon REAL,
#     eta REAL
# );
#                  """)
# conn.commit()


#print ('la table teammembers    historoutages a ete creee si elle n existait pas ')

# cursor.execute("""
# ALTER TABLE boatinfos2 RENAME TO historoutages;
#                  """)
# conn.commit()

# cursor.execute("""
# ALTER TABLE historoutages
# ADD COLUMN course TEXT;""")
# conn.commit()

###############################################################################
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
    conn = None
    cursor = None
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
    except Exception as e:
        print("Erreur lors de l'exécution SQL :", e)

    finally:
        if cursor:
            cursor.close()
        if conn:
            pg_pool.putconn(conn)  # très important : rend la connexion au pool    
        
    


def rechercheTableBoatActions(user_id, course):
    # conn = None
    # cursor = None
    conn = pg_pool.getconn()
    try:
      
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp,boatactions
            FROM boatactions
            WHERE user_id = %s AND course = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id, course))

        result = cursor.fetchone()
        # print("Résultat brut de la requête SQL :", result)
        return (result[0], result[1]) if result else (None, None)
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            pg_pool.putconn(conn)  #  rend la connexion au pool    
        
    





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
    # print ('Ligne 557 course ',course )
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
        # print("Résultat brut de la requête SQL :", result)
        return (result[0], result[1]) if result else (None, None)
    finally:
        cursor.close()
        pg_pool.putconn(conn)



def rechercheTableFleetinfos(user_id, course):
    conn = None
    cursor = None
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT fleetinfos 
        FROM fleetinfos 
        WHERE user_id=%s AND course=%s 
        ORDER BY timestamp DESC 
        LIMIT 1
        """, (user_id, course))
        result = cursor.fetchone()
        return result[0] if result else None
    
    except Exception as e:
        print("Erreur lors de l'exécution SQL :", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            pg_pool.putconn(conn)  # très important : rend la connexion au pool    




def rechercheTableTeammembers(teamname):
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT teammembers
            FROM teammembers
            WHERE teamname = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (teamname,))
        result = cursor.fetchone()
        #print("Résultat brut de la requête :", result)
        return (result[0]) if result else (None)
    finally:
        cursor.close()
        pg_pool.putconn(conn)       
        
    
      


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




def findUsername(user_id):
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT username
            FROM boatinfos
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id,))
        row = cursor.fetchone()
        return row[0] if row else None
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
            print('recalcul des polaires pour le bateau ',_id)     

            timestamp,polar_id,polairesjsonstr= rechercheTablePolaires( _id)                       # nouvelle version
        
          
            polairesjson=json.loads(polairesjsonstr) 
            polairesjson=json.loads(polairesjson)    
            stringPolairestoNpy(polairesjson)

            


            
        
        #ATTENTION IL FAUT AUSSI LES RECALCULER
        #""""""""""""""""""""""""""""""""""""""
        
    finally:
        cursor.close()
        pg_pool.putconn(conn)




def insert_historoutages(  username,course, user_id, status, heuredepart, lat, lon, eta):
    conn = pg_pool.getconn()
    try:
        cursor = conn.cursor()
        timestamp=int(time.time())
        sql = """
            INSERT INTO historoutages
            (timestamp, username, course, user_id, status, heuredepart, lat, lon, eta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        with conn.cursor() as cur:
            cur.execute(sql, (timestamp, username, course, user_id, status,
                            heuredepart, lat, lon, eta))
            conn.commit()
    finally:
        cursor.close()
        pg_pool.putconn(conn)
        


###############################################################################
########### Fonctions diverses        #########################################
###############################################################################


def extraitcourse (infotoutescoursesStr,course):
    listetoutescourses = json.loads(infotoutescoursesStr)

# Extraction des courses
    nombreCourses = len(listetoutescourses['res'])
    tabNameCourses = []

    for i in range(nombreCourses):
        coursei = listetoutescourses['res'][i]
        numero = f"{coursei['raceId']}.{coursei['legNum']}"
        startdate = coursei['startDate']
        status = coursei['status']
        polar_id = coursei['boat']['polar_id']
        try :
            val=coursei['start']['radius']
        except:
            val=0    
            
        depart = [
            coursei['start']['lat'],
            coursei['start']['lon'],
            coursei['start']['name'],
            coursei['start']['heading'],
            val,
        ]
        
        try :
            val=coursei['end']['radius']
        except:
            val=0   

        try :
            heading=coursei['end']['heading']
        except:
            heading=0   



        end = [
            coursei['end']['lat'],
            coursei['end']['lon'],
            coursei['end']['name'],
            heading,
            val,
        ]
        
        tabNameCourses.append([numero, f"{coursei['legName']} ({numero})", startdate, status, depart, end, polar_id])

    print ('\n tabnamecourses ',tabNameCourses)

    # Recherche d'une course spécifique par numéro
    
    course_trouvee = next((courseelt for courseelt in tabNameCourses if courseelt[0] == course), None)

    if course_trouvee:
        print("Course trouvée :", course_trouvee)
    else:
        print("Course non trouvée.")

    return course_trouvee    





def condenseprogs(progs):
    # print ('Dans condense\n************')
    #print (progs)

    progsx=[]
    try:
        for i in range (len(progs)):
            deg     = progs [i]['deg']
            autoTwa = progs [i]['autoTwa']
            isProg  = progs [i]['isProg']
            user_id = progs [i]['_id']['user_id']
            race    = progs [i]['_id']['race_id']
            leg     = progs [i]['_id']['leg_num']
            ts      = progs [i]['_id']['ts']
            action  = progs [i]['_id']['action']

            # print (progs[i])
            if (progs[i]['autoTwa'] ):                # twa= true
                twa=progs[i]['deg']
                value=twa
                option=1
                        
            else :
                option=0
                value=progs[i]['deg']

            progsx.append([int(ts/1000),option,value,isProg])
    except:
        print('\npas de programmation dans boataction\n' )

    # print()
    # print ('progsx\n',progsx)
    return progsx



#################################################################################################################################################
############      Fonctions de recherche des donnnees pour la classe  routage    #########################################################################
#################################################################################################################################################
def charger_donnees(course):
    '''A partir de la reference de la course retourne '''
    '''leginfos,tabexclusions,tabicelimits,carabateau,polairesglobales10to,tabvmg10to '''    
   
    print ('Recherche de leginfos pour course ',course)
    
    leginfostr  = rechercheTableLegInfos(course)
    leginfos    = json.loads(leginfostr)
      
    # print(leginfos)
    polar_id=leginfos['boat']['polar_id']
    

    filenamelocal1='polairesglobales10_'+str(polar_id)+'.npy'
    filename1=basedirnpy+filenamelocal1
    with open(filename1,'rb')as f:
         polairesglobales10 = np.load(f)
        
    filenamelocal2='vmg10_'+str(polar_id)+'.npy'
    filename2=basedirnpy+filenamelocal2
    with open(filename2,'rb')as f:
         tabvmg10 = np.load(f)   

    timestamp,polar_id,polairesjsonstr= rechercheTablePolaires( polar_id)                       # nouvelle version
    polairesjson=json.loads(polairesjsonstr) 
    polairesjson=json.loads(polairesjson) 
    # print ('polairesjson ',polairesjson)
    # print (polairesjson['sail'])
    nbvoiles=len(polairesjson['sail'])
    # print('nbvoiles',nbvoiles)
    typevoile          = []
    

    polar_id                    = polairesjson['_id']
    label                       = polairesjson['label']                               # type du bateau
    
    globalSpeedRatio            = polairesjson['globalSpeedRatio']
   
    foilSpeedRatio              = polairesjson['foil']['speedRatio']
    autoSailChangeTolerance     = polairesjson["autoSailChangeTolerance"]
    badSailTolerance            = polairesjson["badSailTolerance"]
    iceSpeedRatio               = polairesjson['iceSpeedRatio']
    maxSpeed                    = polairesjson['maxSpeed']
    twaMin                      = polairesjson['foil']['twaMin']
    twaMax                      = polairesjson['foil']['twaMax']
    twaMerge                    = polairesjson['foil']['twaMerge']
    twsMin                      = polairesjson['foil']['twsMin']
    twsMax                      = polairesjson['foil']['twsMax']
    twsMerge                    = polairesjson['foil']['twsMerge']
    speedRatio                  = polairesjson['foil']['speedRatio']
    hull                        = polairesjson['hull']['speedRatio']
    lws                         = polairesjson['winch']['lws']
    hws                         = polairesjson['winch']['hws']
    lwtimer                     = polairesjson['winch']['sailChange']['pro']['lw']['timer']
    hwtimer                     = polairesjson['winch']['sailChange']['pro']['hw']['timer']
    lwratio                     = polairesjson['winch']['sailChange']['pro']['lw']['ratio']
    hwratio                     = polairesjson['winch']['sailChange']['pro']['hw']['ratio']
    tackprolwtimer              = polairesjson['winch']['tack']['pro']['lw']['timer']
    tackprolwratio              = polairesjson['winch']['tack']['pro']['lw']['ratio']
    tackprohwtimer              = polairesjson['winch']['tack']['pro']['hw']['timer']
    tackprohwratio              = polairesjson['winch']['tack']['pro']['hw']['ratio']
    gybeprolwtimer              = polairesjson['winch']['gybe']['pro']['lw']['timer']
    gybeprolwratio              = polairesjson['winch']['gybe']['pro']['lw']['ratio']
    gybeprohwtimer              = polairesjson['winch']['gybe']['pro']['hw']['timer']
    gybeprohwratio              = polairesjson['winch']['gybe']['pro']['hw']['ratio']
    update                      = polairesjson['_updatedAt']
    
    # recuperation d'un tableau des polaires 
    tabtws                      = np.asarray(polairesjson['tws'])                                            
    tabtwa                      = np.asarray(polairesjson['twa'])
    nbvoiles                    = len(polairesjson['sail'])
    toutespolaires              = np.zeros((len(tabtwa),len(tabtws),nbvoiles))
    for i in range(nbvoiles) :
        typevoile.append( polairesjson['sail'][i]['name'])
        toutespolaires[:,:,i] = polairesjson['sail'][i]['speed']
        polairesmax           = globalSpeedRatio *np.amax(toutespolaires,axis=2) 
    
    coeffboat                   = tabCoeffboat[polar_id]['stamina']

    carabateau={ "polar_id":polar_id,'typevoile':typevoile, "label":label,"globalSpeedRatio":globalSpeedRatio,"foilSpeedRatio":foilSpeedRatio,"coeffboat":coeffboat,\
            "iceSpeedRatio":iceSpeedRatio,"autoSailChangeTolerance":autoSailChangeTolerance,"badSailTolerance":badSailTolerance,\
            "maxSpeed":maxSpeed,"twaMin":twaMin,"twaMax":twaMax,"twaMerge":twaMerge,"twsMin":twsMin,"twsMax":twsMax,"twsMerge":twsMerge,\
            "hull":hull,'lws': lws ,'hws':hws, "lwtimer":lwtimer,"hwratio": hwratio ,"hwtimer":hwtimer,"lwratio":lwratio,\
            "lwtimer":lwtimer,"tackprolwtimer":tackprolwtimer,"tackprolwratio":tackprolwratio,"tackprohwtimer":tackprohwtimer,\
            "tackprohwratio":tackprohwratio,"gybeprolwtimer":gybeprolwtimer,"gybeprolwratio":gybeprolwratio,\
            "gybeprohwtimer":gybeprohwtimer,"gybeprohwratio":gybeprohwratio,'typevoile':typevoile,'tabtws':tabtws,'tabtwa':tabtwa,'polairesmax':polairesmax}

     # Recuperation des zones d exclusion VR
    #**************************************
    try:
        print()
        zones=leginfos['restrictedZones']
     
        tabexclusions={}
        for zone in zones:
            name = zone["name"]
            vertices = [[pt["lat"], pt["lon"]] for pt in zone["vertices"]]
            tabexclusions[name] =  vertices
    except:        
        tabexclusions={}       
  

    try:
        iceLimits= leginfos['ice_limits']['south']   # print ('\n IceLimits',iceLimits)  c est la forme developpee
        tabicelimits=[]
        for element in iceLimits:
            lat=element['lat']
            lon=element['lon']
            tabicelimits.append([lat,lon])
        tabicelimits.extend(([-90,180],[-90,-180],tabicelimits[0]))     # on referme le polygone correctement
    except:
        tabicelimits=[]

    # polairesglobales10to = torch.from_numpy(polairesglobales10).to('cuda')
    # tabvmg10to           = torch.from_numpy(tabvmg10).to('cuda')     
    # 
    # 
    # 
    twa=55
    tws=12.1
    typeVoiles = ['jib', 'Spi', 'Staysail', 'LightJib', 'Code0', 'HeavyGnk', 'LightGnk']
    voile=typeVoiles[int(polairesglobales10[8,int(tws*10),int(twa*10)])]


    print ('\n*************************************************************************************************')   
    print ('Dernier Update chargé pour le bateau  ',update)
    print ('Resultats calculés avec polairesglobales (numpy) pour {:25} twa= {} tws= {} voile {} vitessemax = {:6.3f} '.format(label,twa,tws,voile,polairesglobales10[7,int(tws*10),int(twa*10)]))
    print ('\n*************************************************************************************************')
    print ('Resultats attendus pour Ocean 50            twa= 55 tws= 12.1 voile LightJib vitessemax = 11.956 ')
    print ('Resultats attendus pour Imoca               twa= 55 tws= 12.1 voile LightJib vitessemax = 11.799 ')
    print ('Resultats attendus pour Figaro3             twa= 55 tws= 12.1 voile LightJib vitessemax =  6.937 ')
    print ('Resultats attendus pour Ultim BP XI         twa= 55 tws= 12.1 voile Jib      vitessemax = 17.110 ')
    print ('Resultats attendus pour Class40             twa= 55 tws= 12.1 voile LightJib vitessemax =  8.656 ')
    print ('Resultats attendus pour CruiserRacer        twa= 55 tws= 12.1 voile Jib      vitessemax =  8.197 ')
    print ('Resultats attendus pour Volvo65             twa= 55 tws= 12.1 voile Jib      vitessemax = 10.402')
    print ('Resultats attendus pour Mini6.5             twa= 55 tws= 12.1 voile Jib      vitessemax =  6.1')
    print ('Resultats attendus pour Super Maxi 100      twa= 55 tws= 12.1 voile LightJib vitessemax = 15.457')
    print ('Resultats attendus pour OffshoreRacer       twa= 55 tws= 12.1 voile LightJib vitessemax =  8.730')
    print ('Resultats attendus pour multi/mod70         twa= 55 tws= 12.1 voile LightJib vitessemax = 14.810')
    print ('**************************************************************************************************\n')

    # twa=75
    # tws=15.1
   
    # print ('\n*************************************************************************************************')   
    # print ('Resultats calculés avec polairesglobales (numpy) pour {:25} twa= {} tws= {} voile {} vitessemax = {:6.3f} '.format(label,twa,tws,voile,polairesglobales10[7,int(tws*10),int(twa*10)]))
    # print ('\n*************************************************************************************************')
    # print ('Resultats attendus pour Ocean 50            twa= 55 tws= 12.1 voile LightJib vitessemax = 11.956 ')
    # print ('Resultats attendus pour Imoca               twa= 75 tws= 15.1 voile LightJib vitessemax = 16.848 ')
    # print ('Resultats attendus pour Figaro3             twa= 75 tws= 12.1 voile Code 0   vitessemax =  8.643 ')
    # print ('Resultats attendus pour Ultim BP XI         twa= 55 tws= 12.1 voile Jib      vitessemax = 17.110 ')
    # print ('Resultats attendus pour Class40             twa= 75 tws= 15.1 voile Code 0   vitessemax = 11.089 ')
    # print ('Resultats attendus pour CruiserRacer        twa= 55 tws= 12.1 voile Jib      vitessemax =  8.197 ')
    # print ('Resultats attendus pour Volvo65             twa= 55 tws= 12.1 voile Jib      vitessemax = 10.402')
    # print ('Resultats attendus pour Mini6.5             twa= 55 tws= 12.1 voile Jib      vitessemax =  6.1')
    # print ('Resultats attendus pour Super Maxi 100      twa= 55 tws= 12.1 voile LightJib vitessemax = 15.457')
    # print ('Resultats attendus pour OffshoreRacer       twa= 55 tws= 12.1 voile LightJib vitessemax =  8.730')
    # print ('Resultats attendus pour multi/mod70         twa= 55 tws= 12.1 voile LightJib vitessemax = 14.810')
    # print ('**************************************************************************************************\n')



    return leginfos,tabexclusions,tabicelimits,carabateau,polairesglobales10,tabvmg10,polairesjson

  




# recherche a partir du cache 
class DonneesCourse:
    def __init__(self, leginfos, tabexclusions, tabicelimits, carabateau,
                 polaires_np, vmg_cpu, polaires_gpu, vmg_gpu,polairesjson):
        self.leginfos = leginfos
        self.tabexclusions = tabexclusions
        self.tabicelimits  = tabicelimits
        self.carabateau    = carabateau
        self.polaires_np   = polaires_np
        self.vmg_cpu       = vmg_cpu
        self.polaires_gpu  = polaires_gpu
        self.vmg_gpu       = vmg_gpu
        self.polairesjson  = polairesjson


cache_donnees = CpuGpuCache(maxsize=10, ttl=3*3600)                          # definition du cache 



def rechercheDonneesCourseCache(course):
    cached = cache_donnees.get(course)

    if cached:
        print(f"[CACHE HIT] Course {course}")
        cpu_data, gpu_data = cached
        leginfos, tabexclusions, tabicelimits, carabateau, polaires_np, vmg_cpu, polairesjson = cpu_data
        polaires_gpu, vmg_gpu = gpu_data

    else:
        print(f"[CACHE MISS] Course {course}")
        leginfos, tabexclusions, tabicelimits, carabateau, polaires_np, vmg_np, polairesjson = charger_donnees(course)
        vmg_cpu = torch.tensor(vmg_np, dtype=torch.float32, device='cpu')
        polaires_cpu = torch.tensor(polaires_np, dtype=torch.float32, device='cpu')
        polaires_gpu = polaires_cpu.clone().cuda()
        vmg_gpu = vmg_cpu.clone().cuda()

        cache_donnees.set(
            course,
            cpu_data=(
                leginfos,
                tabexclusions,
                tabicelimits,
                carabateau,
                polaires_np,
                vmg_cpu,
                polairesjson
            ),
            gpu_data=(
                polaires_gpu,
                vmg_gpu
            )
        )

    return DonneesCourse( leginfos, tabexclusions, tabicelimits, carabateau, polaires_np, vmg_cpu, polaires_gpu, vmg_gpu, polairesjson )    # en fait on renvoie la classe DonneesCourse 











###########  2  #################################################################################################################################

def rechercheDonneesCourseUser( user_id,course): 
    '''Recherche les donnees generales boatinfos pour le user_id sur la course '''   
   # print ('RechercheDonneeesCourseUser pour user_id {} ,course {} '.format(user_id,course))
    result    = rechercheTableBoatInfos(user_id,course)  #recherche dans la table locale  
    
    boatinfos = json.loads(result) 
    boatinfosbs=boatinfos['bs']

    typebateau          = boatinfosbs['boat']['name']
    user_id             = boatinfosbs['_id']['user_id']
    username            = boatinfosbs['displayName']
    racevr              = boatinfosbs['_id']['race_id']
    legvr               = boatinfosbs['_id']['leg_num']
    polar_id            = boatinfosbs['boat']['polar_id']
    state               = boatinfosbs['state']
    headingvr           = boatinfosbs['heading']
    y0vr                = boatinfosbs['pos']['lat']
    x0vr                = boatinfosbs['pos']['lon']
    sailvr              = boatinfosbs['sail']
    speedvr             = boatinfosbs['speed']
    staminavr           = boatinfosbs['stamina']
    #statsstamina       = boatinfosbs['stats']                    # pas dans le vg
    gateGroupCounters   = boatinfosbs['gateGroupCounters']
    legStartDate        = boatinfosbs['legStartDate']
    
    print ('state',state)
    
    if state=='waiting':
        # si le state est waiting on a seulement un heading et legstartdate     
        lastCalcDate         = boatinfosbs['legStartDate']   # normalement deja recupere en dessus 
        t0=legStartDate/1000
        dtig=t0-tigGFS
        twsvr,twdvr = gfs_interpolateur_np(dtig,y0vr,x0vr)        
        capvr     = headingvr
        twavr     = ftwao(capvr,twdvr)
      
        twaAutovr =twavr
        # print ('Depart en attente le  {} avec tws {:6.2f} ,twd {:6.2f} ,cap {:6.2f} ,twa {:6.2f}'.format(time.strftime(" %d %b %H:%M ",time.localtime(t0)),twsvr,twdvr,headingvr,twavr))

    else:                                   # cas standard                           
        twavr                = boatinfosbs['twa']
        twsvr                = boatinfosbs['tws']
        twdvr                = boatinfosbs['twd']
        lastCalcDate         = boatinfosbs['lastCalcDate']

       
    try:
        twaAutovr =boatinfosbs['twaAuto']
    except:
        twaAutovr = twavr


    if twaAutovr==twavr:
        twaAuto=True
        option=1             # pour le torch departvr 
        valeur=twavr 
    else:
        twaAuto=False
        option=0             # pour le torch departvr 
        valeur= headingvr
   
    
    try:
        rankvr =boatinfosbs['rank']
    except:
        rankvr = 'NA'
    try:
        tsEndOfSailChange=boatinfosbs['tsEndOfSailChange']
    except:
        tsEndOfSailChange= lastCalcDate
    try:
        tsEndOfGybe=boatinfosbs['tsEndOfGybe']
    except:
        tsEndOfGybe= lastCalcDate
    try:
        tsEndOfTack=boatinfosbs['tsEndOfTack']
    except:
        tsEndOfTack= lastCalcDate
    t0vr=lastCalcDate/1000

    # on calcule la penalite en cours  le boost en cours s'il y a
    penovr      =(max((tsEndOfGybe-lastCalcDate),(tsEndOfTack-lastCalcDate))+(tsEndOfSailChange-lastCalcDate))/1000   # c'est la peno vr en secondes au moment du debut du routage
  
    if sailvr>10 :
        voileAuto=True 
    else:
        voileAuto=False
    if sailvr!=10:                    # voile en position waiting
        voile       = sailvr-1        #voile pour notation jp et sail pour notation vr
    else:
        voile=0 
    # voilemax    = polairesglobales10[8,tws10,twaroundabs]
    # vitvoilevr  = polairesglobales10[voilevrjp,tws10,twaroundabs]
    # vitvoilemax = polairesglobales10[7,tws10,twaroundabs]
    boost         = 1   #ne presente pas gros interet car n a pas d incidence sur le suivant et necessiterai de charger polairesglobales pour le calcul


    posStart={'username':username ,'race':racevr,'leg':legvr,'state':state,'numisoini':0,'npt':0,'nptmere':0,'nptini':0,'tws':twsvr,'twd':twdvr,'twa':twavr,'twaAuto':twaAuto,'y0':y0vr,'x0':x0vr,\
      't0':t0vr,'heading':headingvr,'rank':rankvr,'speed':speedvr,'voile':voile ,'voileAuto':voileAuto,'stamina':staminavr,\
      'lastCalcDate':lastCalcDate,'tsEndOfSailChange':tsEndOfSailChange,'tsEndOfGybe':tsEndOfGybe,'tsEndOfTack':tsEndOfTack,'gateGroupCounters':gateGroupCounters,'penovr':penovr,'boost':boost}


    dt1=60   # parametrage par defaut pour le calcul des progs 
    positionvr=torch.tensor([0,t0vr,dt1,option,valeur,y0vr,x0vr,voile,twavr,headingvr,speedvr,staminavr,penovr,twdvr,twsvr,voileAuto,boost],dtype=torch.float64,device='cpu')

    # recherche des zones d exclusions
    try:
        zones=boatinfos['leg']['restrictedZones']
        tabexclusions={}
        for zone in zones:
            
            name    = zone['name']
            vertices = [[pt["lat"], pt["lon"]] for pt in zone["vertices"]]
            tabexclusions[name] = vertices
            
    except:   
         tabexclusions={}
    
    return boatinfos,posStart,tabexclusions,positionvr    

    


###########  3  #################################################################################################################################



def rechercheDonneesPersoCourseUser( user_id,course): 
    '''pour l instant a besoin  user_id '''
    
    personalinfostr    = rechercheTablePersonalInfos(user_id,course)  #recherche dans la table , ce n est pas une recherche dans le serveur 
    personalinfos      = json.loads(personalinfostr)
    ari                = personalinfos['ari']
    try:
        waypoints  = sorted(personalinfos['wp'].values(), key=lambda x: x[0])     #waypoints sous forme de tableau trié par ordre
    except:
        waypoints=[]

    try:
        exclusions    = personalinfos['exclusions']
    except:
        exclusions=[] 

    try:
       barrieres  = personalinfos['barrieres']
       for k in barrieres:
           barrieres[k] = torch.tensor(barrieres[k], device='cuda') 

    except:
       barrieres=[]  
        
    try:
        trajets  = personalinfos['trajets']
    except:
       trajets=[]

    try :
        tolerancehvmg=personalinfos['tolerancehvmg']
    except:
        tolerancehvmg=0    
    
    try :
        retardpeno=personalinfos['retardpeno']
    except:
        retardpeno=0   
  
    return personalinfos,ari,waypoints,exclusions,barrieres,trajets,tolerancehvmg,retardpeno




###########  3  #################################################################################################################################

def calculePosDepart(posStartVR,polairesglobales,carabateau,dt=60,furler=0.8): 

    ''' donne la position de depart au bout de 60 s'''
    ''' Sous la meme forme que posStartVR          '''
    print ('\nposStartVR\n',posStartVR)
    print()

    state       = posStartVR['state']
    posStart = copy.deepcopy(posStartVR)
    
    if state != 'waiting'  :     
        numisoini   = posStartVR['numisoini']
        npt         = posStartVR['npt']
        nptmere     = posStartVR['nptmere']
        nptini      = posStartVR['nptini']
        y0          = posStartVR['y0']
        x0          = posStartVR['x0']
        t0          = posStartVR['t0']
        sail        = posStartVR['voile']
        voileAuto   = posStartVR['voileAuto']
        tws         = posStartVR['tws']
        twd         = posStartVR['twd']
        twa         = posStartVR['twa']
        twaAuto     = posStartVR['twaAuto']
        cap         = posStartVR['heading']
        speed       = posStartVR['speed']
        stamina     = posStartVR['stamina']
        soldepeno   = posStartVR['penovr']        
        boost       = posStartVR['boost']    
        lastCalcDate= posStartVR['lastCalcDate'] 
        numisoini   = posStartVR['numisoini'] 
        # on calcul le deplacement 
        lwtimer             = carabateau['lwtimer']
        hwtimer             = carabateau['hwtimer']
        lw                  = carabateau['lws']
        hw                  = carabateau['hws']
        lwtimerGybe         = carabateau['gybeprolwtimer']
        hwtimerGybe         = carabateau['gybeprohwtimer']
        coeffboat           = carabateau['coeffboat']
        MF                  = 0.8
        tws10=   round(tws*10)
        twa10=   round(abs(twa)*10)
        if twaAuto==twa:
            cap= fcap(twa,twd)
        else :
            twa =ftwaos(cap, twd)
        if sail>10:
            voileAuto=10
            voile=sail%10
        else :
            voileAuto=0
            voile=sail%10
        
            
        vitessevoileini   = polairesglobales[int(voile), tws10, twa10] 
        meilleurevitesse  = polairesglobales[  7         , tws10, twa10] 
        meilleurevoile    = polairesglobales[  8         , tws10, twa10] 
        boost             = meilleurevitesse/(vitessevoileini+0.0001)    
        if boost  >1.014:
            Chgt=1
        else :
            Chgt=0   

        Tgybe=0       # on ne fait aucune manoeuvre car le bateau continue sur sa lancee 
        Cstamina = 2 - 0.015 * stamina
    

#       # Pénalités

        peno_chgt = Chgt *peno_np(lwtimer, hwtimer, tws, stamina)* furler
#       peno_gybe = Tgybe * *peno_np(lwtimer, hwtimer, Tws, Stamina)   # pas de tackgybe dans les 60 s   
        peno_globale= soldepeno + peno_chgt
        peno_fin    = max(0,peno_globale-dt)
    
#       # Stamina
        perte_stamina = calc_perte_stamina_np(tws, int(Tgybe), Chgt, coeffboat)
        recup         = pts_recuperes(dt, tws)
        stamina       = stamina - perte_stamina + recup
        staminafin    =  max (0.0, min(100.0, stamina))
    
#         # Temps effectif après pénalités
        dt_ite = dt - 0.3 * max(peno_globale , dt)
    
        # Coordonnées après déplacement
        cap_rad = cap *math.pi/180
        y0_rad  = y0  *math.pi/180
        dlat = meilleurevitesse * dt_ite / 3600 / 60 * math.cos(cap_rad)
        dlon = meilleurevitesse * dt_ite / 3600 / 60 * math.sin(cap_rad) / math.cos(y0_rad)
        y1   = y0   + dlat
        x1   = x0   + dlon
        t1   = t0   + dt  
        #twsf,twdf= prevision025(GRGFS, t0+dt, y1, x1)

        # print ('dans calcul isodepart twsf {},twdf {} t0+dt {} '.format(twsf,twdf ,time.strftime(" %d %b %H:%M %S ",time.localtime(t0+dt)) ))  
      
        # dtig=torch.tensor(dtig, device='cuda') 
        # lat1 = torch.tensor([y1], device="cuda")
        # lon1 = torch.tensor([x1], device="cuda")
        dtig=   t0+dt-tigGFS

        # print ()
        # print (dtig)
        # print ('y1',y1)
        # print ('x1',x1)
        # twsf,twdf=gfs_interpolateur_np(dtig, y1, x1)    # vent au point d arrivee 
        # print ('twsf',twsf)
        # print ('twdf',twdf)
        # print()

        lat1 = torch.tensor([y1], device="cuda")
        lon1 = torch.tensor([x1], device="cuda")
        twsf,twdf=gfs_interpolateur_to(dtig, lat1, lon1)

       #twsf1,twdf1=  prevision025dtig(GRGFS, dtig, y1, x1)       # ancienne interpolation 
        
        
        # print ('dans calcul isodepart twsf1  {},twdf1 {} t0+dt {} '.format(twsf1,twdf1 ,time.strftime(" %d %b %H:%M %S ",time.localtime(t0+dt)) ))  
    
       
        posStart['tws']=float(twsf[0])
        posStart['twd']=float(twdf[0])
        posStart['twa']=float(twa)
        posStart['y0'] = float(y1)
        posStart['x0'] =float(x1)
        posStart['t0'] =t1
        posStart['heading']=cap
        posStart['speed']  = float(meilleurevitesse)
        posStart['voile']  = int(meilleurevoile +voileAuto)
        posStart['stamina'] = float(staminafin)
        posStart[ 'lastCalcDate'] = lastCalcDate +dt        
        posStart[ 'penovr'] = peno_fin    
    return posStart  
  


def calculeisodepart2(posStart):
    ''' Transforme le posStart en un isochrone de depart '''
    state       = posStart['state']
    numisoini   = posStart['numisoini']
    npt         = posStart['npt']
    nptmere     = posStart['nptmere']
    nptini      = posStart['nptini']
    y0          = posStart['y0']
    x0          = posStart['x0']
    t0          = posStart['t0']
    voile       = posStart['voile']%10
    voileAuto   = posStart['voileAuto']
    tws         = posStart['tws']
    twd         = posStart['twd']
    twa         = posStart['twa']
    twaAuto     = posStart['twaAuto']
    cap         = posStart['heading']
    speed       = posStart['speed']
    stamina     = posStart['stamina']
    soldepeno   = posStart['penovr']        
    boost       = posStart['boost']      
    numisoini   = posStart['numisoini'] 
    #x0=lon_to_360(x0)  # passage en 0 360 
    # isovr 22 elements  en 12  c'est l ecart par rapport a l iso precedent 
    isodepart       = torch.tensor ([[numisoini,npt,nptmere,y0,x0,voile,twa,stamina,soldepeno,tws,twd,cap,0,0,0,0,speed,0,boost,0,0,0]], dtype=torch.float32, device='cuda')
   
    return isodepart
       

# fonction pour passer de l isofinal a posend  
def pointfinalToPosEnd(pointfinal):
    # pointfinal          = torch.tensor ([[numisoini,npt,nptmere,ydep,xdep,voiledep,twadep,staminadep,soldepenodep,twsdep,twddep,capdep,60,0,0,0,vitdep,0,boostdep,0]], dtype=torch.float32, device='cuda')  
    numisoini    =int( pointfinal[0].item())
    npt          = int(pointfinal[1].item())
    nptmere      = int(pointfinal[2].item())
    y0           = pointfinal[3].item()
    x0           = pointfinal[4].item()
    voile        = pointfinal[5].item()
    twa          = pointfinal[6].item()
    stamina      = pointfinal[7].item()
    soldepeno    = pointfinal[8].item()
    tws          = pointfinal[9].item()
    twd          = pointfinal[10].item()
    cap          = pointfinal[11].item()
    ecartTemps   = pointfinal[12].item()
    vitesse      = pointfinal[16].item()
    boost        = pointfinal[18].item()
    voileAuto    = True
    
    posEnd={'numisoini':numisoini,'npt':npt,'nptmere':nptmere,'nptini':npt,'tws':tws,'twd':twd,'twa':twa,'twaAuto':twa,'y0':y0,'x0':x0,\
          'ecart':ecartTemps,'heading':cap,'speed':vitesse,'voile':voile ,'voileAuto':voileAuto,'stamina':stamina,'penovr':soldepeno,'boost':boost}
    return posEnd
  

def calculeisodepart2(posStart):
    ''' Transforme le posStart en un isochrone de depart '''
    state       = posStart['state']
    numisoini   = posStart['numisoini']
    npt         = posStart['npt']
    nptmere     = posStart['nptmere']
    nptini      = posStart['nptini']
    y0          = posStart['y0']
    x0          = posStart['x0']
    t0          = posStart['t0']
    voile       = posStart['voile']%10
    voileAuto   = posStart['voileAuto']
    tws         = posStart['tws']
    twd         = posStart['twd']
    twa         = posStart['twa']
    twaAuto     = posStart['twaAuto']
    cap         = posStart['heading']
    speed       = posStart['speed']
    stamina     = posStart['stamina']
    soldepeno   = posStart['penovr']        
    boost       = posStart['boost']      
    numisoini   = posStart['numisoini'] 
    # isovr 22 elements  en 12  c'est l ecart par rapport a l iso precedent 
    print ('posStart \n',posStart)
    isodepart       = torch.tensor ([[numisoini,npt,nptmere,y0,x0,voile,twa,stamina,soldepeno,tws,twd,cap,0,0,0,0,speed,0,boost,0,0,0]], dtype=torch.float32, device='cuda')
   
    return isodepart
    


###############################################################################################################################################################################""



def plagevoiles2(polar_id,tws):
    # On n 'est pas pressé par le temps on peut aller chercher l info sur le fichier  "
    basedir='/home/jp/staticLocal/npy/'

    
    filenamelocal1='polairesglobales10_'+str(polar_id)+'.npy'
    filename1=basedir+filenamelocal1
    polairesglobales10=get_fichier(filename1)
    # with open(filename1,'rb')as f:
    #      polairesglobales10 = np.load(f)
    tws10 = round(tws*10)
    voilei=0
    listevoiles=[]
    minvoiles=[]
    # On recherche toutes les voiles utilisees de 40 a 150
    
    for twa10 in range (0,1800):   # on doit pouvoir limiter
        voile=int(polairesglobales10[8,tws10,twa10])
        vit=(polairesglobales10[7,tws10,twa10])
        if voilei!=voile:
            listevoiles.append(voile)
            minvoiles.append(twa10/10)
            voilei=voile
            minvoiles[0]=0

    tab=np.zeros((len(listevoiles),5))
    
    tab[:,0]= listevoiles
    tab[:,1]= minvoiles
    tab[:,2]= np.roll(tab[:,1],-1)-0.1
    tab[-1,2]=180

# calcul des twamax en recouvrement
    twa0=tab[0,2]
    twaa   = round(twa0*10)
    for i in range (tab.shape[0]-1):
        boost=1
        while boost<1.014 and twaa<1800:
            twaa+=1
            vitessemax=polairesglobales10[7,tws10,twaa]
            vitessevoileinit=polairesglobales10[int(tab[i,0]),tws10,twaa]
            boost=vitessemax/vitessevoileinit
        twamax= (twaa-1)/10
        tab[i,4]=twamax
# calcul des twamin en recouvrement
    for i in range (1,tab.shape[0]):
        twa0=   tab[i,1]
        twaa   = round(twa0*10)
        boost=1
        while boost<1.014  :
                twaa-=1
                vitessemax=polairesglobales10[7,tws10,twaa]
                vitessevoileinit=polairesglobales10[int(tab[i,0]),tws10,twaa]
                if vitessemax!=0  :
                    boost=vitessemax/vitessevoileinit
                else:
                    break
        twamin= (twaa+1)/10
        tab[i,3]=twamin
    tab[-1,4]=180
    return tab


###############################################################################
########### Reponses appels ajax      #########################################
###############################################################################

from flask import request, jsonify


@app.route('/windy-extension')
def windy_extension():
    return render_template('windy_extension.html')






@app.route('/calcul_dash', methods=['POST'])
def calcul_dash():

    data = request.get_json()

    print("data :", data)
    type_message = data.get('type')
    course       = data.get('course')
    user_id      = data.get('user_id')
    Etat         = data.get('Etat')
    
    print('****************************************************************************************************************')
    print('****************************************************************************************************************')
    print('****************************************************************************************************************')
    print(r'****************************************************************************************************************')

    print("type    : ", type_message)
    print ("course : ",course)
    print('user_id : ',user_id)
    print("message :", Etat)
    print('****************************************************************************************************************')
    print('****************************************************************************************************************')
    # on calcule le tableau comme dans rechercheprogsvr2

    # ton calcul
    
    positionsLigneProj=rechercheligneproj(course,user_id,Etat)

    return jsonify({'course':course,'user_id':user_id,'resultat': positionsLigneProj  })



def rechercheligneproj(course,user_id,Etat):
    echouage   = 0                                   # on ne recherche pas les echouages 
    #cache = CarteCache()                            # on ne recherche pas les echouages   

    
    simulation                  = Simulateur (course, user_id)
    lat=Etat['lat']
    lon=Etat['lon']
    tws=Etat['tws']
    twd=Etat['twd']
    ts=Etat['ts']
    sail=Etat['sail']
    speed=Etat['speed']
    stamina=Etat['stamina']
    penoSail=Etat['penoSail']
    penoGybe=Etat['penoGybe']
    penoTack=Etat['penoTack']
    programmations=Etat['programmations']

    posDash = simulation.positionDash(lat,lon,twd,tws,ts,sail,speed,stamina,penoSail,penoGybe,penoTack,programmations)
    print ('posDash',posDash)
    print ('programmations',programmations )
    if "progs" in programmations:
        print ('\nOn est dans le cas de programmations\n')
        print ('programmations ',programmations )
        positions                   =simulation.trajectoire_24h( posDash, programmations, duree_h=24)

    if "wps" in programmations:  
        print ('\non est dans le cas de waypoints\n')
        # print ('\nposdepart\n',simulateur.posDepart)
        sections, routeprogs, positions = simulation.trajectoire_waypoints(  posDash, programmations, duree_h=24, dt_standard=60)  

    # if echouage==1:
    #     y0  =  positions[0,4]
    #     x0  =  positions[0,5]
    #     carte =  get_carte3(y0,x0)
    #     masque          = cache.get_masque_carte(y0, x0, get_carte3)
    #     touche          = cache.segments_touchent_masque(positions, masque, y0, x0)
    #     indices_croises = cache.indices_intersections_route(  positions, carte,touche, marge=0.1)
    #     positions[indices_croises + 1, 21] = 1


   # NB positions est un np.array
    positions_affichage= simulation.positions_pour_affichage( positions) 
    positionsLigneProj =[arr.tolist() for arr in positions_affichage]
    print ('positionsligneProj \n',positionsLigneProj[0])
    print (positionsLigneProj[1])
    print (positionsLigneProj[2])

    return positionsLigneProj



@app.route('/majgrib', methods=["GET", "POST"])
# recherche les coursesactives pour le username
def majgribajax():
    
    majgrib(basedirGribs025)   #GRGFS,GR_t,tigGFS,indiceGR se mettent à jour en global 
    response   = make_response(jsonify({'result':'maj grib faite '}))
    response.headers.add('Access-Control-Allow-Origin', '*')  # Autorise toutes les origines
    return response




@app.route('/recherchecoursesuser', methods=["GET", "POST"])
# recherche les coursesactives pour le username
def frecherchecoursesuser():
    username   = request.args.get('username')
    result     = rechercheTableCoursesActives(username)               # va chercher dans la table racesinfos
    response   = make_response(jsonify({'result':result}))
    print()
    print ('Coursesuser pour {} {}\n*****************************************************************************'.format (username,result ))
    print()
    response.headers.add('Access-Control-Allow-Origin', '*')  # Autorise toutes les origines
    return response




@app.route('/rechercheracesinfos', methods=["GET", "POST"])
def rechercheracesinfos():
    result   = rechercheTableRacesinfos()               # va chercher dans la table racesinfos
    response = make_response(jsonify({'result':result}))
    response.headers.add('Access-Control-Allow-Origin', '*')  # Autorise toutes les origines
    return response





@app.route('/recherchepersonalinfos', methods=["GET", "POST"])
def recherchepersonalinfos():
    username            = request.args.get('username')            # recupere les donnees correspondant a valeur dans le site html
    user_id             = request.args.get('user_id')
    course              = request.args.get('course')              # recupere les donnees correspondant a valeur dans le site html  


    
    infos=rechercheTablePersonalInfos (user_id,course)           # recupere directement un fichier texte    
    print()
    print ('Personalinfos récupérées  pour username {} course {} : \n \n'.format(username,course) )    # ici on affiche pas infos (trop long)
    print() 

    if infos is None:                           # on va creer un premier enregistrement pour l utilisateur sur la course specifiee 
       
        print ('pas d enregistrement dans personalinfos pour  {}  id {} sur course {} ',username,user_id,course  )
       
        # pour cela il nous faut aller rechercher dans la liste des courses comme chercheracesinfos 
        infotoutescoursesStr = rechercheTableRacesinfos()               # va chercher dans la table racesinfos
        print ('infotoutescoursesStr', infotoutescoursesStr)
        print('course',course)
        tabcourse            = extraitcourse (infotoutescoursesStr,course)
        print ('coursecomplete ',tabcourse )
        ari=['Arrivee']
        wp= {'Arrivee':[99,'Arrivee',tabcourse[5][0],tabcourse[5][1],tabcourse[5][4],'yellow'] }
        exclusions={}
        trajets={}
        timestamp=time.time()
        infos=json.dumps({"username":username,"course":course,'ari':ari,"wp":wp,"exclusions":exclusions,"trajets":trajets,"tolerancehvmg":0})
      

        conn   = pg_pool.getconn()
        cursor = conn.cursor()
        donnees= (timestamp,username,user_id,course,infos)
        
        majtable = "INSERT INTO personalinfos (timestamp,username,user_id,course,personalinfos ) VALUES (%s,%s,%s,%s,%s)"
        cursor.execute(majtable,donnees)
        conn.commit()
        cursor.close()
        pg_pool.putconn(conn) 
        
        print ('enregistrement créé pour {} sur course {} \n {} '.format(username,course,user_id,infos))       
     
        dico=json.dumps({"message":"Enregistrement créé pour utilisateur Inconnu",'infos':infos  })

    dico={'message':'Utilisateur connu , Infos renvoyées','infos':infos }
    return dico



@app.route('/modifpersonalinfos', methods=["POST"])
def modifpersonalinfos():
    data = request.get_json()
    if not data:
        return jsonify({"message": "Aucune donnée reçue"}), 400

    username   = data.get('username')
    user_id    = data.get('user_id')
    course     = data.get('course')
    typeinfo   = data.get('typeinfo')
    typeaction = data.get('typeaction')
    nom        = data.get('nom')
    valeur     = data.get('valeur')

    
    # print(type(valeur))          # → <class 'list'>

    # if typeinfo=='wp':
    #     elements = valeur.split(',')

    #     valeur = [
    #         int(elements[0]),
    #         elements[1],
    #         float(elements[2]),
    #         float(elements[3]),
    #         float(elements[4]),
    #         elements[5]
    #     ]
    #     print('valeur',valeur)
    


    # print ('on est dans modifpersonalinfos reception en POST ') 
    # print('user_id {} course  {} typeinfo {} typeaction {} nom  {} valeur  {} '.format( user_id, course, typeinfo,typeaction,nom,valeur))

    if not all([username,user_id, course, typeinfo, typeaction, nom]):
        return jsonify({"error": "Parametres Manquant"}), 400
    


    row=rechercheTablePersonalInfos(user_id,course)           # recupere directement un fichier texte
    if not row:
        return jsonify({"error": "User or course not found"}), 404
    infos = json.loads(row) if row else {}
    
    # print ('Ligne 3353 infos recuperees dans la table pour le user_id' ,infos )
    # print()

    if typeinfo not in infos:
        infos[typeinfo] = {}
    
    if typeaction == "insert":
        # print('1695 on est dans insert valeur ',valeur)
        if nom not in infos[typeinfo]:
           infos[typeinfo][nom] =valeur if valeur else 0
        #    print ('valeur de infos pour insertion' ,infos )
        #    print()
    
    
    elif typeaction == "delete":
        #   print ('ligne 2934  avant suppression  ',infos )
        #   print ('typeinfo a supprimer ',typeinfo)
        #   print ('nom de l info ',nom)
          infos[typeinfo].pop(nom, None)
        #   print('valeur de infos apres suppression ', infos )
        #   print()
        #   del infos[typeinfo][nom] 

    elif typeaction == "modify":

        print('on est dans modify')
        print ('typeinfo :',typeinfo)
        print ('valeur',valeur)

        if not valeur:
            return jsonify({"error": "Missing valeur for modification"}), 400
       
        
        if typeinfo == "tolerancehvmg":
            try:
                valeur = float(valeur)  # ou int(valeur) si tu veux un entier
            except ValueError:
                return jsonify({"error": "valeur invalide pour tolerancehvmg"}), 400
            infos["tolerancehvmg"] = valeur

        elif typeinfo=="ari":
            print ('On est bien dans le cas ari ')
            print ('ast.literal_eval(valeur) =',ast.literal_eval(valeur))
            infos["ari"] = ast.literal_eval(valeur)

        elif typeinfo=="tabwp":
            infos["wp"]    = ast.literal_eval(valeur)


        else:
            print ('on est dans ce cas')
            infos[typeinfo][nom] = json.loads(valeur)

        # print ('valeur de infos apres modification' ,infos )
        # print()

    else:
        return jsonify({"error": "Invalid typeaction"}), 400
    # print ('valeur json.dumps(infos) qui va etre enregistree \n',json.dumps(infos))

    # print()
    # print('test ', json.dumps(infos), username, course)
    # print()
    conn = pg_pool.getconn()
    cursor = conn.cursor()
    cursor.execute("""UPDATE personalinfos SET personalinfos = %s WHERE username = %s AND course = %s""", (json.dumps(infos), username, course))
    conn.commit()
    print ('La modification s est executee') 
    cursor.close()
    pg_pool.putconn(conn)  
    

    return jsonify({"message":'La modification de personalinfos s est deroulee avec succes ', "success": True,  "updated_infos": infos })
  



@app.route('/rechercheleginfos', methods=["GET", "POST"])
def rechercheleginfos():
    course              = request.args.get('course')              # recupere les donnees correspondant a valeur dans le site html  
    print('course',course)
    result   = rechercheTableLegInfos(course)               # va chercher dans la table racesinfos
    response = make_response(jsonify({'result':result}))
    response.headers.add('Access-Control-Allow-Origin', '*')  # Autorise toutes les origines
    return response







@app.route('/rechercheboatinfos', methods=["GET", "POST"])
def frechercheboatinfos():
    global GRGFS,tigGFS,indiceGR,heure
  
    username   = request.args.get('username')                # recupere les donnees correspondant a valeur dans le site html
    course     = request.args.get('course')                  # recupere les donnees correspondant a valeur dans le site html
    user_id    = request.args.get('user_id')                  # recupere les donnees correspondant a valeur dans le site html

    # Problème si le bateau n est pas identifié sur la course
    boatinfostr    = rechercheTableBoatInfos(user_id,course)
    boatinfos      = json.loads(boatinfostr)
  
    print('\n A {} Ligne 1175  Recuperation de boatinfos dans la base pour username {} user_id {} course {} boatinfostr :\n '.format(time.strftime(" %d %b %Y %H:%M ",time.localtime()),username,user_id,course ))  #,boatinfostr
    
    print()
    try:
        lastCalcDate = boatinfos['bs']['lastCalcDate']
    except:
        lastCalcDate = boatinfos['bs']['legStartDate'] 
                                 # la course n est pas partie 
    print ('\nEnvoi de  boatinfos recupere dans la base  du ',time.strftime(" %d %b %Y %H:%M ",time.localtime(lastCalcDate/1000)))


    majgrib(basedirGribs025) 
    indicemajgrib=indiceGR
    print()
    print ('Dernières valeurs de tigGFS:{}  indice {} soit +{}h '.format(time.strftime(" %d %b %H:%M ",time.gmtime(tigGFS)),indicemajgrib,indicemajgrib*3)) 
    
    # y0vr=boatinfos['bs']['lat']
    # print ('y0vr ',y0vr)
    # print ()

    heure = 99
    
   
    response   = make_response(jsonify({'result':boatinfostr,'tig':tigGFS,'heure':heure,'indicemajgrib':indicemajgrib}))
    response.headers.add('Access-Control-Allow-Origin', '*')  # Autorise toutes les origines
    return response




@app.route('/rechercheteam', methods=["GET", "POST"])
def frechercheteam():
    
    try: 
        teamname  = request.args.get('teamname')                # recupere les donnees correspondant a valeur dans le site html   
        ls1=rechercheTableTeammembers(teamname)
        liste = ast.literal_eval(ls1)
        
        print()
        print ('Team \n',liste)
        print()

    except:
        teamname=""
        liste=[]    

   
    # on renvoie le resultat 
    response = make_response(jsonify({'result':liste}))
    response.headers.add('Access-Control-Allow-Origin', '*')  # Autorise toutes les origines
    return response
    
    # print ('*****************************************************************')
    
    # return None 






def extraitprogrammations (boatInfos,user_id,course):
    # boatInfos,poSStart,positionvr=rechercheDonneesCourseUser( user_id,course)
    # print (boatinfos)

     #3) # on recupere les dernières valeurs de boatactions qui on tete stockees dans la base de donnees 
    timestamp,progstable           = rechercheTableProgsvr(user_id,course)                                  # progstablestr
    # print(progstable)

  
   
    
    # print()
    # print ('dernier enregistrement boatAction',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBA)))
    # print ('progstablestr ', progstablestr)
    # print()

    
    #######################################################################################################
    #1)  Etat et Position dans VR   pour calcul de progDepart 
    state       = boatInfos['bs']['state']
    heading     = boatInfos['bs']['heading']
    try :
        ts          = int(boatInfos['bs']['lastCalcDate']/1000)
    except :   
        ts          = int(boatInfos['bs']['legStartDate']/1000)  
    tsmn           = math.ceil(ts / 60) * 60  # Arrondi à la minute supérieure 
    lastCalcDateBI = ts                                              # si on n est pas parti la premeire prog est au moment du depart
    
    try:    
        twa         = boatInfos['bs']['twa']                       # c est la twa initiale, si on n'est pas parti pas de twa 
    except :
        twa = 45     #par defaut 
    try : 
        twaAuto     = boatInfos['bs']['twaAuto']
    except:     
        twaAuto     = twa              # il n'y a pas de twaauto au depart 
    # sail=boatInfos['bs']['sail']
    # voile=sail%10-1
    # voileAuto=False
    # if sail>10:
    #     voileAuto=True
    # print('infos statiques :\nheading {} twa {}  twaAuto {} voile {} voileAuto {}  '.format(heading,twa,twaAuto,voile,voileAuto) )     
    # d' ou la prog correspondant a l etat actuel 
    if twa!= twaAuto:
        option=0
        valeur=heading
    else:       
        option=1    
        valeur=twa   
    
    progDepart=[[ts,option,valeur,False]]   # c est la prog depart pour progs ( c'est la situation vr et le False indique que ce n'est pas une programmation mais l etat actuel ) 
    
    
    try:
        progstable                             = json.loads(progstable) 
        lastCalcDateBA=timestamp.timestamp()
    except:
        progstable=progDepart                   # s il n y a jamais eu de programmations sur cette course 
        lastCalcDateBA=time.time()-3600         # pour que la programmation prise en charge soit boatinfos  
        





    ###################################################################################################

    # on va reaorganiser car il peut y avoir une contradiction entre les 2 
    # print  ('lastCalcDateBA',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBA)),lastCalcDateBA)
    # print  ('lastCalcDateBI',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBI)),lastCalcDateBI)

    
    if (lastCalcDateBA>lastCalcDateBI-60 or state=='waiting') :           # c estle boat action  et donc progstable qui s applique 
        date    = lastCalcDateBA
        origine='Boatactions'
        
        if  "wps" in  progstable  :
            # print ('c\'est progtable issu de BoatActions qui s applique pour les wp')            
            origine ='Boatactions'
            progsBA=progstable['wps'] 
            # print ('progsBA',progsBA)
            programmations=json.dumps({'wps': progsBA})  

        else:
            # print ('c\'est progtable issu de BoatActions qui s applique pour les progs et qui existe ') 
            # print ('progstable ',progstable)
            try:
                progsBA=progstable['progs']
                if progsBA[0][3]!=False:
                    progsBA.insert(0, progDepart[0])  # Insère 1 à l'index 0
                    # print ('progsBA',progsBA)    
                programmationsBA=json.dumps({'progs': progsBA})  
                programmations=programmationsBA
            except:
                progsBA=  progDepart 
                programmationsBA=json.dumps({'progs': progsBA})  
                programmations=programmationsBA


    else:
        date=lastCalcDateBI    
        origine='BoatInfos' 

        if "waypoints" in boatInfos['bs']:
            programmations=json.dumps({'wps':boatInfos['bs']['waypoints']  })

        else:
            try:                                            # boatInfos['ba'] peut ne pas exister 
                boatInfosba=boatInfos['ba']   
                # print()
                # print ('boatinfosba',boatInfosba)
                # print()
                progsboatInfosba = [item for item in boatInfosba if item.get('_id', {}).get('action') == 'heading']
                progsInfosGlobales=condenseprogs(progsboatInfosba)
                if progsInfosGlobales[0][3]!=False:
                    progsInfosGlobales.insert(0, progDepart[0])  # Insère 1 à l'index 0  
                programmations=json.dumps({'progs': progsInfosGlobales})  
            
            except:
                print ("Il n'y a pas de boatinfos['ba'] on prend la programmation du depart progDepart")  
                origine='Pas de progs'  
                progsInfosGlobales=progDepart
                programmations=json.dumps({'progs': progsInfosGlobales})    

 
    programmations=json.loads(programmations)      # on transforme en objetjson
    return date,origine,programmations


def calculLigneProj(user_id,course,lat,lon,ts,programmations):

    simulateur                  = Simulateur (course, user_id)
    if "wps" in programmations:  
        print ('\non est dans le cas de waypoints\n')
        # print ('\nposdepart\n',simulateur.posDepart)
        sections, routeprogs, positions = simulateur.trajectoire_waypoints(  simulateur.posDepart, programmations, duree_h=24, dt_standard=60) 

    if "progs" in programmations:
            print ('\nOn est dans le cas de programmations\n')
            print ('programmations ',programmations )
            positions                   = simulateur.trajectoire_24h( simulateur.posDepart, programmations, duree_h=24)

    return positions        
        







@app.route('/rechercheprogsvr2', methods=["GET", "POST"])   
def rechercheprogsvr2():
    user_id    = request.args.get('user_id')                                 # recupere les donnees correspondant a valeur dans le site htmlstrf
    course     = request.args.get('course')  
    echouage   = int(request.args.get('echouage'))
    cache = CarteCache()

    print ('valeur de echouage = ',echouage) 
    simulateur                  = Simulateur (course, user_id)
    date,origine,programmations = simulateur.extraitprogrammations ()
    print ('programmations',programmations )
    if "progs" in programmations:
        print ('\nOn est dans le cas de programmations\n')
        print ('programmations ',programmations )
        positions                   = simulateur.trajectoire_24h( simulateur.posDepart, programmations, duree_h=24)

    if "wps" in programmations:  
        print ('\non est dans le cas de waypoints\n')
        # print ('\nposdepart\n',simulateur.posDepart)
        sections, routeprogs, positions = simulateur.trajectoire_waypoints(  simulateur.posDepart, programmations, duree_h=24, dt_standard=60)  

    if echouage==1:
        y0  =  positions[0,4]
        x0  =  positions[0,5]
        carte =  get_carte3(y0,x0)
        masque          = cache.get_masque_carte(y0, x0, get_carte3)
        touche          = cache.segments_touchent_masque(positions, masque, y0, x0)
        indices_croises = cache.indices_intersections_route(  positions, carte,touche, marge=0.1)
        positions[indices_croises + 1, 21] = 1


   # NB positions est un np.array
    positions_affichage= simulateur.positions_pour_affichage( positions) 

    routeprogs2 =[arr.tolist() for arr in positions_affichage]
    response=make_response(jsonify({'routeprogs2':routeprogs2,'programmations':programmations}))
    response.headers.add('Access-Control-Allow-Origin', '*')                # Autorise toutes les origines


    return response
    




@app.route('/rechercheprogsjaunes', methods=["GET", "POST"])   
# const url = serveur + "/rechercheprogsjaunes?user_id=" + user_id + "&course=" + course + "&programmations" + programmations+ "&echouage="+echouage
def rechercheprogsjaunes():
    user_id    = request.args.get('user_id')                                 # recupere les donnees correspondant a valeur dans le site htmlstrf
    course     = request.args.get('course')  
    echouage   = int(request.args.get('echouage'))
    progsjaunes = request.args.get('programmations') 
    
    progsjaunes=json.loads(progsjaunes)
    print ('valeur de echouage = ',echouage) 
    print ('programmationsjaunes ligne 3187 ')
    print (progsjaunes)
    print ('ligne 3189 ',progsjaunes['progs'][0])                      # position initiale 
    print()
    print()
    for i in range (len(progsjaunes['progs'])):
        print ('ligne 3193', time.strftime(" %d %b %H:%M ",time.localtime(progsjaunes['progs'][i][0])))
    
    simulateur                  = Simulateur (course, user_id)
    cache = CarteCache()

    if "progs" in progsjaunes:
        print ('\nOn est dans le cas de programmations\n')
        print ('programmationsjaunes ',progsjaunes )
        if progsjaunes['progs'][0][1]==1 :   # on est dans l option twa 
            simulateur.posDepart [1]=1
            simulateur.posDepart [2]=progsjaunes['progs'][0][2]                   # la valeur de la twa est la valeur de la 2eme prog

        positions                   = simulateur.trajectoire_24h( simulateur.posDepart, progsjaunes, duree_h=24)


    if echouage==1:
        y0  =  positions[0,4]
        x0  =  positions[0,5]
        carte =  get_carte3(y0,x0)
        masque          = cache.get_masque_carte(y0, x0, get_carte3)
        touche          = cache.segments_touchent_masque(positions, masque, y0, x0)
        indices_croises = cache.indices_intersections_route(  positions, carte,touche, marge=0.1)
        positions[indices_croises + 1, 21] = 1
  
    positions_affichage= simulateur.positions_pour_affichage( positions)           # NB positions est un np.array

    routeprogs2 =[arr.tolist() for arr in positions_affichage]
    response=make_response(jsonify({'routeprogs2':routeprogs2,'programmations':progsjaunes}))
    response.headers.add('Access-Control-Allow-Origin', '*')                # Autorise toutes les origines
    return response









# @app.route('/rechercheprogsvr', methods=["GET", "POST"])
# def rechercheprogsvr():
#     # print()
#     cache = CarteCache()
#     print ('On est dans rechercheprogsvr suite à la demande de rafraichissement generee par boataction ')
#     user_id    = request.args.get('user_id')                                 # recupere les donnees correspondant a valeur dans le site htmlstrf
#     course     = request.args.get('course')                                  # recupere les donnees correspondant a valeur dans le site
#     #t0routage      = int(float(request.args.get('t0routage')))
#     isMe='yes'
#     ari=['WP1']   # ne sert a rien dans le cas present mais est necessaire pour que ari soit defini

#     session         = RoutageSession(course, user_id,isMe,ari)        # permet de recuperer les polaires et autres 

#     positionvr      = session.positionvr
#     boatinfos       = session.boatinfos
#     positionvr_np   = positionvr.detach().cpu().numpy()
#     try:
#         lastCalcDateBI= int(boatinfos['bs']['lastCalcDate']/1000)
#     except:
#         lastCalcDateBI=time.time()

#     #print ('boatinfos dans rechercheprogsvr\n ',boatinfos)

#     date,origine,programmations = extraitprogrammations (boatinfos,user_id,course)      # va chercher dans la table 

#     #  print  ('lastCalcDateBA',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBA)),lastCalcDateBA)
#     # print  ('lastCalcDateBI',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBI)),lastCalcDateBI)
    

#     print('\n****************************************************************************************************************************************************************')
#     print ('Programmations retenues heure Boatinfos {} BAction  :{}  ,origine : {} ,programmations {}\n'.format(time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBI)),time.strftime(" %d %b %H:%M ",time.localtime(date)),origine,programmations))
#     print('******************************************************************************************************************************************************************')

#     Position_np =np.zeros((24*60,17),dtype=np.float64)   
#     Position_np[0]=positionvr_np

         
#     print()
    
#     t0=float(Position_np[0,1])  
#     dtig=t0-tigGFS  
#     twdcalc=float(Position_np[0,13])
#     twscalc=float(Position_np[0,14])

#     print ('Dand deplacement engine Point de depart ')
#     print ('fileName',fileName)
#     print ('ligne 3081 t0 du depart',time.strftime(" %d %b %Y %H:%M ",time.localtime(t0)) )
#     print ('tigGFS       UTC     ', time.strftime(" %d %b %Y %H:%M ",time.gmtime(tigGFS)))
#     print ('dtig en h              {:6.3f}  en indice {:6.3f}'.format( dtig/3600 ,dtig/3600/3))
#     print ('ligne 3236 twdcalcnp {:6.3f}  twscalcnp {:6.3f} '.format(twdcalc,twscalc))
#     print ('\n***************************************************')


#     # donneesCourse    = rechercheDonneesCourseCache(course)
    
#     engine = DeplacementEngine(course)      # c'est la classe qui calcule les deplacements 
#     sections = []

#     if "progs" in programmations:
#         typeprogs='progs'
#         print('On a affaire a des programmations')
#         tic=time.time()
#         prog0=programmations['progs']
#         prog1 = copy.deepcopy(prog0)
#         applied_prog = [False] * len(prog1)
        
#         Position_np[0, 3] = prog1[0][1]
#         Position_np[0, 4] = prog1[0][2]

#         applied_prog = [False] * len(prog1)

#         tic =time.time()
       
#         for i in range(len(Position_np) - 1):
#             t = Position_np[i, 1]  # temps courant            
#             for j in range(len(prog1)):
#                 if not applied_prog[j] and t >= prog1[j][0]:
#                     Position_np[i, 3] = prog1[j][1]  # option
#                     Position_np[i, 4] = prog1[j][2]  # valeur
#                     applied_prog[j] = True          # marquer comme appliquée

           
#             Position_np[i + 1] = engine.deplacement_np(Position_np[i])
#      # position_np=([0)0, 1)t0,2)dt,3)option,4)valeur,5)y0,6)x0,7)voile,8)twa,9)cap,10)speed,11)stamina,12)soldepeno,13)twd,14)tws,15)voileAuto,16)boost],dtype=torch.float64)'''   
    
   
            
#         routeprogs = Position_np[:,[1,3,4,5,6,7,8,9,10,13,14]]      # ensemble des positions 
#         routeprogs =[arr.tolist() for arr in routeprogs]
          

#         #print ('routeprogs ', routeprogs)  
#         # on transforme en tableau pour la transmission 
#         # impressionpointdep(Position_np)

        

#     if "wps" in programmations:
#         tic=time.time()
#         print ('programmations',programmations)
#         progswaypoints=programmations['wps']
#         typeprogs='wps'
#         print('On a affaire a des wps')
        
#         y0       = Position_np[0, 5]
#         x0       = Position_np[0, 6]
#         tini     = Position_np[0,1]         # issu de la position de depart
        

#         WP=[[round(y0,5),round(x0,5)]]     #Wp du point de depart
#         for elt in progswaypoints:
#             WP.append([elt['lat'],elt['lon']])

#         print ('tableau des wp avec le point de depart',WP )
#         capini   = calcul_cap_loxodromique    (WP[0][0], WP[0][1], WP[1][0], WP[1][1])
#         distance = distance_haversine         (WP[0][0], WP[0][1], WP[1][0], WP[1][1])
        
#         optionini = 0   # pour les waypoints on suit un cap constant 
#         valeurini = capini

#         sections = []
#         i = 0
#         idx          = 0  # Index dans Position_np
#         tempsrestant = 1000

       
#         print ('lenWP',len(WP))
        
#         tic=time.time()

        
#         print()
    
#         tic=time.time()
#         for j in range(len(WP) - 1):
#             section = { "y0": WP[0][0], "x0": WP[0][1],  "t_start": tini,"twa": Position_np[idx, 8], "cap": Position_np[idx, 9],"stamina": Position_np[idx, 11],  "voile": Position_np[idx, 7],\
#             "soldepeno": Position_np[idx, 12],"twd": Position_np[idx, 13], "tws": Position_np[idx, 14]      }
        
#             # on initialise le trajet *
#             lat0=WP[j][0]
#             lon0=WP[j][1]
#             lat1=WP[j+1][0]
#             lon1=WP[j+1][1]
#             cap =calcul_cap_loxodromique(lat0, lon0, lat1, lon1)  
            
#             distanceRestante = distance_haversine(WP[j][0], WP[j][1], WP[j+1][0], WP[j+1][1])        # C est la distance au depart
#             print ('WP depart {}  WP visé {}  ,capini {},distance {}'.format(WP[j],WP[j+1],cap,distanceRestante))
#             Position_np[idx, 3] =0     # option cap pur 
#             Position_np[idx, 4] = cap  # heading vers le nouveau wp
#             Position_np[idx, 2] = 60   # on reinitialise le dt 
#             tempsrestant=1000

#             while idx < Position_np.shape[0] - 1:
#                 # cap = calcul_cap_loxodromique_tensor(Positionto[idx][5], Positionto[idx][6], WP[j+1][0], WP[j+1][1])
#                 # Positionto[idx][4] = cap
            
#                 distWP = distance_haversine(Position_np[idx][5], Position_np[idx][6], WP[j+1][0], WP[j+1][1])
#                 speed = Position_np[idx][10]
            
#                 tempsrestant = (distWP / (speed + 1e-6)) * 3600  # évite la division par zéro
            
#                 if tempsrestant < 60:  # On approche du waypoint
#                     print(f'tempsrestant {tempsrestant:.1f}s, distance au WP {distWP:.3f} NM, vitesse {speed:.2f} kn')                    
#                     Position_np[idx, 2] = tempsrestant  # ajuste dt AVANT le dernier déplacement
#                     Position_np[idx + 1] = engine.deplacement_np(Position_np[idx])
#                     Position_np[idx + 1][2] =60
#                     print ('Position du WP suivant ', lat1,' ', lon1)
#                     Position_np[idx + 1][5] =lat1
#                     Position_np[idx + 1][6] =lon1
#                     # on va forcer la position 
#                     idx += 1
#                     break  # on termine ce tronçon 5 et  6 , si il y a changement de voile la derniere position n 'est pas bonne 
#                 else:
#                     Position_np[idx, 2] = 60  # sinon, dt standard
#                     Position_np[idx + 1] = engine.deplacement_np(Position_np[idx])
#                     idx += 1
#             section.update({
#                 "y1": Position_np[idx, 5],
#                 "x1": Position_np[idx, 6],
#                 "t_end": Position_np[idx, 1],
#                 "stam_end": Position_np[idx, 11],
#                 "voile_end": Position_np[idx, 7],
#                 "peno_end": Position_np[idx, 12],
#                 "twd_end": Position_np[idx, 13],
#                 "tws_end": Position_np[idx, 14],
#                 "twa_end": Position_np[idx, 8],
#                 "duree": Position_np[idx, 1] - section["t_start"]
#             })
            
#             sections.append(section)


#         Position_np=Position_np[Position_np[:, 0] != 0] 
#         routeprogs = Position_np[:,[1,3,4,5,6,7,8,9,10,13,14]] 
#         routeprogs =[arr.tolist() for arr in routeprogs]
#         # print('temps de calcul des programmations',time.time()-tic)

#         # impressionpointdep(Position_np)



#     polyline=[]
#     timestamp= int(time.time() )
   
#     #print ({'polyline':polyline,'origine':origine,'date':date, 'routeprogs':routeprogs,  'timestamp':timestamp,'typeprogs' :typeprogs, 'programmations':programmations,'sections':sections})

#     response=make_response(jsonify({'polyline':polyline,'origine':origine,'date':date, 'routeprogs':routeprogs,  'timestamp':timestamp,'typeprogs' :typeprogs, 'programmations':programmations,'sections':sections}))
#     response.headers.add('Access-Control-Allow-Origin', '*')                # Autorise toutes les origines
#     return response








@app.route('/recherchevoiles', methods=["GET", "POST"])
def recherchevoiles():
    # recupere la meteo sur le point les vmg et les recouvrements  


    y0vr     = float(request.args.get('y0'))
    x0vr     = float(request.args.get('x0'))
    try:
      t0vr     = float(request.args.get('t0'))               
    except:
      t0vr= time.time()  
    tws      = float(request.args.get('tws'))
    polar_id = int(request.args.get('polar_id'))
    twa      = float(request.args.get('twa'))
    # print('y0vr {}, x0vr {} ,t0vr {} ,twsvr {} polar_id {}  twa {}'.format(y0vr,x0vr,t0vr,tws,polar_id,twa ))

    twa10        = abs(round(twa*10))
    tws10        = int(round(tws*10))
    dtig         = t0vr-tigGFS
    twscalc,twdcalc=gfs_interpolateur_np(dtig,y0vr,x0vr)
    
    filenamelocal2='vmg10_'+str(polar_id)+'.npy'
    filename2=basedirnpy+filenamelocal2
    tabvmg10=get_fichier(filename2)
    
    filenamelocal1='polairesglobales10_'+str(polar_id)+'.npy'
    filename1=basedirnpy+filenamelocal1
    polairesglobales10=get_fichier(filename1)
    
    speedCalc=float(polairesglobales10[7,tws10,twa10])

    # print ('***************************************************')
    # print ('vitessemax',speedCalc)
    # print ('***************************************************')
       
    tabvmg       = tabvmg10[tws10]                        #on calcule les vmg pour tws
    tabvmg       = tabvmg.tolist()
    tabrecouvrements = plagevoiles2(polar_id,tws)
    tabrecouvrements = [arr.tolist() for arr in tabrecouvrements]

    response=make_response(jsonify({"twscalc":twscalc, "twdcalc":twdcalc,"tabvmg":tabvmg,"tabrecouvrements":tabrecouvrements,"vitessemax":speedCalc }))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response




###############################################################################
########### Reponses appels pages web #########################################
###############################################################################



@app.route("/", methods=["GET", "POST"])
def index():
    #global GRGFS 

    client    = request.remote_addr
    host      = request.host
    origine    = request.url_root
    print()
    print('Client {} host {} origin {}'.format(client,host,origine))

    try :                                                                                             # pour l instant ne sert a rien
        source                  = request.args.get('source')      
        posStart                = request.args.get('posStart')  
        posStart                = json.loads(posStart)
        print('sail : ',posStart['sail'])
    
    except:
        #print ('Pas de reception de valeurs par le dash \n')
        source='direct'
        posStart={"username": 'Takron-BSP','source':source}
        print('\nDemande accès direct au site par : ',origine)
        #print ('\nLigne 1808  Demande de routage direct sans transmission par l URL\n source : {} \n posStart :{}   \n '.format(source,posStart))

    return render_template("index.html",origine=origine,source=source,posStart=posStart)



@app.route("/windybase")
def windybase():
     return render_template("windybasique.html")










@app.route("/index", methods=["GET", "POST"])
def index3():
    '''SErt uniquement a donner les indications de base a index '''

    client     = request.remote_addr
    host       = request.host
    origine    = request.url_root
    print()
    print('Client {} host {} origin {}'.format(client,host,origine))

    try : 
        #si l origine de la demande est le dash il y a une source de definie et une posStart 
                                                                                                   # pour l instant ne sert a rien
        source                  = request.args.get('source')      
        posStart_raw            = request.args.get('posStart')  
        posStart = None
        if posStart_raw:
            try:
                # 1) Décoder les %xx en caractères normaux
                decoded = urllib.parse.unquote(posStart_raw)
                # 2) Charger le JSON en dict Python
                posStart = json.loads(decoded)
            except Exception as e:
                print("Erreur décodage posStart:", e)





        print()
        print ('username : ',posStart['username'])
        print ('sail :   : ',posStart['sail'])
    
    except:
        #print ('Pas de reception de valeurs par le dash  on indique que la source est "direct" \n')
        source='direct'
        posStart={"username": 'Inconnu car Accès direct)','source':source}
        print('\nDemande accès direct au site par : ',origine)
        #print ('\nLigne 1808  Demande de routage direct sans transmission par l URL\n source : {} \n posStart :{}   \n '.format(source,posStart))

    return render_template("index.html",origine=origine,source=source,posStart=posStart)






###############################################################################
########### Reception transmission  ajax du dashboard ########################
###############################################################################



@app.route('/api/ajaxmessage', methods=["POST"])
def ajaxmessage():
    global users
    # recupere les donnees du dashboard normalement ce sont des objets qu'il faut serialiser pour les sauver dans les bases
    # Sauve les donnees suivant leur nature dans les bases de donnees pour recuperation sur demande de index
    # a chaque fois que j ai un nouveau message les donnees des autres deviennent caduques
    try:
        conn = pg_pool.getconn()
        cursor = conn.cursor()

      
        data    = request.get_json()
        # message = data.get('message' ,'')
        message = data.get('message')
        typeinfos    = data.get('type')
        envoi = {'Reception':typeinfos}                              # sert a preciser le type d info que l on a recu pour envoi par websocket

        reponse ='Donnes recues par le serveur '  # reponse envoyee au dash pour test
        print ("\nType du Message reçu : {}".format(typeinfos) )     
        # print ('Message         : {} \n***********************************'.format( message))


        if typeinfos=='AccountDetailsRequest':
                #print ('Message         : {} \n***********************************'.format( data.get('message')))
                # print ('AccountDetailsRequest \n',message)
                timestamp        = time.time() # Convertit en timestamp (secondes Unix)
                username         = message['displayName']
                user_id          = message['userId']
                coursesactives   = message['scriptData']['currentLegs']
                coursesactives   = json.dumps(coursesactives)   

                print('coursesactives a enregistrer ',coursesactives )
                # print('userid          ',user_id)
                cursor.execute(""" INSERT INTO coursesactives (timestamp, username, user_id, coursesactives)
                VALUES (%s, %s, %s, %s) """, (timestamp, username, user_id, coursesactives))
                conn.commit()
                # verification de l enregistrement 
                result= rechercheTableCoursesActives(username)  
                print ('Pour username {} \ncourses Actives {}\n'.format(username, result ))
                print ('  Reception de  username {} user_id {} '.format(username,user_id))
                updateUsers(users, username, user_id)
                print('users',users)
    


                
                
        if typeinfos=='getracesinfos':
                
                timestamp = time.time()
                # print ('message racesinfos ligne 608  ', json.dumps(message))
                cursor.execute("""INSERT INTO racesinfos (timestamp, racesinfos) VALUES (%s,%s)""", (timestamp, json.dumps(message)))
                conn.commit()
                #print ('\nLes caracteristiques de toutes les courses ont ete enregistrees dans la base racesinfos ')
                # on va verifier la recuperation
                #result= rechercheTableRacesinfos()
                # print()  
                # print ('ligne 611 Verification de l enregistrement result', result )





        # on va tenter d enregistrer dans la base postgre boatinfos 
        if typeinfos=='boatinfos':
               # print('ligne 897 boatinfos \n ',message)
                # message est un objet il est necessaire de le serialiser si on veut l enregistrer dans la base
            try:
                username    = message['bs']['displayName']
                user_id     = message['bs']['_id']['user_id']
                race        = message['bs']['_id']['race_id']
                leg         = message['bs']['_id']['leg_num']
                course      = str(race)+'.'+str(leg)
                # state       = message['bs']['state']
                # tini0       = message['bs']['lastCalcDate']/1000


                updateUsers(users, username, user_id)
                print('users',users)
                # print ('\n boatinfos pour  {}  le  {} \n********************************************************\n {} \n'.format(username,time.strftime(" %d %b %Y %H:%M ",time.localtime(tini0)), message))
                # print()    
               

                # enregistrement dans la base
                timestamp   = time.time()            
                donnees=(timestamp,username,user_id,course,json.dumps(message)  )
                cursor.execute( """INSERT INTO boatinfos ("timestamp", username, user_id, course, boatinfos)   VALUES (%s, %s, %s, %s, %s)""",donnees  )      
                conn.commit()           
            
                # debuggage
                # on va verifier la recuperation
                # result= rechercheTableBoatInfos(user_id,course)
                # boatInfos=json.loads(result)
                # try:                                                     # erreur si la course n est pas partie 
                #     tini=boatInfos['bs']['lastCalcDate']/1000
                # except:
                #     tini=time.time()     
                # print ('\n Heure du  boatinfos enregistré ',time.strftime(" %d %b %Y %H:%M ",time.localtime(tini)))
                # on va recuperer boatinfos dans la bas epour voir si c'est bien enregistré
                # print ('ligne 936 Verification de l enregistrement boatinfos  \n ', boatInfos )
                # print()  

                # envoi de l info de la reception par websocket à user_id  
                envoi = {'Reception':typeinfos ,'course':course} 
                send_update_to_client( user_id, envoi)          # user_id est l identifiant et envoi les valeurs 
                print('****************************************************************************************************************') 
                print ('une demande d update suite à boatinfos a ete envoyée pour user id {} et course {}'.format (user_id,course))                     
                print('****************************************************************************************************************')  
                print()
            except:
                    print (' Les boatinfos ne contiennent pas bs ')
                    






        if typeinfos=='boatActions':
                # try:         # destine a couvrir si pas de prog dans boatActions
               
            boatActions = message['boatActions']

            #print ("Ligne 3024 message['boatActions']")
            print (boatActions)
            print()

            try:
                user_id     = boatActions[0]['_id']['user_id']
                race        = boatActions[0]['_id']['race_id']
                leg         = boatActions[0]['_id']['leg_num']
                course      = str(race)+'.'+str(leg)
                timestamp   = boatActions[0]['_id']['ts']/1000  # c'est l heure de depart de la course ne doit pas servir de reference pour l enregistrement 
            # timestamp   = time.time()           
            #timestamp =time.time()
              




                # try :
                        
                  

                #     # On va enregistrer betement dans boatactions 

                #     if any(  isinstance(item, dict)    and item.get('_id', {}).get('action') in ('heading', 'wp')   for item in boatActions):   
                #         print()
                #         print ('ligne 3601 enregistrement de action')   
                #         print ('user_id',user_id)   
                #         print ('course',course)
                #         print ('message',json.dumps(message))            
                #         print()        
                #         timestamp   = time.time()            
                #         donnees=(timestamp,user_id,course,json.dumps(message)  )
                #         cursor.execute( """INSERT INTO boatactions ("timestamp",  user_id, course, boatactions)   VALUES (%s, %s, %s, %s)""",donnees  )      
                #         conn.commit() 

                #         # envoi de l info de la reception par websocket à user_id  
                #         envoi = {'Reception':typeinfos ,'course':course} 
                #         send_update_to_client( user_id, envoi)          # user_id est l identifiant et envoi les valeurs                      
                #         print('****************************************************************************************************************')  
                #         print()
                    
                #     # debuggage
                #     #on va verifier la recuperation
                #     timestamp,result= rechercheTableBoatActions(user_id,course)
                #     boatActions=json.loads(result)
                #     print()
                #     print ('Verification enregistrement boatActions \n',boatActions)
                #     # try:                                                     # erreur si la course n est pas partie 
                #     #     tini=boatActions['bs']['lastCalcDate']/1000
                #     # except:
                #     #     tini=time.time()     
                #     # print ('\n Heure du  boatinfos enregistré ',time.strftime(" %d %b %Y %H:%M ",time.localtime(tini)))
                #     # on va recuperer boatinfos dans la bas epour voir si c'est bien enregistré
                #     # print ('ligne 936 Verification de l enregistrement boatinfos  \n ', boatInfos )
                #     # print()  

                # except:
                #     print ('PROBLEME ENREGISTREMENT DE BOATACTIONS DANS LA BASE\n**********************************************************')    





                # ancienne demarche meilleure, on evite les enregistrements parasites 

                # try :
                if (any(item.get('_id', {}).get('action') == 'heading' for item in boatActions)) :
                    # print ('Programmation detectee dans boat action')
                    progs = [item for item in boatActions if item.get('_id', {}).get('action') == 'heading']                                       
                    progsx= condenseprogs( progs)                                                             # Ce ne sont que les elements de boataction et ca ne reprend pas l etat actuel
                    print (progsx)                   
                    
                    
                    timestamp = datetime.now(timezone.utc)
                    # timestamp=int(time.time())
                    print ('heure d enregistrement',timestamp )    
                    progsx=json.dumps({"progs": progsx } ) 
                    
                
                    cursor.execute(""" INSERT INTO progsvr (user_id, course, timestamp, progsvr)  VALUES (%s, %s, %s, %s)""", (user_id, course, timestamp,json.dumps(progsx)))
                    conn.commit()

                    # #On va verifier que l enregistrement est bon 
                    timestamp,progstable  = rechercheTableProgsvr(user_id,course)
                    print ('ligne 3061 progstable ', progstable)
                    if progstable==progsx:
                        print ('\nl enregistrement des programmations est correct\n********************************************')
                    else:
                        print ('\n l enregistrement est incorrect \n********************************************')   


                if (any(item.get('_id', {}).get('action') == 'wp' for item in boatActions)):

                    print ('Programmation par WP  detectee dans boat action')
                    #print ('t0 ',time.strftime(" %d %b %H:%M %S",time.localtime(tiso.item())))
                    waypointsvr=boatActions[0]['pos']
                    print ('waypointsvr',waypointsvr)
                    # on va enregistrer les waypoints dans la base
                    strwaypointsvr=json.dumps({"wps":waypointsvr})
                    timestamp = datetime.now(timezone.utc)
                    print ()
                    print('test des valeurs avant enregistrement  ',user_id,course,timestamp,strwaypointsvr)
                    print ('strwaypointsvr',strwaypointsvr)
                    print ()  
                    cursor.execute("""INSERT INTO progsvr (user_id,course,timestamp,progsvr ) VALUES (%s,%s,%s,%s)""", (user_id,course,timestamp,json.dumps(strwaypointsvr)))
                    conn.commit()


                    # #On va verifier que l enregistrement est bon 
                    timestamp,progstable  = rechercheTableProgsvr(user_id,course)
                    print('progstable ',progstable)

                    if progstable==strwaypointsvr:
                        print ('\nl enregistrement  des waypoints est correct\n********************************************')

                    else:
                        print ('\n l enregistrement des waypoints est incorrect \n********************************************')  


                    # print ('\n\nLes Waypoints de {}  pour la course {}  ont ete enregistrees  dans la base progsvr à {}\n '.format(user_id ,course,time.strftime(" %d %b %Hh %Mmn ", time.localtime(timestamp))))
                    # print('******************************************************************************************')
                    # print ('Verification des programmations enregistrees  dans la base le {}  pour {}  sur la course {} {} '.format(time.strftime(" %d %b %Hh %Mmn ", time.localtime(timestamp)),user_id,course,strwaypointsvr))
                    # print()
                envoi = {'Reception par Vrouteur de ':typeinfos }
                print ('tentative d envoi au client  ' ,envoi )
                send_update_to_client( user_id, envoi)         # envoi par websocket de l info que l on a recu une actualisation  de boatActions 
                    
                # except:
                #     print ('On est dans le except de boatactions')
                
            except:
                print ('boatactions est probablement vide ')
            



        if typeinfos=='leginfos':
                print ('\nLigne 2548 leginfos ',message)
                        
                timestamp = time.time()
                race    = message['_id']['race_id']
                leg     = message['_id']['num']
                updated = message['_updatedAt']
                course  = str(race)+'.'+str(leg)
                cursor.execute("""INSERT INTO leginfos ("timestamp",course,leginfos) VALUES(%s,%s,%s) """,(timestamp,course,json.dumps(message)))
                conn.commit()
                #verification de l enregistrement 
                leginfos  = rechercheTableLegInfos(course)
                # print()
                # print ('Verification de l enregistrement de  leginfos pour la course {} \n {} '.format(course,leginfos))
                # print()
                # je n ai pas de user id donc je ne peux pas envoyer au client 
                # envoi={'Reception':typeinfos,'course':course}
                # send_update_to_client( user_id, envoi)         # envoi par websocket de l info que l on a recu une actualisation  probleme avec usere_id que l on a pas 

            


        if typeinfos=='polaires':
                # print('message polaires',message)
                _id=message['_id']
                label        = message['label']
                date_str     = message['_updatedAt']
                date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                timestamp=int(date_obj.timestamp())
                print()
                print ('_id', _id)
                print ('label ',label)
                print ('update ',date_str )
                print('date_obj',date_obj)
                print (timestamp)
                print('Derniere date de mise a jour {} pour la polaire  id {} de {}'.format( date_obj,_id,label))
                enregistrerPolaireSiPlusRecent(timestamp, _id, json.dumps(message))
                # verification de l enregistrement 
                # updated,polar_id,polaires=rechercheTablePolaires(_id)
                # print () 
                # print ('Verification de l enregistrement de polaires pour id {} mise a jour le  {} \n {}'.format(polar_id,updated,polaires))
                # print()





        if typeinfos=='teamList':
           # print ('Message         : {} \n***********************************'.format( message))           
            
            teamnames=[]                                                 # on va juste constituer un tableau avec les noms des membres de l equipe
            for i in range (len(message)):
                teamnames.append(message[i]['displayName'])
            timestamp=  time.time()  
            username='Takron-BSP'
            user_id='59c2706db395b292ed622d84'
            teamname='BSP'
            team_id='xxxxxx'
            teammembers=json.dumps(teamnames)    
            cursor.execute("""INSERT INTO teammembers ("timestamp",username,user_id,teamname,team_id,teammembers) VALUES(%s,%s,%s,%s,%s,%s) """,(timestamp,username,user_id,teamname,team_id,teammembers))
            conn.commit() 

            # Debuggage
            # print()
            # res1=rechercheTableTeammembers('BSP')
            # print()
            # print ('verification de l enregistrement teammembers res1',res1)
          

             

        if typeinfos =='track':
            print ('track reçue ')
            envoi = {'Reception':typeinfos }



        if typeinfos =='getfleet':
            
                #print(message)
                message2=json.loads(message)
                user_id=message2['user_id']
                course= message2['course']
                fleetinfos=message2['message']        
                fleet=[]
                for i in range (len(fleetinfos)):
                    try:
                        nom=fleetinfos[i]['displayName']
                    except:
                        nom='Inconnu'
                    categorie=fleetinfos[i]['type']
                    lat=fleetinfos[i]['pos']['lat']
                    lon=fleetinfos[i]['pos']['lon']
                    heading= fleetinfos[i]['heading']
                    speed =fleetinfos[i]['speed']
                    try:
                        tws =fleetinfos[i]['tws']
                    except:
                        tws=0
                    # twd=fleetinfos[i]['twd']
                    try:
                        twa=fleetinfos[i]['twa']
                    except:
                        twa=0
                    try:
                        sail =fleetinfos[i]['sail']
                    except:
                        sail=5
                    try:
                        team= fleetinfos[i]['team']
                    except: 
                        team=False   
                    #print ('nom {}\t categorie {} \t lat {:6.4f}  lon {:6.4f} heading {:6.2f} speed {:6.2f} twa {:6.2f}  tws{:6.2f}   team {} '.format (nom,categorie,lat,lon,heading,speed,twa,tws,team) )
            
                    fleet.append([nom, categorie,lat,lon,heading,speed,tws,twa,sail,team])
                # for i in range (len(fleet)):
                #     print (fleet[i] )

                # # on va sauver dans la base fleetinfos 
                timestamp=time.time()             

                cursor.execute('INSERT INTO fleetinfos (timestamp,user_id,course,fleetinfos ) VALUES (%s,%s,%s,%s)', (timestamp,user_id,course, json.dumps(fleet) ))
                conn.commit()
                envoi={'Reception':typeinfos,'course':course}
                send_update_to_client( user_id, envoi)         # envoi par websocket de l info que l on a recu une actualisation  de boatActions 
              
    finally:
        if cursor:
            cursor.close()
        if conn:
            pg_pool.putconn(conn)  # très important : rend la connexion au pool



    return jsonify({"resultat":reponse})



@app.route("/parametres", methods=["GET", "POST"])
def parametres():
    origine    = request.url_root
    course     = request.args.get('course') 
    polar_id   = request.args.get('polar_id') 
    tws        = request.args.get('tws')
    twa        = request.args.get('twa')
    voile      = request.args.get('voile')
    lat        = request.args.get('lat')
    lon        = request.args.get('lon')


    updated,polar_id,polairesjsonstr=rechercheTablePolaires(polar_id)   
    polairesjson=json.loads(polairesjsonstr) 
    # print('course',course)
    # print('polar_id',polar_id)
    # print('polairesjson',polairesjson)
    polairesjsonstr=json.dumps(polairesjson)
    # print ('polairesjsonstr \n',polairesjsonstr)
    # ce serait bien de ne pas rechercher les polaires a chaque fois qu 'il y a ue modif sur parametres 
 
    return render_template('parametres.html',origine=origine,course=course,polairesjson=polairesjson,tws=tws,twa=twa,voile=voile,lat=lat,lon=lon )




@app.route("/modiflocalstorage", methods=["GET", "POST"])
def modiflocalstorage():
    return render_template("modiflocalstorage.html")

@app.route("/modiflocalstorage2", methods=["GET", "POST"])
def modiflocalstorage2():
    return render_template("modiflocalstorage2.html")


@app.route("/gestionlocalstorage", methods=["GET", "POST"])
def gestionlocalstorage():
    return render_template("gestionlocalstorage.html")




###############################################################################
########### Fonctions de routage  et Classes           ########################
###############################################################################
# Classe routageSession
# Classe Simulateur
# Classe CarteCache 


class RoutageSession:                                                  # version avec nouvel algorithme

    
    def __init__(self, course, user_id, isMe ,ari):
        tic=time.time()
        self.device='cuda'
        self.dtype=torch.float32
        # Initialisation globale
        ########################
        self.n3=512
        self.isoglobal = torch.zeros((1200*self.n3,12), device= self.device, dtype=self.dtype)    # isoglobal est destiné a recevoir 850 isochrones avec leurs 512 points     
        self.kglobal   = 1
        self.next_npt = 1
        self.isoglobal_meta = []        
        self.course    = course
        self.user_id   = user_id
        self.ari       = ari
        
        # Chargement des données
        DonneesCourse1                                                      = rechercheDonneesCourseCache(course)   # c 'est un objet de la classe DonneesCourse
        self.boatinfos,self.posStartVR,self.exclusionsVR2  ,self.positionvr = rechercheDonneesCourseUser( user_id,course)                  # self.exclusionsVR2 est un dictionnaire avec les exclusions VR qui ne sert pas 
        self.personalinfos,self.ari,self.waypoints,self.exclusionsperso,self.barrieres,self.trajets,self.tolerancehvmg,retardpeno = rechercheDonneesPersoCourseUser( user_id,course)
     
        self.leginfos              = DonneesCourse1.leginfos
        self.exclusionsVR          = DonneesCourse1.tabexclusions                     # ce sont les exclusions VR
        self.tabicelimits          = DonneesCourse1.tabicelimits
        self.carabateau            = DonneesCourse1.carabateau
        self.polairesglobales10to  = DonneesCourse1.polaires_gpu
        self.tabvmg10to            = DonneesCourse1.vmg_gpu
        self.polaires_np           = DonneesCourse1.polaires_np                               #ce sont les polaires10
        self.tabvmg                = DonneesCourse1.vmg_cpu

        self.typevoile             = self.carabateau['typevoile']  
        # self.posStart ,self.isodep_record    = calculePosDepart(self.posStartVR,self.polaires_np,self.carabateau,dt=60)    # Position de depart du routage au bout de 60 s
        # self.isodepart             = calculeisodepart2(self.posStart)                                            # Transformation en iso de depart de la position  
        self.exclusionsVR.update(self.exclusionsperso)          # on rajoute exclusionsperso à exclusions VR 
       

        try:
            self.exclusions = {  nom: torch.tensor([[lon, lat] for lat, lon in coords], device='cuda')   for nom, coords in self.exclusionsVR.items()  }    
            self.exclusions_json = {  nom: tensor.cpu().tolist()  for nom, tensor in self.exclusions.items()}

        except : 
            self.exclusions = {}  
            self.exclusions_json = {}    
       


        self.segments             = prepare_segments(self.exclusions)  # [S, 4] segments [x1, y1, x2, y2]        
        self.t0vr                 = self.posStartVR['t0']
        self.MF                   = 0.8
        self.furler               = 0.8
        self.lw                   = self.carabateau["lws"]
        self.hw                   = self.carabateau["hws"]
        self.lwtimer              = self.carabateau["lwtimer"]
        self.hwtimer              = self.carabateau["hwtimer"]
        self.gybeprolwtimer       = self.carabateau['gybeprolwtimer']
        self.gybeprohwtimer       = self.carabateau['gybeprohwtimer']
        self.coeffboat            = self.carabateau['coeffboat']
         ### Pour Calcul meteo
        self.dtig0GFS               = self.t0vr-tigGFS          
        self.dtig0ECM               = self.t0vr-tigECM 
        self.ouverture =220                 # angle ouverture du routage 
        self.pas       =1                   # pas pour l'exploration des caps 
        self.cocheexclusions=1              # verification des zones exclusion 
        self.wp_passages = {}
        
      
    def calculePosDepart(self,dt=60): 
        ''' Version dans session'''
        ''' Donne l'iso de depart de depart au bout de 60 s'''
        ''' Sous la meme forme que posStartVR          '''
        
        state       = self.posStartVR['state']
        posStart    = copy.deepcopy(self.posStartVR)      # on est allé cherché posstartvr en debut de session 
        print ('On calculeposdepart')
        print ('posStart' ,posStart)  
      # Exemple Waiting  {'username': 'Sacron', 'race': 801, 'leg': 1, 'state': 'waiting', 'numisoini': 0, 'npt': 0, 'nptmere': 0, 'nptini': 0, 'tws': 15.699833869934082, 'twd': 26.41778564453125,
    #         #'twa': 28.41778564453125, 'twaAuto': True, 'y0': 15.41757, 'x0': -23.5411, 't0': 1778436000.0, 'heading': 358, 'rank': 'NA', 'speed': 0, 'voile': 0, 'voileAuto': False, 'stamina': 100, 'lastCalcDate': 1778436000000, 'tsEndOfSailChange': 1778436000000, 'tsEndOfGybe': 1778436000000, 'tsEndOfTack': 1778436000000, 'gateGroupCounters': [0, 0, 0, 0, 0, 0], 'penovr': 0.0, 'boost': 1}
      
 
        # Dans tous les cas 
        posStartVR=self.posStartVR     
        numisoini   = posStartVR['numisoini']
        npt         = posStartVR['npt']
        nptmere     = posStartVR['nptmere']
        nptini      = posStartVR['nptini']
        y0          = posStartVR['y0']
        x0          = posStartVR['x0']
        t0          = posStartVR['t0']
        sail        = posStartVR['voile']
        voileAuto   = posStartVR['voileAuto']
        tws         = posStartVR['tws']
        twd         = posStartVR['twd']
        twa         = posStartVR['twa']
        twaAuto     = posStartVR['twaAuto']
        cap         = posStartVR['heading']
        speed       = posStartVR['speed']
        stamina     = posStartVR['stamina']
        soldepeno   = posStartVR['penovr']        
        boost       = posStartVR['boost']    
        
       
        if sail>10:
                voileAuto = 10
                voile     = sail%10
        else :
                voileAuto = 0
                voile     = sail

        print ('state ',state)

        if state == 'waiting'  :
            # Les valeurs de vent ont deja ete calculees dans posStartVR , donc il n y a rien a faire    
            dt = 0
            isodepart       = torch.tensor ([[0,0,y0,x0,voile,twa,stamina,soldepeno,tws,twd]], dtype=torch.float32, device='cuda')


        
       
        if state != 'waiting'  :  
            lastCalcDate= posStartVR['lastCalcDate'] 
            
           
            if twaAuto==twa:                  # navigation au cap 
                cap= fcap(twa,twd)
            else :                            # navigation a la twa 
                twa =ftwaos(cap, twd)     
            
    #       # Pénalités pour le deplacement , on suppose pas de manoeuvre tackgybe ou changement de voile se declenchat toute seule dans les 60 sdeclenchee dans les 60s 
            # d ou pas de perte de stamina non plus seulement recup 
    
            peno_fin    = max(0,soldepeno-dt)    
    #       # Stamina
      
            recup = pts_recuperes(dt, tws)
            stamina = stamina + recup
            staminafin   =  max (0.0, min(100.0, stamina))
        
    #         # Temps effectif après pénalités pas de penalite 
            dt_ite = dt                                                 # - 0.3 * max(peno_globale , dt)
        
            # Coordonnées après déplacement
            cap_rad = cap *math.pi/180
            y0_rad  = y0  *math.pi/180
            dlat    = speed * dt_ite / 3600 / 60 * math.cos(cap_rad)                                       # on suppose que la vitesse est constante 
            dlon    = speed * dt_ite / 3600 / 60 * math.sin(cap_rad) / math.cos(y0_rad)
            y1      = y0   + dlat
            x1      = x0   + dlon
            dtig=   t0+dt-tigGFS
            twsf,twdf=gfs_interpolateur_to(dtig,y1, x1)       
          
    
            isodepart       = torch.tensor ([[0,0,y1,x1,voile,twa,stamina,soldepeno,twsf,twdf]], dtype=torch.float32, device='cuda')  # transforme tout en torch d'un coup 
            # Modele iso 
            # 0 npt : numero du point
            # 1 nptmere : numero du point dont est issu le point
            # 2 lat : latitude du point
            # 3 lon : longitude du point
            # 4 voile : voile du bateau à l arrivee sur ce point
            # 5 twa_ant : twa du deplacement precedent
            # 6 stamina_ant : energie à l'arrivee sur ce point
            # 7 peno_ant : penalite à l'arrivee sur ce point
            # 8 tws : vitesse du vent sur ce point
            # 9 twd : direction du vent sur ce point
            
        iso_record_depart = {
                "numiso": 0,
                "ecart" : dt,                  # secondes ou minutes depuis t0vr
                "points": isodepart,           # tenseur uniquement avec les colonnes utiles par point
                "tmini" : 10000,
                "distmini": 100,
                "nptmini": 0,
              }
            
       
        return iso_record_depart  
      

    
    def initialiseRoutagePartiel(self, indiceroutage,iso_record_depart):
        ''' fonction de classe '''
        '''Initialisation des données nécessaires au routage pour chaque waypoint'''
        ''' on cherche juste a construire  l iso de depart et les donnees du routage partielrecuper juste l'isochrone de depart '''       
         # iso_record_depart = {
         #                "numiso": 0,
         #                "ecart" : dt,                  # secondes ou minutes depuis t0vr
         #                "points": isodepart,           # tenseur uniquement avec les colonnes utiles par point
         #                "tmini" : 10000,
         #                "distmini": 100,
         #                "nptmini": 0,
         #               "idxptmini": 0
         #              }

        
        # determination du point d arrivee         
        waypoint = next((ligne for ligne in self.waypoints if ligne[1] == self.ari[indiceroutage]), None)    # on extrait le waypoint correspondant à ari[indiceroutage]
        y1          = waypoint[2]
        x1          = waypoint[3]
        rwp         = waypoint[4]
        if rwp==0:            # probleme des lignes d arrivee a gerer ulterieurement
           rwp = 0.2


        # determination de l isochrone de depart    
        if indiceroutage==0:                     # c'est le debut du routage , on utilise la position de vr et on la decale de 60 s par defaut             
            numisoini   = iso_record_depart['numiso'] 
            ecart       = iso_record_depart['ecart'] 
            iso         = iso_record_depart['points'] 
            tmini       = iso_record_depart['tmini'] 
            nptmini     = iso_record_depart['nptmini']       #nptmini , c'est le pointle plus proche qui nous interesse  
            y0          = iso[0,2].item()
            x0          = iso[0,3].item()
           
            seuils = [[180,60],[144, 300],[108,600], [672, 1800], [240, 3600]]       # 14h a 5 mn = 120 --  18h a 10 mn =108 -- 14 jours a 30mn =24*2*14=672
            tabdt  = construire_dt(seuils, taille=1000)
            self.isoglobal[0,0:10] = iso[0,0:10]                  # il est necessaire de remplir le premier terme de isoglobal uniquement pour le premier tronçon 



        else :                               # ce n'est pas le debut du routage , il faut recuperer le dernier point pour avoir le dernier iso  

            # print('iso_record_depart',iso_record_depart)

            numisoini   = iso_record_depart['numiso'] 
            ecart       = iso_record_depart['ecart'] 
            points      = iso_record_depart['points'] 
            tmini       = iso_record_depart['tmini'] 
            nptmini     = iso_record_depart['nptmini']       #nptmini , c'est le pointle plus proche qui nous interesse  
            nptini      = iso_record_depart['idxptmini']
         
            ptplusproche      = self.isoglobal[nptini] , # c'est le point du dernier isochrone le plus proche de l arrivee              
            iso = ptplusproche[0].unsqueeze(0)                                  # l iso n a qu'un point on ajoute une dimension  
            y0                = iso[0,2].item()
            x0                = iso[0,3].item()
            
            seuils = [[30, 60], [20, 600], [10, 1800]]
            tabdt = construire_dt(seuils, taille=1000)

        
        # Dans tous les cas        
        dy = y1 - y0
        dx = dlon_short(x1, x0)         # delta longitude court  # en degrés, dans [-180, +180]        
        x1u = x0 + dx                   # longitude "déroulée" de l'arrivée, cohérente avec le départ

        # coordonnées de départ et arrivée sous forme torch
        depto = torch.tensor([y0, wrap360(x0)], device=self.device, dtype=self.dtype)
        arito = torch.tensor([y1, wrap360(x1)], device=self.device, dtype=self.dtype)
        
        cap_ar = round((450.0 - math.degrees(math.atan2(dy, dx))) % 360.0)
        range_caps = torch.arange(  cap_ar - self.ouverture / 2, cap_ar + self.ouverture/2, self.pas, device=self.device,dtype=self.dtype ) % 360.0  
        range_capsR = torch.deg2rad(range_caps)
        eps = 1e-12
        if abs(dx) < eps:
            m_ar = float('inf')
        else:
            m_ar = dy / dx

       # centre du cercle englobant le routage
        centre_y = (y0 + y1) / 2.0
        centre_x = wrap360((x0 + x1u) / 2.0)
        centre = [centre_y, centre_x]
      
        # rayon "plan" en degrés (si tu veux juste une bbox grossière)
        rayon = (math.hypot(dy, dx) / 2.0) * 1.10
    
        deptoR      = torch.deg2rad(depto)
        aritoR      = torch.deg2rad(arito)
        dti         = 60 

        distDepAri = dist_mn_vec(deptoR[0], deptoR[1], aritoR[0], aritoR[1])      #distance depart arrivee en MN         
        rayonRoutage = distDepAri / 2.0
        centreRoutage = torch.tensor( [ (y0 + y1) / 2.0, (x0 + x1) / 2.0 ], device='cuda', dtype=torch.float32 )
        centreRoutageR = torch.deg2rad(centreRoutage)

        self.indiceroutage=indiceroutage
        self.numisoini = numisoini
        self.deptoR =  deptoR
        self.aritoR =  aritoR
   
        self.lat_depart_rad = torch.tensor( math.radians(y0), device='cuda', dtype=torch.float32)
        self.lon_depart_rad = torch.tensor( math.radians(wrap180(x0)), device='cuda', dtype=torch.float32)        
        self.lat_depart=y0
        self.lon_depart=x0
        
        self.y0 =  y0
        self.x0 =  x0
        self.y1 =  y1
        self.x1 =  x1

        self.lat_arrivee_rad = torch.tensor( math.radians(y1), device='cuda', dtype=torch.float32)
        self.lon_arrivee_rad = torch.tensor( math.radians(wrap180(x1)), device='cuda', dtype=torch.float32)  
        
        self.rwp                      =  rwp
        self.dti                      =  dti
        self.cap_ar                   =  cap_ar
        self.range_caps               =  range_caps
        self.range_capsR              =  range_capsR
        self.m_ar                     =  m_ar
        self.dx_ar                    = dlon_short(x1, x0)
        self.dy_ar                    = y1 - y0
        self.rayonRoutage             =  rayonRoutage
        self.distDepAri               =  distDepAri
        self.cocheexclusions          =  1
        self.dtglobal                 = tabdt                                  # c est le tableau des dt 
        #self.retardpeno               = retardpeno
        self.latcR, self.loncR        = centreRoutageR
        self.latc, self.lonc          = centreRoutage
        self.nmax                     = 512
        self.p                        = self.range_caps.shape[0] 
        self.trajets_buf              = torch.empty((self.p * self.nmax, 16), device=self.device,dtype=torch.float32 )     

        return
    
    def fractions_controle_terre(self,dt):
        dt_s = float(dt.item()) if torch.is_tensor(dt) else float(dt)
        if dt_s <= 60:
            return (0.25,0.5,0.75)
        elif dt_s <= 300:
            return (0.2, 0.4, 0.6, 0.8)
        elif dt_s <= 1800:
            return (0.25, 0.5, 0.75)
        else:
            # tous les 10 %
            return tuple(i / 10 for i in range(1, 10))
        


        
    @torch.no_grad()
    def isoplusun(self, iso_record, mode):
        # -----------------------------
        # Colonnes isochrone
        # -----------------------------
    
        COL_NPT      = 0
        COL_NPTMERE  = 1
        COL_LAT      = 2
        COL_LON      = 3
        COL_VOILE    = 4
        COL_TWA_ANT  = 5
        COL_STAMINA  = 6
        COL_PENO     = 7
        COL_TWS      = 8
        COL_TWD      = 9
        
    
        # -----------------------------
        # Colonnes supplémentaires trajet
        # -----------------------------
        COL_MEMO     = 10
        COL_CAP      = 11
        COL_TWA      = 12
        COL_SPEED    = 13
        COL_LAT_AR   = 14
        COL_LON_AR   = 15
    
        # -----------------------------
        # Réutilisation mémoire temporaire
        # -----------------------------
        COL_PROJ     = 8
        COL_DIST_AR  = 9
        
        device = self.device
        dtype  = self.dtype

        empty_iso = torch.empty((0, 11), device=device, dtype=dtype)
        numisoprecedent = iso_record['numiso']
        iso             = iso_record['points']
        ecart_precedent = iso_record['ecart']             # ecart par rapport à t0vr
        tmini           = iso_record['tmini']
        nptmini         = iso_record['nptmini']
        
        n = iso.shape[0]

        if n == 0:                     # cas ou l isochrone precedent etait vide 
            iso_record = {
                "numiso": numisoprecedent,
                "ecart": ecart_precedent,          # secondes ou minutes depuis t0
                "points": empty_iso,           # tenseur uniquement avec les colonnes utiles par point
                "tmini": 0,
                "distmini": 0,
                "nptmini": nptmini,
                "idxptmini": 0
              }
            
            return iso_record
  
        numiso  = numisoprecedent + 1    
        ordre   = numiso - self.numisoini - 1         
    
        if tmini > 3600:
            dt = self.dtglobal[ordre]               # tensor scalaire
        else:
            dt = torch.tensor(60.0, dtype=torch.float32, device=device)
    
        ecart = ecart_precedent + dt                      # ecart par rapport à t0vr
        ecart_val = float(ecart.item())
    
        # constantes scalaires réutilisées
        inv_3600_60    = 1.0 / 216000.0
        bonus          = 0.0   #0.6
        memo_reset     = 12.0
        seuil_min_dist = torch.tensor(0.5, device=device, dtype=dtype)

    
        # ---------------------------------------------------------
        # Construction des trajets candidats
        # ---------------------------------------------------------
        m              = n * self.p
        trajets        = self.trajets_buf[:m]
        
        trajets3d      = trajets.view(self.p, n, 16)                            # trajet 3 d ne comprend que 16 colonnes 
        trajets3d[:, :, :iso.shape[1]] = iso.unsqueeze(0)
        trajets3d[:, :, COL_CAP] = self.range_caps[:, None]
    
        # décrément pénalité restante
        trajets3d[:, :, COL_PENO].sub_(dt).clamp_(min=0.0)
    
        # TWA
        trajets3d[:, :, COL_TWA]     = ftwao( self.range_caps[:, None], trajets3d[:, :, COL_TWD] )
        trajets3d[:, :, COL_NPTMERE] = trajets3d[:, :, COL_NPT] 
        tws                          = trajets3d[:, :, COL_TWS]
        twa                          = trajets3d[:, :, COL_TWA]
        tws10 = torch.round(tws * 10).to(torch.long).clamp_(0, 699)
    
        # ---------------------------------------------------------
        # Filtre VMG
        # ---------------------------------------------------------

        twa_abs = twa.abs()
        twa_min = self.tabvmg10to[tws10, 2] - self.tolerancehvmg
        twa_max = self.tabvmg10to[tws10, 4] + self.tolerancehvmg
        mask    = (twa_abs >= twa_min) & (twa_abs <= twa_max)
        trajets_valid = trajets3d[mask]

        if trajets_valid.shape[0] == 0 :
                iso_record = {
                "numiso": numisoprecedent,
                "ecart": ecart_precedent,          # secondes ou minutes depuis t0
                "points": empty_iso,           # tenseur uniquement avec les colonnes utiles par point
                "tmini": 0,
                "distmini": 0,
                "nptmini": nptmini,
                "idxptmini": 0
              }
            
                return iso_record
        
        # vues locales
        tws_valid       = trajets_valid[:, COL_TWS]
        twa_ant_valid   = trajets_valid[:, COL_TWA_ANT]
        twa_valid       = trajets_valid[:, COL_TWA]
        stamina_valid   = trajets_valid[:, COL_STAMINA]
        voile_ini_valid = trajets_valid[:, COL_VOILE].to(torch.long)
    
        tws10_valid = torch.round(tws_valid * 10).to(torch.long).clamp_(0, 699)
        twa10_valid = torch.round(twa_valid.abs() * 10).to(torch.long).clamp_(0, 1799)

        
    
        # ---------------------------------------------------------
        # Voile / vitesse / changement
        # ---------------------------------------------------------
        pol = self.polairesglobales10to
        vitessevoileini_valid = pol[voile_ini_valid, tws10_valid, twa10_valid]
        vitessemaxi_valid = pol[7, tws10_valid, twa10_valid]
        meilleurevoile_valid = pol[8, tws10_valid, twa10_valid].to(torch.long)
    
        boost = vitessemaxi_valid / (vitessevoileini_valid + 1e-4)
        chgt = boost > 1.014
        nouvelle_voile_valid = torch.where(chgt, meilleurevoile_valid, voile_ini_valid)
    
        trajets_valid[:, COL_VOILE] = nouvelle_voile_valid.to(dtype)
        trajets_valid[:, COL_SPEED] = vitessemaxi_valid
    
        tackgybe = (twa_ant_valid * twa_valid) < 0
        trajets_valid[:, COL_TWA_ANT] = twa_valid
    
        # MEMO avant filtrages
        memo_col = trajets_valid[:, COL_MEMO]
        memo_col.sub_(1.0).clamp_(min=0.0)
        memo_col[chgt | tackgybe] = memo_reset
    
        # ---------------------------------------------------------
        # Pénalités
        # ---------------------------------------------------------
        penalite_chgt = torch.zeros_like(tws_valid)
        penalite_tackgybe = torch.zeros_like(tws_valid)
    
        if chgt.any():
            penalite_chgt[chgt] = (
                peno_torch(
                    self.lwtimer,
                    self.hwtimer,
                    tws_valid[chgt],
                    stamina_valid[chgt]
                ) * self.furler
            )
    
        if tackgybe.any():
            penalite_tackgybe[tackgybe] = peno_torch(
                self.gybeprolwtimer,
                self.gybeprohwtimer,
                tws_valid[tackgybe],
                stamina_valid[tackgybe]
            )
    
        trajets_valid[:, COL_PENO].add_(penalite_chgt + penalite_tackgybe)
    
        peno_active = torch.clamp(trajets_valid[:, COL_PENO], max=dt)
        dt_penalise = torch.clamp(dt - 0.3 * peno_active, min=0.0)

        
    
        # ---------------------------------------------------------
        # Déplacement
        # ---------------------------------------------------------
        cap_rad = torch.deg2rad(trajets_valid[:, COL_CAP])
        lat_rad = torch.deg2rad(trajets_valid[:, COL_LAT])
    
        cos_cap = torch.cos(cap_rad)
        sin_cap = torch.sin(cap_rad)
        coslat  = torch.cos(lat_rad).clamp_min_(1e-6)
    
        speed = trajets_valid[:, COL_SPEED]
        
        facteur = speed * dt_penalise * inv_3600_60
        # print ('facteur',facteur)
        # print('sin_cap',sin_cap)
        
        trajets_valid[:, COL_LAT_AR] = trajets_valid[:, COL_LAT] + facteur * cos_cap
        trajets_valid[:, COL_LON_AR] = trajets_valid[:, COL_LON] + facteur * sin_cap / coslat
    
        # ---------------------------------------------------------
        # Stamina
        # ---------------------------------------------------------
        trajets_valid[:, COL_STAMINA].add_(
            -calc_perte_stamina_to(tws_valid, tackgybe, chgt, self.coeffboat)
            + frecupstaminato(dt, tws_valid)
        )
        trajets_valid[:, COL_STAMINA].clamp_(min=0.0, max=100.0)
    
        # ---------------------------------------------------------
        # Filtre cercle de routage
        # ---------------------------------------------------------
        lat_ar = trajets_valid[:, COL_LAT_AR]
        lon_ar = trajets_valid[:, COL_LON_AR]
    
        lat_pt_rad = torch.deg2rad(lat_ar)
        lon_pt_rad = torch.deg2rad(lon_ar)
    
        distcentre = dist_mn_vec(lat_pt_rad, lon_pt_rad, self.latcR, self.loncR)
        mask_centre = distcentre < (self.rayonRoutage * 1.15)
    
        trajets_valid = trajets_valid[mask_centre]

        if trajets_valid.shape[0] == 0 :
                iso_record = {
                "numiso": numisoprecedent,
                "ecart": ecart_precedent,          # secondes ou minutes depuis t0
                "points": empty_iso,           # tenseur uniquement avec les colonnes utiles par point
                "tmini": 0,
                "distmini": 0,
                "nptmini": nptmini,
                "idxptmini": 0
              }
            
                return iso_record
       
        # vues réactualisées
        lat_ar = trajets_valid[:, COL_LAT_AR]
        lon_ar = trajets_valid[:, COL_LON_AR]
        lat_pt_rad = torch.deg2rad(lat_ar)
        lon_pt_rad = torch.deg2rad(lon_ar)

        # print("après filtre centre", trajets_valid.shape[0])
    
        # ---------------------------------------------------------
        # Réutilisation mémoire : colonnes 9 et 10
        # ---------------------------------------------------------
        lon_proj = dlon_short_torch(lon_ar, self.lon_depart)
        trajets_valid[:, COL_PROJ] = self.dx_ar * lat_ar - self.dy_ar * lon_proj
        trajets_valid[:, COL_DIST_AR] = dist_mn_vec(
            lat_pt_rad,
            lon_pt_rad,
            self.lat_arrivee_rad,
            self.lon_arrivee_rad
        )
    
        proj = trajets_valid[:, COL_PROJ]
        dist_col = trajets_valid[:, COL_DIST_AR]
        memo_manoeuvre = trajets_valid[:, COL_MEMO]
        score = dist_col - bonus * (memo_manoeuvre / 12.0)
    
        ordomini = proj.min()
        ordomaxi = proj.max()
    
        # ---------------------------------------------------------
        # Réduction par buckets
        # ---------------------------------------------------------
        if torch.isclose(ordomaxi, ordomini):
            idx = torch.argsort(score, stable=True)
            trajets_valid = trajets_valid[idx[:1]]
        else:
            coeff = (self.nmax - 1) / (ordomaxi - ordomini + 1e-12)
            buckets = torch.round((proj - ordomini) * coeff).to(torch.long)
            buckets.clamp_(0, self.nmax - 1)
    
            idx = torch.argsort(score, stable=True)
            trajets_valid = trajets_valid[idx]
            buckets = buckets[idx]
    
            idx = torch.argsort(buckets, stable=True)
            trajets_valid = trajets_valid[idx]
            buckets = buckets[idx]
    
            keep = torch.ones_like(buckets, dtype=torch.bool)
            keep[1:] = buckets[1:] != buckets[:-1]
            trajets_valid = trajets_valid[keep]
            
    
        if trajets_valid.shape[0] == 0 :
                iso_record = {
                "numiso": numisoprecedent,
                "ecart": ecart_precedent,          # secondes ou minutes depuis t0
                "points": empty_iso,           # tenseur uniquement avec les colonnes utiles par point
                "tmini": 0,
                "distmini": 0,
                "nptmini": nptmini,
                "idxptmini": 0
              }
            
                return iso_record
                    
        # print("après buckets", trajets_valid.shape[0])

        # impression_trajets (trajets_valid,titre='Trajet ')
    
        # # ---------------------------------------------------------
        # # Terre sur point d'arrivée
        # # ---------------------------------------------------------
        # mask_mer = terremer(
        #     trajets_valid[:, COL_LAT_AR],
        #     trajets_valid[:, COL_LON_AR]
        # ) != 0
    
        # trajets_valid = trajets_valid[mask_mer]
        # if trajets_valid.shape[0] == 0:

        #     iso_record = {
        #         "numiso": numisoprecedent,
        #         "ecart": ecart_precedent,          # secondes ou minutes depuis t0
        #         "points": empty_iso,           # tenseur uniquement avec les colonnes utiles par point
        #         "tmini": 0,
        #         "distmini": 0,
        #         "nptmini": nptmini,
        #         "idxptmini": 0
        #       }
        #     return iso_record
    
        # ---------------------------------------------------------
        # Terre sur segments +point arrivee 
        # ---------------------------------------------------------
        lat0 = trajets_valid[:, COL_LAT]
        lon0 = trajets_valid[:, COL_LON]
        lat1 = trajets_valid[:, COL_LAT_AR]
        lon1 = trajets_valid[:, COL_LON_AR]

        dlat = lat1 - lat0
        dlon = dlon_short_torch(lon1, lon0)

        mask_mer = torch.ones(
            trajets_valid.shape[0],
            dtype=torch.bool,
            device=trajets_valid.device
        )

        for f in (*self.fractions_controle_terre(dt), 1.0):
            lat_t = lat0 + f * dlat
            lon_t = wrap180_t(lon0 + f * dlon)
            mask_mer &= (terremer(lat_t, lon_t) != 0)

        trajets_valid = trajets_valid[mask_mer]

        if trajets_valid.shape[0] == 0:
            iso_record = {
                "numiso": numisoprecedent,
                "ecart": ecart_precedent,
                "points": empty_iso,
                "tmini": 0,
                "distmini": 0,
                "nptmini": nptmini,
                "idxptmini": 0
            }
            return iso_record
          
        # print("après terre segment", trajets_valid.shape[0])    
    
        # ---------------------------------------------------------
        # Zones d'exclusion
        # ---------------------------------------------------------
        maskexclu = points_in_any_polygon_vectorized(
            trajets_valid[:, COL_LAT_AR],
            trajets_valid[:, COL_LON_AR],
            self.segments
        )
        trajets_valid = trajets_valid[~maskexclu]
        if trajets_valid.shape[0] == 0:
             iso_record = {
                "numiso": numisoprecedent,
                "ecart": ecart_precedent,          # secondes ou minutes depuis t0
                "points": empty_iso,           # tenseur uniquement avec les colonnes utiles par point
                "tmini": 0,
                "distmini": 0,
                "nptmini": nptmini,
                "idxptmini": 0
              }
            
             return iso_record
          
        # print("après exclusions", trajets_valid.shape[0])    
        # ---------------------------------------------------------
        # Barrières
        # ---------------------------------------------------------
        lat0 = trajets_valid[:, COL_LAT]
        lon0 = trajets_valid[:, COL_LON]
        lat1 = trajets_valid[:, COL_LAT_AR]
        lon1 = trajets_valid[:, COL_LON_AR]
    
        mask_barriere = detect_barrier_crossings(
            lat1, lon1, lat0, lon0, self.barrieres
        )
        trajets_valid = trajets_valid[~mask_barriere]
        if trajets_valid.shape[0] == 0:
             iso_record = {
                "numiso": numisoprecedent,
                "ecart": ecart_precedent,          # secondes ou minutes depuis t0
                "points": empty_iso,           # tenseur uniquement avec les colonnes utiles par point
                "tmini": 0,
                "distmini": 0,
                "nptmini": nptmini,
                "idxptmini": 0
              }
            
             return iso_record
    
        # ---------------------------------------------------------
        # Réduction à 3x la meilleure distance
        # ---------------------------------------------------------
        dist = trajets_valid[:, COL_DIST_AR]
        distmini = dist.min()
    
        seuil_dist = torch.maximum(3.0 * distmini, seuil_min_dist)
        trajets_valid = trajets_valid[dist <= seuil_dist]
        if trajets_valid.shape[0] == 0:
             iso_record = {
                "numiso": numisoprecedent,
                "ecart": ecart_precedent,          # secondes ou minutes depuis t0
                "points": empty_iso,           # tenseur uniquement avec les colonnes utiles par point
                "tmini": 0,
                "distmini": 0,
                "nptmini": nptmini,
                 "idxptmini": 0

              }
            
             return iso_record
      
    
        # ---------------------------------------------------------
        # Mise à jour colonnes isochrone
        # ---------------------------------------------------------
        n_out = trajets_valid.shape[0]
        trajets_valid[:, COL_NPT] = torch.arange( self.next_npt, self.next_npt + n_out,  device=device, dtype=dtype)
        self.next_npt += n_out
        trajets_valid[:, COL_LAT] = trajets_valid[:, COL_LAT_AR]
        trajets_valid[:, COL_LON] = trajets_valid[:, COL_LON_AR]

        # ---------------------------------------------------------
        # Meilleur point
        # ---------------------------------------------------------
        dist = dist_mn_vec(
        torch.deg2rad(trajets_valid[:, COL_LAT_AR]),
        torch.deg2rad(trajets_valid[:, COL_LON_AR]),
        self.lat_arrivee_rad,
        self.lon_arrivee_rad
        )
    
        idx_best = torch.argmin(dist)
        distmini = dist[idx_best]
      
        v_best   = trajets_valid[idx_best, COL_SPEED]
        tmini    = (distmini / torch.clamp(v_best, min=1e-3)) * 3600.0
        tmini = float(tmini.item())
        # identifiant du point le plus proche
        nptmini = int(trajets_valid[idx_best, COL_NPT].item())
    
        # ---------------------------------------------------------
        # Météo aux points d'arrivée
        # ---------------------------------------------------------
        lat_arr = trajets_valid[:, COL_LAT]
        lon_arr = trajets_valid[:, COL_LON]
    
        if mode == 'gfs':
            dtigGFS = self.dtig0GFS + ecart_val
            trajets_valid[:, COL_TWS], trajets_valid[:, COL_TWD] = gfs_interpolateur_to(
                dtigGFS, lat_arr, lon_arr
            )
    
        elif mode == 'mixte':
            if ecart_val < 9 * 3600:
                dtigGFS = self.dtig0GFS + ecart_val
                trajets_valid[:, COL_TWS], trajets_valid[:, COL_TWD] = gfs_interpolateur_to(
                    dtigGFS, lat_arr, lon_arr
                )
            else:
                dtigECM = self.dtig0ECM + ecart_val                                                                                       
                trajets_valid[:, COL_TWS], trajets_valid[:, COL_TWD] = ecm_interpolateur(
                    dtigECM, lat_arr, lon_arr
                )
        else:
            raise ValueError(f"mode inconnu : {mode}")
     
        # ---------------------------------------------------------
        # Iso suivant
        # ---------------------------------------------------------
        
        iso_suivant = trajets_valid[:, 0:11].clone()
    
        # ---------------------------------------------------------
        # isoglobal on copie cap et speed en colonnes 10 et 
        # col 10 <- CAP
        # col 11 <- SPEED
        # ---------------------------------------------------------
        trajets_valid[:, 10] = trajets_valid[:, COL_CAP]
        trajets_valid[:, 11] = trajets_valid[:, COL_SPEED]

        self.npt_to_global_idx = {}       #dictionnaire pour garder les references de points 
        start=self.kglobal
        end = self.kglobal + n_out

        npts = trajets_valid[:, COL_NPT].long().detach().cpu().tolist()

        for i, npt in enumerate(npts):
            self.npt_to_global_idx[npt] = start + i


        self.isoglobal[start:end] = trajets_valid[:, 0:12]

        idx_global = self.npt_to_global_idx[nptmini]
        distmini = float(distmini.item())
        
        self.isoglobal_meta.append({
            "numiso": numiso,
            "ecart": ecart_val,
            "start": start,
            "end": end,   
            "tmini": tmini,
            "distmini": distmini,
            "nptmini": nptmini,
            "idxptmini": 0
           
        })
       
        self.kglobal = end
        iso_record = {
                "numiso"  : numiso,
                "ecart"   : ecart,          # secondes ou minutes depuis t0
                "points"  : iso_suivant,           # tenseur uniquement avec les colonnes utiles par point
                "tmini"   : tmini,
                "distmini": distmini,
                "nptmini" : nptmini,
                "idxptmini": idx_global
              }
          
        return iso_record


        
        
    def construit_dico_isochrones_1sur6(self, seuil=0.01):
        dico_isochrones = {}
        for rec in self.isoglobal_meta:
            numiso = rec["numiso"]
    
            if numiso % 6 != 0:
                continue
    
            start = rec["start"]
            end = rec["end"]
    
            bloc = self.isoglobal[start:end]
    
            # nouvelles colonnes : LAT=2, LON=3
            lat_lon = bloc[:, [2, 3]]
    
            segs = decoupe_latlon_folium(lat_lon, seuil=seuil)
    
            if segs:
                dico_isochrones[int(numiso)] = segs
    
        return dico_isochrones

    #@torch.no_grad()
    def reconstruire_chemin_rapide(self,isoglobal, nptmini) :
        """
        Reconstruit le chemin depuis le point d’ID nptmini jusqu’au point d’ID 0 (inclus).
        """
        # Création des dictionnaires pour remonter efficacement
        point_ids = isoglobal[:, 0].to(torch.int32).tolist()
        parent_ids = isoglobal[:, 1].to(torch.int32).tolist()
        dico_parents = dict(zip(point_ids, parent_ids))
        dico_index = {pid: idx for idx, pid in enumerate(point_ids)}
        chemin_indices = []
        pt = nptmini
        while True:
            idx = dico_index.get(pt)
            if idx is None:
                raise ValueError(f"Point {pt} introuvable dans isoglobal")
            chemin_indices.append(idx)
            if pt == 0:
                break
            pt = dico_parents.get(pt, 0)   
        chemin_indices.reverse()
        return chemin_indices
    




    def construit_routage_depuis_chemin(self, chemin_indices):
        chemin = self.isoglobal[chemin_indices]        
        routage = torch.zeros((chemin.shape[0], 16), device=self.device, dtype=self.dtype)
    
        # Colonnes récupérées depuis isoglobal
        routage[:, [2, 3, 5, 6, 9, 10, 12, 13, 14, 15]] = chemin[:, [2, 3, 5, 10, 11, 4, 8, 9, 6, 7]]
    
        # numiso / ecart depuis isoglobal_meta
        chemin_indices_t = torch.as_tensor(chemin_indices, device=self.device)
    
        for rec in self.isoglobal_meta:
            mask = (chemin_indices_t >= rec["start"]) & (chemin_indices_t < rec["end"])
            if mask.any():
                routage[mask, 0] = rec["numiso"]
                routage[mask, 1] = rec["ecart"]
    
        # temps unix
        routage[:, 4] = self.t0vr + routage[:, 1]
    
        # vmgmin / vmgmax
        tws10 = torch.round(routage[:, 12] * 10).long().clamp(0, 699)
        routage[:, 7] = self.tabvmg10to[tws10, 2]
        routage[:, 8] = self.tabvmg10to[tws10, 4]
    
        # boost
        voile = routage[:, 10].long()
        twa10 = torch.round(routage[:, 5].abs() * 10).long().clamp(0, 1799)
    
        vitesse_voile = self.polairesglobales10to[voile, tws10, twa10]
        vitesse_maxi = self.polairesglobales10to[7, tws10, twa10]
    
        routage[:, 11] = vitesse_maxi / (vitesse_voile + 1e-4)    
        return routage

        
    def lissage_routage_np(self, routage_np, window=3, poly=1, marge=1):
        ''' La fonction travaille sur du numpy '''
        tabres = np.copy(routage_np)
    
        n = len(routage_np)
        if n < 3:
            return tabres
    
        numiso = routage_np[:, 0]
        ecart  = routage_np[:, 1]
        twa    = routage_np[:, 5]
    
        pos_twa = np.where(twa[:-1] * twa[1:] < 0)[0] + 1
    
        dt = np.diff(ecart)
        pos_dt = np.where(np.diff(dt) != 0)[0] + 1 if len(dt) >= 2 else np.array([], dtype=int)
    
        numiso_wp = np.array(
            [rec["numiso"] for rec in self.wp_passages.values()],
            dtype=numiso.dtype
        ) if getattr(self, "wp_passages", None) else np.array([], dtype=numiso.dtype)
    
        pos_wp = np.where(np.isin(numiso, numiso_wp))[0]

        pos_fix_base = np.unique(np.concatenate((
            np.array([0, n - 1], dtype=int),
            pos_twa.astype(int),
            pos_dt.astype(int),
            pos_wp.astype(int),
        )))
        
        pos_fix = ajoute_marge_points_fixes(pos_fix_base, n, marge=2)
        pos_fix = pos_fix[(pos_fix >= 0) & (pos_fix < n)]
    
        window = int(window)
        if window % 2 == 0:
            window += 1
        poly = min(int(poly), window - 1)
    
        for k in range(len(pos_fix) - 1):
            i0 = pos_fix[k]
            i1 = pos_fix[k + 1]
    
            j0 = i0 + marge
            j1 = i1 - marge
    
            if j1 <= j0:
                continue
    
            segment = routage_np[i0:i1 + 1]
            if len(segment) < window:
                continue
    
            lat = segment[:, 3]
            lon = segment[:, 4]
    
            lat_s = savgol_filter(lat, window, poly, mode="nearest")
            lon_s = savgol_filter(lon, window, poly, mode="nearest")
    
            # On ne modifie ni les points fixes ni leurs voisins si marge=1
            tabres[j0:j1 + 1, 3] = lat_s[(j0 - i0):(j1 - i0 + 1)]
            tabres[j0:j1 + 1, 4] = lon_s[(j0 - i0):(j1 - i0 + 1)]
    
        return tabres

    def reconstruit_colonnes_routage_lisse(self, tabpointslisses, mode="gfs"):
        """
        tabpointslisses :
          col 0 = numiso
          col 1 = ecart
          col 2 = lat
          col 3 = lon
          col 4 = timestamp unix
    
        retourne un tableau numpy/torch routage avec 16 colonnes
        """
    
        if not torch.is_tensor(tabpointslisses):
            routage = torch.as_tensor(tabpointslisses, dtype=self.dtype, device=self.device)
        else:
            routage = tabpointslisses.to(device=self.device, dtype=self.dtype)
    
        n = routage.shape[0]
    
        if routage.shape[1] < 16:
            tmp = torch.zeros((n, 16), device=self.device, dtype=self.dtype)
            tmp[:, :routage.shape[1]] = routage
            routage = tmp
    
        if n < 2:
            return routage
    
        lat = routage[:, 2]
        lon = routage[:, 3]
        ecart = routage[:, 1]
    
        # --------------------------------------------------
        # Météo aux points lissés
        # --------------------------------------------------
        if mode == "gfs":
            dtig = self.dtig0GFS + ecart
            routage[:, 12], routage[:, 13] = gfs_interpolateur_to(dtig, lat, lon)
    
        elif mode == "mixte":
            mask_gfs = ecart < 9 * 3600
    
            if mask_gfs.any():
                dtig_gfs = self.dtig0GFS + ecart[mask_gfs]
                routage[mask_gfs, 12], routage[mask_gfs, 13] = gfs_interpolateur_to(
                    dtig_gfs,
                    lat[mask_gfs],
                    lon[mask_gfs]
                )
    
            if (~mask_gfs).any():
                dtig_ecm = self.dtig0ECM + ecart[~mask_gfs]
                routage[~mask_gfs, 12], routage[~mask_gfs, 13] = ecm_interpolateur(
                    dtig_ecm,
                    lat[~mask_gfs],
                    lon[~mask_gfs]
                )
        else:
            raise ValueError(f"mode inconnu : {mode}")
    
        # --------------------------------------------------
        # Cap et vitesse entre i-1 et i
        # --------------------------------------------------
        lat0 = lat[:-1]
        lon0 = lon[:-1]
        lat1 = lat[1:]
        lon1 = lon[1:]
    
        dlat = lat1 - lat0
        dlon = dlon_short_torch(lon1, lon0)
    
        latm_rad = torch.deg2rad((lat0 + lat1) * 0.5)
        dx = dlon * torch.cos(latm_rad)
        dy = dlat
    
        cap = torch.rad2deg(torch.atan2(dx, dy)) % 360.0
    
        dist = dist_mn_vec(
            torch.deg2rad(lat0),
            torch.deg2rad(lon0),
            torch.deg2rad(lat1),
            torch.deg2rad(lon1)
        )
    
        dt_h = (ecart[1:] - ecart[:-1]) / 3600.0
        speed = dist / torch.clamp(dt_h, min=1e-6)
    
        routage[1:, 6] = cap
        routage[1:, 9] = speed
    
        # Pour le premier point : pas de déplacement précédent
        routage[0, 6] = routage[1, 6]
        routage[0, 9] = routage[1, 9]
    
        # --------------------------------------------------
        # TWA au point d'arrivée
        # --------------------------------------------------
        routage[1:, 5] = ftwao(routage[1:, 6], routage[1:, 13])
        routage[0, 5] = routage[1, 5]
    
        # --------------------------------------------------
        # VMG min / max
        # --------------------------------------------------
        tws10 = torch.round(routage[:, 12] * 10).long().clamp(0, 699)
    
        routage[:, 7] = self.tabvmg10to[tws10, 2]
        routage[:, 8] = self.tabvmg10to[tws10, 4]
    
        
        # --------------------------------------------------
        # Voile et boost avec logique séquentielle
        # --------------------------------------------------
        twa10 = torch.round(routage[:, 5].abs() * 10).long().clamp(0, 1799)
        
        voile_best = self.polairesglobales10to[8, tws10, twa10].long()
        vitesse_best = self.polairesglobales10to[7, tws10, twa10]
        
        voile_calc = torch.empty_like(voile_best)
        boost_calc = torch.empty_like(vitesse_best)
        
        # voile initiale : on garde celle du routage original si disponible
        voile_prec = int(routage[0, 10].item())
        
        for i in range(routage.shape[0]):
            tws_i = tws10[i]
            twa_i = twa10[i]
        
            vitesse_voile_prec = self.polairesglobales10to[voile_prec, tws_i, twa_i]
            boost_i = vitesse_best[i] / (vitesse_voile_prec + 1e-4)
        
            if boost_i > 1.014:
                voile_i = int(voile_best[i].item())
            else:
                voile_i = voile_prec
        
            voile_calc[i] = voile_i
            boost_calc[i] = boost_i
        
            voile_prec = voile_i
        
        routage[:, 10] = voile_calc.to(self.dtype)
        routage[:, 11] = boost_calc
        routage[:, 9] = vitesse_best
        
    
        # --------------------------------------------------
        # Stamina / peno
        # --------------------------------------------------
        # Option simple : on garde les anciennes valeurs si elles existent,
        # sinon on laisse à 0.
        #
        # Pour recalculer précisément stamina et peno, il faut rejouer toute
        # la logique manoeuvres/changements de voile depuis le début du routage.
    
        return routage

    def routage_actions(self, routage):
        ''' On decale les colonnes pour avoir les actions à faire '''
        cols_actions = [5, 6, 7, 8, 9, 10, 11, 14, 15]
        # 5 twa
        # 6 cap
        # 9 speed
        # 7 vmgmin
        # 8 vmgmax
        # 10 voile
        # 11 boost
        # 14 stamina
        # 15 peno
        
        routage_actions = routage.clone()   # si torch
        routage_actions[:-1, cols_actions] = routage[1:, cols_actions]
        
        # dernière ligne : pas d'action suivante
        routage_actions[-1, cols_actions] = routage[-1, cols_actions]
        return routage_actions
       
              
# fin de la classe RoutageSession               

# classe Simulateur
###################

class Simulateur:                                                  # version avec nouvel algorithme

    
    def __init__(self, course, user_id):
        
        tic=time.time()
        self.device='cuda'
        self.dtype=torch.float32

        self.carte_cache = {}                           #Dictionnaire pour le cache des cartes 
        # Initialisation globale
        ########################
               
        self.course    = course
        self.user_id   = user_id
        self.tigGFS    = tigGFS
                      
        # Chargement des données
        DonneesCourse1                                                      = rechercheDonneesCourseCache(course)   # c 'est un objet de la classe DonneesCourse
        self.boatinfos,self.posStartVR,self.exclusionsVR2  ,self.positionvr = rechercheDonneesCourseUser( user_id,course)                  # self.exclusionsVR2 est un dictionnaire avec les exclusions VR qui ne sert pas 
        self.personalinfos,self.ari,self.waypoints,self.exclusionsperso,self.barrieres,self.trajets,self.tolerancehvmg,retardpeno = rechercheDonneesPersoCourseUser( user_id,course)
        baDate,boatActions         = rechercheTableBoatActions(user_id,course)
        # self.boatactions           = json.loads(boatActions)
        self.baDate                = baDate
        self.leginfos              = DonneesCourse1.leginfos
        self.exclusionsVR          = DonneesCourse1.tabexclusions                     # ce sont les exclusions VR
        self.tabicelimits          = DonneesCourse1.tabicelimits
        self.carabateau            = DonneesCourse1.carabateau
        self.polairesglobales10to  = DonneesCourse1.polaires_gpu
        self.tabvmg10to            = DonneesCourse1.vmg_gpu
        self.polaires_np           = DonneesCourse1.polaires_np                               #ce sont les polaires10
        self.tabvmg                = DonneesCourse1.vmg_cpu
        self.polairesjson          = DonneesCourse1.polairesjson
        self.typevoile             = self.carabateau['typevoile']  
        self.exclusionsVR.update(self.exclusionsperso)          # on rajoute exclusionsperso à exclusions VR 

        # print ('boatinfos',self.boatinfos)
        # print ('boatactions',boatActions)
        
        
        try:
            self.exclusions = {  nom: torch.tensor([[lon, lat] for lat, lon in coords], device='cuda')   for nom, coords in self.exclusionsVR.items()  }    
            self.exclusions_json = {  nom: tensor.cpu().tolist()  for nom, tensor in self.exclusions.items()}

        except : 
            self.exclusions = {}  
            self.exclusions_json = {}    
              

        self.segments             = prepare_segments(self.exclusions)  # [S, 4] segments [x1, y1, x2, y2]        
        self.t0vr                 = self.posStartVR['t0']
        self.MF                   = 0.8
        self.furler               = 0.8

        polairesjson    =   self.polairesjson 
        self.polar_id                    = polairesjson['_id']
        self.label                       = polairesjson['label']                              # type du bateau
        self.globalSpeedRatio            = polairesjson['globalSpeedRatio']       
        self.foilSpeedRatio              = polairesjson['foil']['speedRatio']
        self.autoSailChangeTolerance     = polairesjson["autoSailChangeTolerance"]
        self.badSailTolerance            = polairesjson["badSailTolerance"]
        self.iceSpeedRatio               = polairesjson['iceSpeedRatio']
        self.maxSpeed                    = polairesjson['maxSpeed']
        self.twaMin                      = polairesjson['foil']['twaMin']
        self.twaMax                      = polairesjson['foil']['twaMax']
        self.twaMerge                    = polairesjson['foil']['twaMerge']
        self.twsMin                      = polairesjson['foil']['twsMin']
        self.twsMax                      = polairesjson['foil']['twsMax']
        self.twsMerge                    = polairesjson['foil']['twsMerge']
        self.speedRatio                  = polairesjson['foil']['speedRatio']
        self.hull                        = polairesjson['hull']['speedRatio']
        self.lws                         = polairesjson['winch']['lws']
        self.hws                         = polairesjson['winch']['hws']
        self.lwtimer                     = polairesjson['winch']['sailChange']['pro']['lw']['timer']
        self.hwtimer                     = polairesjson['winch']['sailChange']['pro']['hw']['timer']
        self.lwratio                     = polairesjson['winch']['sailChange']['pro']['lw']['ratio']
        self.hwratio                     = polairesjson['winch']['sailChange']['pro']['hw']['ratio']
        self.tackprolwtimer              = polairesjson['winch']['tack']['pro']['lw']['timer']
        self.tackprolwratio              = polairesjson['winch']['tack']['pro']['lw']['ratio']
        self.tackprohwtimer              = polairesjson['winch']['tack']['pro']['hw']['timer']
        self.tackprohwratio              = polairesjson['winch']['tack']['pro']['hw']['ratio']
        self.gybeprolwtimer              = polairesjson['winch']['gybe']['pro']['lw']['timer']
        self.gybeprolwratio              = polairesjson['winch']['gybe']['pro']['lw']['ratio']
        self.gybeprohwtimer              = polairesjson['winch']['gybe']['pro']['hw']['timer']
        self.gybeprohwratio              = polairesjson['winch']['gybe']['pro']['hw']['ratio']

        self.tabtws                      = polairesjson['tws']                                            
        self.tabtwa                      = polairesjson['twa']
        self.update                      = polairesjson['_updatedAt']
        self.nbvoiles                    = len(polairesjson['sail'])
        self.typevoile= []
        for i in range(self.nbvoiles) :
            self.typevoile.append( self.polairesjson['sail'][i]['name'])
        # print (self.typevoile) 
        self.coeffboat                   = tabCoeffboat[self.polar_id]['stamina']
        voile_auto=1
        
         ### Pour Calcul meteo
        self.dtig0GFS               = self.t0vr-tigGFS          
        self.dtig0ECM               = self.t0vr-tigECM 

        posStartVR=self.posStartVR     
        self.state       = posStartVR['state']
        self.y0          = posStartVR['y0']
        self.x0          = posStartVR['x0']
        self.t0          = posStartVR['t0']
        self.sail        = posStartVR['voile']
        self.voileAuto   = posStartVR['voileAuto']
        self.tws         = posStartVR['tws']
        self.twd         = posStartVR['twd']
        self.twa         = posStartVR['twa']
        self.twaAuto     = posStartVR['twaAuto']
        self.heading     = posStartVR['heading']
        self.speed       = posStartVR['speed']
        self.stamina     = posStartVR['stamina']
        self.soldepeno   = posStartVR['penovr']        
        self.boost       = posStartVR['boost']
        self.lastCalcDate= posStartVR['lastCalcDate'] 
       
        tws, twd = gfs_interpolateur_np(self.dtig0GFS, self.y0, self.x0)   
        print ('tws et twd calcules au depart  tws {:6.3f} twd {:6.3f} '.format(tws,twd))
        
        if self.sail>10:
                self.voileAuto = 1
                self.voile     = self.sail%10
        else :
                self.voileAuto = 0
                self.voile     = self.sail

        # print ('state ',state)
        print ('twas auto',self.twa,self.twaAuto)
        if self.twaAuto==True:
            Option=1
            valeur=self.twa
         
        else :
            Option=0
            valeur=self.heading
            

        tws10=round(self.tws*10)    

        # if self.twaAuto==True:
        #     Option=1
        #     valeur=self.twa
        # else :
        #     Option=0
        #     valeur=self.heading

        if self.state == 'waiting'  :
            # Les valeurs de vent ont deja ete calculees dans posStartVR , donc il n y a rien a faire    par contre heading peut etre faux , il faut aller chercher boatactions
            timestamp,progstable           = rechercheTableProgsvr(user_id,course) 
            
            progstable                             = json.loads(progstable) 
            print ('\n***************************************dans init de Simulateur',progstable)
            if "progs" in progstable:
                Option=progstable['progs'][0][1]
                valeur=progstable['progs'][0][2]
                if Option==0:
                    self.heading=valeur                
                    self.twa=ftwaos(self.heading, self.twd)  
                if Option==1:
                    self.twa=valeur 
                    self.heading= fcap(self.twa,self.twd)
                    
               # il faut aussi mettre à jour la voile et la vitesse qui pour l instant ne sont pas donnés par vr 

            if "wps" in progstable:
                Option = 1                # j ai necessairement 2 waypoints depart et waypoint 1 
                 
                print (' La valeur de progsstable est la suivante ****************************************************',progstable )   
                progswaypoints = progstable["wps"]
                            
                WP0 = [self.y0,self.x0]

                WP1 =[progswaypoints[0]['lat'],progswaypoints[0]['lon']]

               
                print ('WP1 ',WP1)
                self.heading = calcul_cap_grand_cercle( WP0 [0],WP0[1], WP1[0], WP1[1] )
                print ('cap_initial',self.heading)
               
                
                self.twa=ftwaos(self.heading, self.twd)     
            twa10=round(abs(self.twa)*10)
              
            vitessevoileini        = self.polaires_np[int(self.voile), tws10, twa10] 
            self.meilleurevitesse  = self.polaires_np[  7         , tws10, twa10] 
            self.meilleurevoile    = self.polaires_np[  8         , tws10, twa10] 
            self.voile             = self.meilleurevoile 
            
            self.boost             = float(self.meilleurevitesse/(vitessevoileini+0.0001))    
            self.speed             = self.meilleurevitesse  
            self.bestvmgmin        = self.tabvmg[tws10, 2].item()
            self.bestvmgmax        = self.tabvmg[tws10, 4].item() 
             
            
            
            dt = 0
            self.lastCalcDate     = self.boatinfos['bs']['legStartDate'] 
            self.bestvmgmin       = self.tabvmg[tws10, 2].item()
            self.bestvmgmax       = self.tabvmg[tws10, 4].item()
            self.penoSail         = 0
            self.penoGybe         = 0
            self.penoTack         = 0      

            
            self.posDepart=np.array([0,Option,valeur,60,self.y0,self.x0,self.t0,self.twa,   self.heading,self.voile,self.voile,\
                                     self.boost,self.speed,self.bestvmgmin,self.bestvmgmax,self.twd,self.tws,self.stamina,self.penoSail,self.penoGybe,self.penoTack,0])        


           
        
       
        if self.state != 'waiting'  :  
            self.lastCalcDate= posStartVR['lastCalcDate']           
           
            if self.twaAuto==self.twa:                  # navigation au cap 
                self.cap= fcap(self.twa,self.twd)
            else :                            # navigation a la twa 
                self.twa =ftwaos(self.heading, self.twd) 
      
                
        # Penalites initiales 
       
        try :
            tsEndOfSailChange = self.boatinfos['bs']['tsEndOfSailChange']
            self.penoSail     = (tsEndOfSailChange - self.lastCalcDate )/1000
        except: 
            self.penoSail     = 0

        try:   
            tsEndOfGybe       = self.boatinfos['bs']['tsEndOfGybe']
            self.penoGybe     = (tsEndOfGybe - self.lastCalcDate )/1000
        except:
            self.penoGybe     = 0

        try:   
            tsEndOfTack       = self.boatinfos['bs']['tsEndOfTack']
            self.penoTack     = (tsEndOfTack - self.lastCalcDate)/1000 
        except:
            self.penoTack     = 0

        # Calcul du boost initial 

        
        twa10=round(abs(self.twa)*10)
              
        vitessevoileini   = self.polaires_np[int(self.voile)%10, tws10, twa10] 
        meilleurevitesse  = self.polaires_np[  7         , tws10, twa10] 
        self.meilleurevoile    = self.polaires_np[  8         , tws10, twa10] 
        self.boost        = float(meilleurevitesse/(vitessevoileini+0.0001))    

        self.bestvmgmin       = self.tabvmg[tws10, 2].item()
        self.bestvmgmax       = self.tabvmg[tws10, 4].item()

        
        # numero ite 
        # 1 Option cap 0 ou twa 1
        # 2 valeur : valeur du cap ou de la twa imposee
        # 3 dt : delta de temps pour la position suivante
        # 4 lat : latitude du point
        # 5 lon : longitude du point
        # 6 voile : voile à l'arrivee sur ce point
        # 7 twa_ant : twa à l'arrivee sur ce point
        # 8 stamina_ant : energie à l'arrivee sur ce point
        # 9 peno        _ant : penalite à l'arrivee sur ce point
        # 10 twd : direction du vent sur le point 10)10 tws : vitesse du vent sur le point
        
     

        self.posDepart=np.array([0,Option,valeur,60,self.y0,self.x0,self.t0,self.twa,   self.heading,self.voile,self.meilleurevoile,self.boost,\
                                 self.speed,self.bestvmgmin,self.bestvmgmax,self.twd,self.tws,self.stamina,self.penoSail,self.penoGybe,self.penoTack,0])        


    
  
    
    def positionDash(self,lat,lon,twd,tws,ts,voile,speed,stamina,penoSail,penoGybe,penoTack,programmations):
        print ('on est dans positionDash')
        print ('programmations',programmations)
        self.dtig0GFS               = ts-tigGFS
        tws10 = int(round(tws * 10))
        bestvmgmin = self.tabvmg10to[tws10, 2]
        bestvmgmax = self.tabvmg10to[tws10, 4]
        boost=1  
        bestvmgmin        = self.tabvmg[tws10, 2]
        bestvmgmax        = self.tabvmg[tws10, 4] 
        voile=voile%10 -1        
        if "progs" in programmations:
                Option=programmations['progs'][0][1]
                valeur=programmations['progs'][0][2]
                if Option==0:
                    heading=valeur                
                    twa=ftwaos(heading,twd)  
                if Option==1:
                    twa=valeur 
                    heading= fcap(twa,twd)               
        
        posDash = np.array([0,Option,valeur,60,lat,lon,ts,twa,heading,voile,voile,boost,speed,bestvmgmin,bestvmgmax,twd,tws,stamina,penoSail,penoGybe,penoTack,0])        
        return posDash 
  
  
    
            
       
    def condenseprogs(self,progs):
        progsx=[]
        try:
            for i in range (len(progs)):
                deg     = progs [i]['deg']
                autoTwa = progs [i]['autoTwa']
                isProg  = progs [i]['isProg']
                user_id = progs [i]['_id']['user_id']
                race    = progs [i]['_id']['race_id']
                leg     = progs [i]['_id']['leg_num']
                ts      = progs [i]['_id']['ts']
                action  = progs [i]['_id']['action']
    
                # print (progs[i])
                if (progs[i]['autoTwa'] ):                # twa= true
                    twa=progs[i]['deg']
                    value=twa
                    option=1
                            
                else :
                    option=0
                    value=progs[i]['deg']
    
                progsx.append([int(ts/1000),option,value,isProg])
        except:
            print('\npas de programmation dans boataction\n' )
    
        # print()
        # print ('progsx\n',progsx)
        return progsx


    def extraitprogrammations (self):
        # boatInfos,poSStart,positionvr=rechercheDonneesCourseUser( user_id,course)
        # print (boatinfos)
    
         #3) # on recupere les dernières valeurs de boatactions qui on tete stockees dans la base de donnees 
        timestamp,progstable           = rechercheTableProgsvr(self.user_id,self.course)                                  # progstablestr
        # print()
        # print ('dernier enregistrement boatAction',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBA)))
        # print ('progstablestr ', progstablestr)
        # print()
    
        
        #######################################################################################################
        boatInfos=self.boatinfos
        #1)  Etat et Position dans VR   pour calcul de progDepart 
        state       = boatInfos['bs']['state']
        heading     = boatInfos['bs']['heading']
        try :
            ts          = int(boatInfos['bs']['lastCalcDate']/1000)
        except :   
            ts          = int(boatInfos['bs']['legStartDate']/1000)  
        tsmn           = math.ceil(ts / 60) * 60  # Arrondi à la minute supérieure 
        lastCalcDateBI = ts                                              # si on n est pas parti la premeire prog est au moment du depart
        
        try:    
            twa         = boatInfos['bs']['twa']                       # c est la twa initiale, si on n'est pas parti pas de twa 
        except :
            twa = 45     #par defaut 
        try : 
            twaAuto     = boatInfos['bs']['twaAuto']
        except:     
            twaAuto     = twa              # il n'y a pas de twaauto au depart 
        if twa!= twaAuto:
            option=0
            valeur=heading
        else:       
            option=1    
            valeur=twa     
        progDepart=[[ts,option,valeur,False]]   # c est la prog depart pour progs ( c'est la situation vr et le False indique que ce n'est pas une programmation mais l etat actuel ) 
        
        
        try:
            progstable                             = json.loads(progstable) 
            lastCalcDateBA=timestamp.timestamp()
        except:
            progstable=progDepart                   # s il n y a jamais eu de programmations sur cette course 
            lastCalcDateBA=time.time()-3600         # pour que la programmation prise en charge soit boatinfos  
            
       ###################################################################################################
    
        # on va reaorganiser car il peut y avoir une contradiction entre les 2 
        # print  ('lastCalcDateBA',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBA)),lastCalcDateBA)
        # print  ('lastCalcDateBI',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBI)),lastCalcDateBI)
    
        
        if (lastCalcDateBA>lastCalcDateBI-60 or state=='waiting') :           # c estle boat action  et donc progstable qui s applique 
            date    = lastCalcDateBA
            origine='Boatactions'
            
            if  "wps" in  progstable  :
                # print ('c\'est progtable issu de BoatActions qui s applique pour les wp')            
                origine ='Boatactions'
                progsBA=progstable['wps'] 
                # print ('progsBA',progsBA)
                programmations=json.dumps({'wps': progsBA})  
    
            else:
                # print ('c\'est progtable issu de BoatActions qui s applique pour les progs et qui existe ') 
                # print ('progstable ',progstable)
                try:
                    progsBA=progstable['progs']
                    if progsBA[0][3]!=False:
                        progsBA.insert(0, progDepart[0])  # Insère 1 à l'index 0
                        # print ('progsBA',progsBA)    
                    programmationsBA=json.dumps({'progs': progsBA})  
                    programmations=programmationsBA
                except:
                    progsBA=  progDepart 
                    programmationsBA=json.dumps({'progs': progsBA})  
                    programmations=programmationsBA
    
    
        else:
            date=lastCalcDateBI    
            origine='BoatInfos' 
    
            if "waypoints" in boatInfos['bs']:
                programmations=json.dumps({'wps':boatInfos['bs']['waypoints']  })
    
            else:
                try:                                            # boatInfos['ba'] peut ne pas exister 
                    boatInfosba=boatInfos['ba']   
                    # print()
                    # print ('boatinfosba',boatInfosba)
                    # print()
                    progsboatInfosba = [item for item in boatInfosba if item.get('_id', {}).get('action') == 'heading']
                    progsInfosGlobales=self.condenseprogs(progsboatInfosba)
                    if progsInfosGlobales[0][3]!=False:
                        progsInfosGlobales.insert(0, progDepart[0])  # Insère 1 à l'index 0  
                    programmations=json.dumps({'progs': progsInfosGlobales})  
                
                except:
                    print ("Il n'y a pas de boatinfos['ba'] on prend la programmation du depart progDepart")  
                    origine='Pas de progs'  
                    progsInfosGlobales=progDepart
                    programmations=json.dumps({'progs': progsInfosGlobales})    
    
     
        programmations=json.loads(programmations)      # on transforme en objetjson
        return date,origine,programmations


    def deplacement(self,position,calcMer=False):
        global tigGFS
        '''A partir des valeurs de depart et de l option cap ou twa calcule le point apres le deplacement'''
        # Colonnes du tableau point
        COL_INDICE     = 0
        COL_OPTION     = 1
        COL_VALEUR     = 2
        COL_DTI        = 3
        COL_LAT        = 4
        COL_LON        = 5
        COL_T0         = 6
        COL_TWA        = 7
        COL_HEADING    = 8
        COL_VOILE      = 9
        COL_MEVOILE    = 10
        COL_BOOST      = 11
        COL_SPEED      = 12
        COL_BESTVMGMIN = 13
        COL_BESTVMGMAX = 14
        COL_TWD        = 15
        COL_TWS        = 16
        COL_STAMINA    = 17
        COL_PENOSAIL   = 18
        COL_PENOGYBE   = 19
        COL_PENOTACK   = 20
        COL_ISMER        = 21
        p=position.copy()

        # -----------------------------
        # Lecture du point d'entrée
        # -----------------------------
        indice      = int(p[COL_INDICE])
        option      = int(p[COL_OPTION])
        valeur      = p[COL_VALEUR]
        dt          = p[COL_DTI]
    
        y0          = p[COL_LAT]
        x0          = p[COL_LON]
        t0          = p[COL_T0]
    
        twa_ant     = p[COL_TWA]                   # sert pour calculer tack Gybe
        heading_ant = p[COL_HEADING]
        twdini      = p[COL_TWD]

        voile_ant   = int(p[COL_VOILE])            # sert pour calculer le boost  
        stamina_ini = p[COL_STAMINA]



        peno_sail_ant = max(0.0, p[COL_PENOSAIL])
        peno_gybe_ant = max(0.0, p[COL_PENOGYBE])
        peno_tack_ant = max(0.0, p[COL_PENOTACK])
        
        # Temps pénalisé pendant cette itération
       
        # temps_penalise = min(dt, peno_totale_avant)
        
        # Décompte au début du pas
        peno_sail_avant = max(0.0, peno_sail_ant - dt)
        peno_gybe_avant = max(0.0, peno_gybe_ant - dt)
        peno_tack_avant = max(0.0, peno_tack_ant - dt)

        # peno_totale_avant =max( peno_sail, peno_gybe, peno_tack)

        # print ('peno ant ',peno_gybe_ant)
        
    
        # -----------------------------
        # Vent au point de départ
        # -----------------------------
   
        
        dtig = t0 - self.tigGFS
        tws, twd = gfs_interpolateur_np(dtig, y0, x0)



        if indice==0:
            print ('*******************************************************************************')
            print ('*******************************************************************************')
            print (' 5576   Verification du calcul vent au point initial ')         
            print ("tigGFS {}  t0vr {}  dtig en h {}  ".format( time.strftime(" %d %b %H:%M ",time.localtime(self.tigGFS)),  time.strftime(" %d %b %H:%M ",time.localtime(t0)),dtig/3600))
           # print (" tigGFS =", tigGFS ,time.strftime(" %d %b %H:%M ",time.gmtime(self.tigGFS)))
           # print (' dtig {} soit  {:6.2f}h '.format(dtig, dtig/3600))
            print ()
            print (' Valeurs issues de position    tws   {:6.5f}  twd   {:6.5f}  t0 {}'.format(position [COL_TWS] ,position [COL_TWD],position [COL_T0]) )
            print (' Valeurs calculees             tws   {:6.5f}  twd   {:6.5f}  t0 {}'.format(tws,twd,t0))
            print ('*******************************************************************************')
            print ('*******************************************************************************')
            print ()

         # -----------------------------
        # Calcul TWA / heading
        # -----------------------------
        if option == 1:
            # TWA fixe
            twa = valeur
            heading = fcap(twa, twdini)
        else:
            # Cap fixe
            heading = valeur
            twa = ftwao(heading, twdini)
    
        twa_abs = abs(twa)

        # -----------------------------
        # Lecture polaires
        # -----------------------------
        # print('voile_ant',voile_ant)
        voile_ant=int(voile_ant)
        voile,speed,meilleure_voile,boost=self.calc_voile(voile_ant ,tws,twa )
         
        if voile != voile_ant:
            chgt = 1
           
        else:
            chgt = 0
         
        
        # -----------------------------
        # Détection gybe / tack
        # -----------------------------
        changement_bord = twa_ant * twa < 0
        TGybe= ((twa_ant*twa)<0)*1
        gybe = int(changement_bord and abs(twa_ant) > 90)
        tack = int(changement_bord and abs(twa_ant) <= 90)


         # -----------------------------
        # Nouvelles pénalités
        # -----------------------------
        peno_chgt,peno_gybe,peno_tack,perte_stamina=0,0,0,0
        if chgt>0 :
            peno_chgt =  chgt * peno_np(self.lwtimer, self.hwtimer, tws, stamina_ini) * self.furler
        if gybe>0 :
            peno_gybe =  gybe* peno_np(self.gybeprolwtimer, self.gybeprohwtimer, tws, stamina_ini)
        if tack>0 : 
            peno_tack =    tack* peno_np(self.tackprolwtimer, self.tackprohwtimer, tws, stamina_ini)  


        # On ajoute les nouvelles pénalités aux pénalités restantes
      
       
        peno_sail_total = peno_sail_avant + peno_chgt
        peno_gybe_total = peno_gybe_avant + peno_gybe
        peno_tack_total = peno_tack_avant + peno_tack
        
        peno_totale =max( peno_sail_total,peno_gybe_total, peno_tack_total)
        temps_penalise = min(dt, peno_totale)

        dt_effectif = dt - 0.30 * temps_penalise
        distance_nm = speed * dt_effectif / 3600.0

        cap_rad = np.round(np.deg2rad(heading),5)
        lat_rad = np.deg2rad(y0)

        dy = distance_nm * np.cos(cap_rad) / 60.0
        dx = distance_nm * np.sin(cap_rad) / (60.0 * np.cos(lat_rad))

        y1 = y0 + dy
        x1 = x0 + dx

        # Normalisation longitude éventuelle
        x1 = ((x1 + 180.0) % 360.0) - 180.0

       

        # -----------------------------
        # Mise à jour de la stamina
        # -----------------------------
        if chgt+gybe+tack>0: 
            perte_stamina   = calc_perte_stamina_np(tws, TGybe, chgt, self.coeffboat)
        
        recup           = pts_recuperes(dt, tws)
        stamina         = stamina_ini - perte_stamina + recup
        stamina         = np.clip(stamina, 0.0, 100.0)
       

        # -----------------------------
        #  Calcul du vent au point de sortie avec le nouveau temps et les nouvelles coordonnées 
        # -----------------------------

        tws, twd = gfs_interpolateur_np(dtig+dt , y1, x1)
        
        if option == 1:
            # TWA fixe
            twa = valeur
            heading = fcap(twa, twdini)

        else:
            # Cap fixe
            heading = valeur
            twa = ftwao(heading, twdini)    

        

        tws10 = int(round(tws * 10))
        tws10 = max(0, min(tws10, self.tabvmg10to.shape[0] - 1))
       
        # -----------------------------
        # VMG min / max pour ce TWS
        # -----------------------------      
        
        bestvmgmin = self.tabvmg10to[tws10, 2]
        bestvmgmax = self.tabvmg10to[tws10, 4]

        # -----------------------------
        # Collision avec les terres
        # -----------------------------
        if calcMer:
            cartepy = self.get_carte_segment(y0, x0, y1, x1)            
            segment = [[y0, x0], [y1, x1]]
            
            coupe_terre = intersection(segment, cartepy)
            isMER = 1 if coupe_terre else 0
        else :
            isMER=0  


        
        
        # -----------------------------
        # Point de sortie ce sont les valeurs affichees 
        # -----------------------------
        sortie = p.copy()   
       
        
        sortie[COL_INDICE]     = indice + 1
        sortie[COL_OPTION]     = option
        sortie[COL_VALEUR]     = valeur
        sortie[COL_DTI]        = dt
        
        sortie[COL_LAT]        = y1
        sortie[COL_LON]        = x1
        sortie[COL_T0]         = t0 + dt
        
        sortie[COL_TWA]        = twa
        sortie[COL_HEADING]    = heading
        
        sortie[COL_VOILE]      = voile
        sortie[COL_MEVOILE]    = meilleure_voile
        
        sortie[COL_BOOST]      = boost
        sortie[COL_SPEED]      = speed
        
        sortie[COL_BESTVMGMIN] = bestvmgmin
        sortie[COL_BESTVMGMAX] = bestvmgmax
        sortie[COL_TWD]        = twd
        sortie[COL_TWS]        = tws
        sortie[COL_STAMINA]    = stamina
        
        sortie[COL_PENOSAIL]   = peno_sail_total
        sortie[COL_PENOGYBE]   = peno_gybe_total
        sortie[COL_PENOTACK]   = peno_tack_total
        sortie[COL_ISMER]        = isMER
   
        return sortie
    
    def trajectoire_24h(
        self,
        position_depart,
        progstable,
        duree_h=24,
        dt_standard=60,
        calcMer=False
    ):
        COL_OPTION = 1
        COL_VALEUR = 2
        COL_DTI    = 3
        COL_T0     = 6

        position = position_depart.copy()
        position[COL_DTI] = dt_standard

        t_depart = int(position[COL_T0])

        progs_originales = sorted(
            progstable.get("progs", []),
            key=lambda x: x[0]
        )

        if not progs_originales:
            raise ValueError("Aucune programmation fournie")

        # La première ligne doit représenter l'état initial
     
        # if is_prog:
        #     raise ValueError(
        #        "Attention : la première ligne est marquée True, "
        # "elle sera néanmoins utilisée comme état de départ."

        #     )

        # -------------------------------------------------
        # Initialisation de la position de départ
        # -------------------------------------------------
        ts_reference, option_initiale, valeur_initiale, _ = progs_originales[0]


        position[COL_OPTION] = option_initiale
        position[COL_VALEUR] = valeur_initiale

        # -------------------------------------------------
        # Recalage temporel de toutes les programmations
        #
        # La première programmation est placée à t_depart.
        # Les autres gardent leur décalage temporel relatif.
        # -------------------------------------------------
        progs = []

            
        for timestamp, option, valeur, is_prog in progs_originales[1:]:
            decalage = int(timestamp) - int(ts_reference)
            nouveau_timestamp = t_depart + decalage

            progs.append([
                nouveau_timestamp,
                option,
                valeur,
                True
            ])
                
        
        print( "\nDépart :", time.strftime("%d %b %H:%M:%S", time.localtime(t_depart)),"option :", position[COL_OPTION], "valeur :",  position[COL_VALEUR]    )
        print("\nProgrammations recalées :")

        for prog in progs:
            print(  time.strftime(  "%d %b %H:%M:%S",  time.localtime(prog[0])  ), "option :", prog[1], "valeur :", prog[2],"isProg :",  prog[3]  )

        n_steps = int(duree_h * 3600 // dt_standard)

        positions = np.zeros( (n_steps + 1, position.shape[0]), dtype=position.dtype  )
        positions[0] = position.copy()
        iprog = 0

        for i in range(n_steps):
            t0 = int(position[COL_T0])
            # Application des programmations atteintes
            while iprog < len(progs) and progs[iprog][0] <= t0:
                timestamp, option, valeur, is_prog = progs[iprog]
                if is_prog:
                    position[COL_OPTION] = option
                    position[COL_VALEUR] = valeur
                    print( "Application programmation",  iprog,  time.strftime( "%d %b %H:%M:%S", time.localtime(timestamp) ),"option :", option,"valeur :", valeur )
                iprog += 1
            position[COL_DTI] = dt_standard
            position_suivante = self.deplacement(  position,  calcMer   )
            positions[i + 1] = position_suivante.copy()
            position = position_suivante.copy()
        return positions




    def positions_pour_affichage(self, positions):
        print ('on applique positions _pour affichage ******************************************************************************')

        COL_INDICE     = 0
        COL_OPTION     = 1
        COL_VALEUR     = 2
        COL_DTI        = 3
        COL_LAT        = 4
        COL_LON        = 5
        COL_T0         = 6
        COL_TWA        = 7
        COL_HEADING    = 8
        COL_VOILE      = 9
        COL_MEVOILE    = 10
        COL_BOOST      = 11
        COL_SPEED      = 12
        COL_BESTVMGMIN = 13
        COL_BESTVMGMAX = 14
        COL_TWD        = 15
        COL_TWS        = 16
        COL_STAMINA    = 17
        COL_PENOSAIL   = 18
        COL_PENOGYBE   = 19
        COL_PENOTACK   = 20
        COL_ISMER        = 21


       
        cols_a_remonter = [

            COL_OPTION,
            COL_VALEUR,
            COL_TWA,
            COL_HEADING,
            COL_VOILE,
            COL_MEVOILE,
            COL_BOOST,
            COL_SPEED,                     #je ne remonte pas la speed qui correspond au tws et twd 
            #COL_BESTVMGMIN,
            #COL_BESTVMGMAX,
            #COL_TWD,       
            #COL_TWS,       
            COL_STAMINA,
            COL_PENOSAIL,
            COL_PENOGYBE,
            COL_PENOTACK,
            #COL_ISMER,
        ]
    
        aff = positions.copy()
    
        # On remonte les valeurs de la ligne i vers i-1,
        # mais seulement à partir de i > 1 pour ne pas modifier le départ.
        for i in range(2, positions.shape[0]):
            aff[i - 1, cols_a_remonter] = positions[i, cols_a_remonter]
            
            if aff[i,COL_BOOST]>1.014:
                # print ('position' ,i,'boost>1.014')
                aff[i-1,COL_BOOST]=1
    
        return aff



    
       
        
    def trajectoire_waypoints(self, position_depart, programmations, duree_h=24, dt_standard=60):
    
        COL_INDICE     = 0
        COL_OPTION     = 1
        COL_VALEUR     = 2
        COL_DTI        = 3
        COL_LAT        = 4
        COL_LON        = 5
        COL_T0         = 6
        COL_TWA        = 7
        COL_HEADING    = 8
        COL_VOILE      = 9
        COL_MEVOILE    = 10
        COL_BOOST      = 11
        COL_SPEED      = 12
        COL_BESTVMGMIN = 13
        COL_BESTVMGMAX = 14
        COL_TWD        = 15
        COL_TWS        = 16
        COL_STAMINA    = 17
        COL_PENOSAIL   = 18
        COL_PENOGYBE   = 19
        COL_PENOTACK   = 20
        COL_ISMER        = 21
    
        if "wps" not in programmations:
            return [], [], np.array([position_depart.copy()])
    
        progswaypoints = programmations["wps"]
        position = position_depart.copy()
    
        WP = [[position[COL_LAT], position[COL_LON]]]
    
        for elt in progswaypoints:
            WP.append([elt["lat"], elt["lon"]])
    
        nmax = int(duree_h * 3600 // dt_standard) + len(WP) + 10
    
        positions = np.zeros(
            (nmax, position_depart.shape[0]),
            dtype=position_depart.dtype
        )
    
        # Cap initial vers le premier waypoint
        if len(WP) > 1:
            cap_initial = calcul_cap_grand_cercle(
                position[COL_LAT],
                position[COL_LON],
                WP[1][0],
                WP[1][1]
            )
    
            position[COL_OPTION]  = 0
            position[COL_VALEUR]  = cap_initial
            position[COL_HEADING] = cap_initial
    
        positions[0] = position.copy()
        idx = 0
        sections = []
    
        for j in range(len(WP) - 1):
    
            lat_wp = WP[j + 1][0]
            lon_wp = WP[j + 1][1]
    
            section = {
                "wp_depart": WP[j],
                "wp_arrivee": WP[j + 1],
                "idx_start": idx,
                "y0": positions[idx, COL_LAT],
                "x0": positions[idx, COL_LON],
                "t_start": positions[idx, COL_T0],
                "twa": positions[idx, COL_TWA],
                "cap": positions[idx, COL_HEADING],
                "stamina": positions[idx, COL_STAMINA],
                "voile": positions[idx, COL_VOILE],
                "meilleure_voile": positions[idx, COL_MEVOILE],
                "twd": positions[idx, COL_TWD],
                "tws": positions[idx, COL_TWS],
                "peno_sail": positions[idx, COL_PENOSAIL],
                "peno_gybe": positions[idx, COL_PENOGYBE],
                "peno_tack": positions[idx, COL_PENOTACK],
            }
    
            while idx < nmax - 1:
    
                position = positions[idx].copy()
    
                lat0 = position[COL_LAT]
                lon0 = position[COL_LON]
    
                # Cap utilisé pour ce déplacement
                cap = calcul_cap_grand_cercle(lat0, lon0, lat_wp, lon_wp)
    
                position[COL_OPTION] = 0
                position[COL_VALEUR] = cap
                position[COL_HEADING] = cap
                position[COL_DTI] = dt_standard
    
                dist_wp = distance_haversine(lat0, lon0, lat_wp, lon_wp)
                speed = position[COL_SPEED]
    
                temps_restant = dist_wp / (speed + 1e-6) * 3600.0
    
                if temps_restant <= dt_standard:
    
                    position[COL_DTI] = max(1.0, temps_restant)
    
                    position_suivante = self.deplacement(position)
    
                    # On force exactement le waypoint
                    position_suivante[COL_LAT] = lat_wp
                    position_suivante[COL_LON] = lon_wp
                    position_suivante[COL_DTI] = dt_standard
    
                    # Au waypoint, on garde le cap d'arrivée
                    position_suivante[COL_OPTION] = 0
                    position_suivante[COL_VALEUR] = cap
                    position_suivante[COL_HEADING] = cap
    
                    idx += 1
                    positions[idx] = position_suivante.copy()
                    break
    
                else:
    
                    position_suivante = self.deplacement(position)
    
                    # Cap de sortie recalculé depuis le point final vers le waypoint
                    cap_sortie = calcul_cap_grand_cercle(
                        position_suivante[COL_LAT],
                        position_suivante[COL_LON],
                        lat_wp,
                        lon_wp
                    )
    
                    position_suivante[COL_OPTION] = 0
                    position_suivante[COL_VALEUR] = cap_sortie
                    position_suivante[COL_HEADING] = cap_sortie
    
                    idx += 1
                    positions[idx] = position_suivante.copy()
    
            section.update({
                "idx_end": idx,
                "y1": positions[idx, COL_LAT],
                "x1": positions[idx, COL_LON],
                "t_end": positions[idx, COL_T0],
                "duree": positions[idx, COL_T0] - section["t_start"],
                "stam_end": positions[idx, COL_STAMINA],
                "voile_end": positions[idx, COL_VOILE],
                "meilleure_voile_end": positions[idx, COL_MEVOILE],
                "twd_end": positions[idx, COL_TWD],
                "tws_end": positions[idx, COL_TWS],
                "twa_end": positions[idx, COL_TWA],
                "cap_end": positions[idx, COL_HEADING],
                "peno_sail_end": positions[idx, COL_PENOSAIL],
                "peno_gybe_end": positions[idx, COL_PENOGYBE],
                "peno_tack_end": positions[idx, COL_PENOTACK],
            })
    
            sections.append(section)
    
        positions = positions[:idx + 1]
    
        routeprogs = positions[:, [
            COL_T0,
            COL_OPTION,
            COL_VALEUR,
            COL_LAT,
            COL_LON,
            COL_VOILE,
            COL_MEVOILE,
            COL_TWA,
            COL_HEADING,
            COL_SPEED,
            COL_TWD,
            COL_TWS,
        ]].tolist()
    
        return sections, routeprogs, positions



    def base_carte_10deg(self, lat, lon):
        """
        Retourne le coin haut-gauche du carré 10° x 10°
        contenant lat/lon.
        """
        lat_base = np.ceil(lat / 10.0) * 10.0
        lon_base = np.floor(lon / 10.0) * 10.0
        return lat_base, lon_base
    
    
    def point_dans_carte_10deg(self, lat, lon, lat_base, lon_base):
        """
        Carte 10° x 10° avec coin haut-gauche :
        lat_base, lon_base
        """
        return (
            lat_base - 10.0 <= lat <= lat_base
            and lon_base <= lon <= lon_base + 10.0
        )
    
    
    def get_carte_segment(self, y0, x0, y1, x1):
        """
        Retourne une carte couvrant le segment y0/x0 -> y1/x1.
        Utilise un cache simple dans self.
        """
    
        lat_base0, lon_base0 = self.base_carte_10deg(y0, x0)
        lat_base1, lon_base1 = self.base_carte_10deg(y1, x1)
    
        # Cas courant : les deux points sont dans le même carré 10°
        if lat_base0 == lat_base1 and lon_base0 == lon_base1:
    
            if (
                not hasattr(self, "carte_lat_base")
                or not hasattr(self, "carte_lon_base")
                or self.carte_lat_base != lat_base0
                or self.carte_lon_base != lon_base0
            ):
                cartenp = get_carte3(y0, x0)
                self.cartepy = cartenp.tolist()
                self.carte_lat_base = lat_base0
                self.carte_lon_base = lon_base0
    
            return self.cartepy
    
        # Cas rare : le segment traverse une limite de carte
        carte0 = get_carte3(y0, x0).tolist()
        carte1 = get_carte3(y1, x1).tolist()
    
        return carte0 + carte1


    def extrait_polaires_voile(self,i):
       
            # print (self.polairesjson['sail'][i]['speed'])
        self.polaires_voile=self.polairesjson['sail'][i]['speed'] 
        # pour 
        return self.polaires_voile
       


    def calc_voile(self,voile_ant ,tws,twa ):
        ''' determine
        voile_def,speed,voile_max,boost
        la voile definitive voiledef 
        sa vitesse speed tenant compte du boost eventuel  
        la meilleure voile voile_max et le boost eventuel par rapport  
        '''
        tabtws = np.asarray(self.tabtws, dtype=float)
        tabtwa = np.asarray(self.tabtwa, dtype=float)
        
        voile_auto=1
        # On limite aux bornes disponibles
        tws = np.clip(tws, self.tabtws[0], self.tabtws[-1])
        twa = np.clip(abs(twa), self.tabtwa[0], self.tabtwa[-1])
    
        # Indices encadrants TWS
        j1 = np.searchsorted(self.tabtws, tws)
        j0 = max(0, j1 - 1)
        j1 = min(j1, len(self.tabtws) - 1)
    
        # Indices encadrants TWA
        i1 = np.searchsorted(self.tabtwa, twa)
        i0 = max(0, i1 - 1)
        i1 = min(i1, len(self.tabtwa) - 1)
    
        tws0, tws1 = self.tabtws[j0], self.tabtws[j1]
        twa0, twa1 = self.tabtwa[i0], self.tabtwa[i1]
    
        # Coefficients d'interpolation
        if tws1 == tws0:
            wtws = 0.0
        else:
            wtws = (tws - tws0) / (tws1 - tws0)
    
        if twa1 == twa0:
            wtwa = 0.0
        else:
            wtwa = (twa - twa0) / (twa1 - twa0)

        nb=self.nbvoiles
        speed=np.zeros(nb)
        for i in range (nb): 
            P = np.asarray(self.extrait_polaires_voile(i), dtype=float)
            # Valeurs aux 4 coins
            v00 = P[i0, j0]
            v01 = P[i0, j1]
            v10 = P[i1, j0]
            v11 = P[i1, j1]
        
            # Interpolation bilinéaire
            v0 = v00 * (1 - wtws) + v01 * wtws
            v1 = v10 * (1 - wtws) + v11 * wtws
            res=self.hull*(v0 * (1 - wtwa) + v1 * wtwa)
            # print ('res',res)
            # print ('coeff',foil(twa,tws,carabateau))
            # print ('coeffFoil',self.foil(twa,tws))
            speed[i]=round(self.foil(twa,tws)*res,5)   #*self.hull
        # print ('speed ',speed)
        # print(speed.dtype)
        # print ('voile_ant',voile_ant)
        voile_ant      =int(voile_ant)%10
        speed_voile_ant= speed[voile_ant]
        speed_max      = np.max(speed)
        voile_max      = np.argmax(speed)
        boost          = speed_max/(speed_voile_ant+0.0001)
        
        if boost>1.014 :
            voile_def=voile_max
        else:
            voile_def=voile_ant
            speedf = speed_max 
        
        if voile_auto!=1:
            voile_def =voile_ant
            speed_max= speed_voile_ant

        return voile_def,speed_max,voile_max,boost




        
    def foil(self, twa, tws):
        """Calcule le coefficient foil en fonction de twa et tws."""

        twa = abs(twa)
    
        def rampe(x, xmin, xmax, merge):
            """
            Retourne un coefficient entre 0 et 1 :
            - 0 hors zone élargie
            - 1 dans la zone pleine [xmin, xmax]
            - interpolation linéaire dans les zones merge
            """
    
            if x <= xmin - merge:
                return 0.0
    
            if x < xmin:
                return (x - (xmin - merge)) / merge
    
            if x <= xmax:
                return 1.0
    
            if x < xmax + merge:
                return (xmax + merge - x) / merge
    
            return 0.0
    
        coeff_twa = rampe(twa, self.twaMin, self.twaMax, self.twaMerge)
        coeff_tws = rampe(tws, self.twsMin, self.twsMax, self.twsMerge)
    
        coeff_activation = coeff_twa * coeff_tws
    
        coeff = 1.0 + (self.speedRatio - 1.0) * coeff_activation
    
        return coeff
   



    def controle_terre_trajectoire(self, positions):
        """
        positions : tableau avec lat/lon en colonnes COL_LAT / COL_LON
        Ajoute ou remplit une colonne isMer :
          1 = segment coupe la terre
          0 = segment ne coupe pas la terre
        """
    
        COL_LAT = 4   # à adapter
        COL_LON = 5   # à adapter
        COL_ISMER = 21  # à adapter si tu as déjà une colonne
    
        n = len(positions)
    
        # par défaut : pas d'intersection terre
        positions[:, COL_ISMER] = 0
    
        # cache local des cartes
        cartes = {}
    
        for i in range(n - 1):
            y0 = positions[i, COL_LAT]
            x0 = positions[i, COL_LON]
            y1 = positions[i + 1, COL_LAT]
            x1 = positions[i + 1, COL_LON]
    
            lat_base0, lon_base0 = self.base_carte_10deg(y0, x0)
            lat_base1, lon_base1 = self.base_carte_10deg(y1, x1)
    
            segment = [[y0, x0], [y1, x1]]
    
            if lat_base0 == lat_base1 and lon_base0 == lon_base1:
                key = (lat_base0, lon_base0)
    
                if key not in cartes:
                    cartes[key] = get_carte3(y0, x0).tolist()
    
                cartepy = cartes[key]
    
            else:
                # cas rare : segment sur frontière de carte
                key0 = (lat_base0, lon_base0)
                key1 = (lat_base1, lon_base1)
    
                if key0 not in cartes:
                    cartes[key0] = get_carte3(y0, x0).tolist()
    
                if key1 not in cartes:
                    cartes[key1] = get_carte3(y1, x1).tolist()
    
                cartepy = cartes[key0] + cartes[key1]
                
            
    
            coupe_terre = intersection(segment, cartepy)
    
            # dans ton code actuel : 1 si coupe terre, 0 sinon
            positions[i + 1, COL_ISMER] = 1 if coupe_terre else 0

        # cartepy=np.asarray(cartepy)
        # print ('cartepy.shape',cartepy.shape)
    
        return positions

    def masque_cellules_carte(self,carte_np, lat_base, lon_base, n=1000):
        """
        carte_np : shape (N, 2, 2)
            carte_np[:, 0, :] = point C [lat, lon]
            carte_np[:, 1, :] = point D [lat, lon]
    
        lat_base, lon_base : coin supérieur gauche de la carte 10°
            exemple lat_base = 50, lon_base = -10
    
        n = 1000 pour pas de 0.01°
            n = 100  pour pas de 0.1°
    
        Retourne :
            masque shape (n, n), bool
            True = au moins un segment de côte touche cette cellule
        """
    
        masque = np.zeros((n, n), dtype=bool)
    
        pas = 10.0 / n
    
        lat0 = carte_np[:, 0, 0]
        lon0 = carte_np[:, 0, 1]
        lat1 = carte_np[:, 1, 0]
        lon1 = carte_np[:, 1, 1]
    
        # indices ligne/colonne des extrémités
        # ligne 0 = nord / haut de carte
        i0 = np.floor((lat_base - lat0) / pas).astype(int)
        j0 = np.floor((lon0 - lon_base) / pas).astype(int)
    
        i1 = np.floor((lat_base - lat1) / pas).astype(int)
        j1 = np.floor((lon1 - lon_base) / pas).astype(int)
    
        i0 = np.clip(i0, 0, n - 1)
        i1 = np.clip(i1, 0, n - 1)
        j0 = np.clip(j0, 0, n - 1)
        j1 = np.clip(j1, 0, n - 1)
    
        for a, b, c, d in zip(i0, j0, i1, j1):
            # segment vertical
            if b == d:
                imin = min(a, c)
                imax = max(a, c)
                masque[imin:imax + 1, b] = True
    
            # segment horizontal
            elif a == c:
                jmin = min(b, d)
                jmax = max(b, d)
                masque[a, jmin:jmax + 1] = True
    
            else:
                # normalement inutile chez toi si les segments sont uniquement H/V
                imin = min(a, c)
                imax = max(a, c)
                jmin = min(b, d)
                jmax = max(b, d)
                masque[imin:imax + 1, jmin:jmax + 1] = True
    
        return masque




        

    def get_carte_cache(self, y0, x0, n=1000):
        """
        Retourne la carte 10° + son masque.
        Cache par carré 10°.
        """
    
        lat_base, lon_base = self.base_carte_10deg(y0, x0)
        key = (lat_base, lon_base, n)
    
        if key not in self.carte_cache:
            carte_np = fcarte3(y0, x0)   # shape (N, 2, 2)
    
            masque = self.masque_cellules_carte(
                carte_np,
                lat_base,
                lon_base,
                n=n
            )
    
            self.carte_cache[key] = {
                "lat_base": lat_base,
                "lon_base": lon_base,
                "segments": carte_np,
                "masque": masque,
            }
    
        return self.carte_cache[key]
            
    
  
# fin de la classe Simulateur
#############################    


#Classe CarteCache

class CarteCache:
    def __init__(self):
        self.cache_masques = {}

    def angle_superieur_gauche(self, lat, lon):
        lat0 = int(10 * (lat // 10) + 10)
        lon0 = int(10 * (lon // 10))
        return lat0, lon0

    def masque_segments_10deg(self, carte, lat0, lon0, n=1460):
        masque = np.zeros((n, n), dtype=bool)
        scale = n / 10.0
    
        lat_min = carte[:, :, 0].min()
        lat_max = carte[:, :, 0].max()
        lon_min = carte[:, :, 1].min()
        lon_max = carte[:, :, 1].max()
    
        print("lat0 lon0 :", lat0, lon0)
        print("lat min/max carte :", lat_min, lat_max)
        print("lon min/max carte :", lon_min, lon_max)
    
        # Cas 1 : carte en coordonnées locales 0..10
        carte_locale = (
            0 <= lat_min <= 10 and 0 <= lat_max <= 10 and
            0 <= lon_min <= 10 and 0 <= lon_max <= 10
        )
    
        print("carte_locale :", carte_locale)
    
        for seg in carte:
            lat1, lon1 = seg[0]
            lat2, lon2 = seg[1]
    
            if carte_locale:
                # Ici lat/lon sont supposées locales dans la tuile
                # lat locale : 0 en haut, 10 en bas
                i1 = int(np.floor(lat1 * scale))
                j1 = int(np.floor(lon1 * scale))
                i2 = int(np.floor(lat2 * scale))
                j2 = int(np.floor(lon2 * scale))
            else:
                # Ici lat/lon sont absolues
                i1 = int(np.floor((lat0 - lat1) * scale))
                j1 = int(np.floor((lon1 - lon0) * scale))
                i2 = int(np.floor((lat0 - lat2) * scale))
                j2 = int(np.floor((lon2 - lon0) * scale))
    
            i1 = max(0, min(n - 1, i1))
            i2 = max(0, min(n - 1, i2))
            j1 = max(0, min(n - 1, j1))
            j2 = max(0, min(n - 1, j2))
    
            imin, imax = sorted((i1, i2))
            jmin, jmax = sorted((j1, j2))
    
            masque[imin:imax + 1, jmin:jmax + 1] = True
    
        return masque

    def get_masque_carte(self, lat, lon, get_carte3):
        tic = time.time()
    
        lat0, lon0 = self.angle_superieur_gauche(lat, lon)
        key = (lat0, lon0)
    
        # print("key masque :", key)
    
        if key not in self.cache_masques:
            # print("le masque n etait pas dans le cache")
    
            lat_carte = lat0 - 5
            lon_carte = lon0 + 5
    
            # print("appel get_carte3 avec :", lat_carte, lon_carte)
    
            carte = get_carte3(lat_carte, lon_carte)
    
            # print("lat min/max carte :", carte[:, :, 0].min(), carte[:, :, 0].max())
            # print("lon min/max carte :", carte[:, :, 1].min(), carte[:, :, 1].max())
    
            self.cache_masques[key] = self.masque_segments_10deg(
                carte,
                lat0,
                lon0
            )
    
        else:
            print("Utilisation du cache*****************************")    
        print("temps de recherche {} s".format(time.time() - tic))
    
        return self.cache_masques[key]    


    def point_to_ij(self,lat, lon, lat0, lon0, n=730):
        scale = n / 10.0
    
        i = np.floor((lat0 - lat) * scale).astype(int)
        j = np.floor((lon - lon0) * scale).astype(int)
    
        return i, j

    def segments_touchent_masque(self,positions, masque, lat, lon):
        """
        Retourne un booléen par segment de route.
        True  -> le segment traverse au moins une case True du masque
        False -> aucune case True dans sa bbox
        """
        lat0, lon0 = self.angle_superieur_gauche(lat, lon)
        pts = positions[:, [4, 5]]  # lat, lon
    
        lat1 = pts[:-1, 0]
        lon1 = pts[:-1, 1]
        lat2 = pts[1:, 0]
        lon2 = pts[1:, 1]
    
        n = masque.shape[0]
    
        i1, j1 = self.point_to_ij(lat1, lon1, lat0, lon0, n)
        i2, j2 = self.point_to_ij(lat2, lon2, lat0, lon0, n)
    
        # clamp dans le masque
        i1 = np.clip(i1, 0, n - 1)
        i2 = np.clip(i2, 0, n - 1)
        j1 = np.clip(j1, 0, n - 1)
        j2 = np.clip(j2, 0, n - 1)
    
        imin = np.minimum(i1, i2)
        imax = np.maximum(i1, i2)
        jmin = np.minimum(j1, j2)
        jmax = np.maximum(j1, j2)
    
        touche = np.zeros(len(lat1), dtype=bool)
    
        for k in range(len(lat1)):
            touche[k] = masque[
                imin[k]:imax[k] + 1,
                jmin[k]:jmax[k] + 1
            ].any()
    
        return touche


    def segments_se_croisent(self,A, B, C, D):
        """
        Teste si les segments [A,B] et [C,D] se croisent
        A, B, C, D : [lat, lon]
        """
        def ccw(P, Q, R):
            return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))
    

    def intersection(self,segment, carte, marge=0.1):
        [[lat0, lon0], [lat1, lon1]] = segment
    
        # Fenêtre élargie autour du segment
        latmin = min(lat0, lat1) - marge
        latmax = max(lat0, lat1) + marge
        lonmin = min(lon0, lon1) - marge
        lonmax = max(lon0, lon1) + marge
    
        A, B = segment
    
        for C, D in carte:
            # Si le segment côtier est hors de la zone élargie, on saute
            if max(C[0], D[0]) < latmin or min(C[0], D[0]) > latmax:
                continue
            if max(C[1], D[1]) < lonmin or min(C[1], D[1]) > lonmax:
                continue
    
            if self.segments_se_croisent(A, B, C, D):
                return True
    
        return False     
        

    def indices_intersections_route(self,positions, carte, touche, marge=0.1):
        """
        Retourne les indices k tels que le segment :
            positions[k] -> positions[k+1]
        croise un segment de côte.
        """
    
        indices_touche = np.where(touche)[0]
        indices_croise = []
    
        for k in indices_touche:
            segment = [
                [positions[k, 4],     positions[k, 5]],
                [positions[k + 1, 4], positions[k + 1, 5]]
            ]
    
            if self.intersection(segment, carte, marge=marge):
                indices_croise.append(k)
    
        return np.array(indices_croise, dtype=int)        

   
   
###################################################
#    Fin de Classe CarteCache
################################################## #








def routageGlobal(course,user_id,isMe,ari,y0,x0,t0,tolerancehvmg,optionroutage,mode):                              # version Vrouteur 1/12/2025
    ''' Calcule le routage '''
    ''' Recupère les données et parcoure les differents waypoints '''

    print ('Calcul du routage ')

    session         = RoutageSession(course, user_id,isMe,ari) 
    waypoints       = session.waypoints
    tabvmg10to      = session.tabvmg10to
    exclusions      = session.exclusions_json

    if optionroutage==1:
        # coordonnees heure manuelle 
        print ('partie a revoir ')
       
   
    if optionroutage==2:
        # coordonnees heure de depart 
        print ('partie a revoir ')

    print ('Dans isMe {} routage global y0 {},x0{} t0 {}'.format(isMe, y0,x0,t0))    
    

    for indiceroutage in range(len(ari)):     # Parcourt les differents  waypoints        
        if indiceroutage==0: 
            iso_record=session.calculePosDepart(dt=60)
            points=iso_record['points'] 
            if isMe=='no':
                ecart=t0-session.t0vr
                dtig=  session.dtig0GFS + ecart
                tws,twd = gfs_interpolateur_np(dtig,y0,x0) 
                #torch.tensor ([[0,0,y0,x0,voile,twa,stamina,soldepeno,tws,twd]], dtype=torch.float32, device='cuda')
                isodepart       = torch.tensor ([[0,0,y0,x0,0,0,100,0,tws,twd]], dtype=torch.float32, device='cuda')
                iso_record = {
                "numiso"  : 0,
                "ecart"   : ecart,          # secondes ou minutes depuis t0
                "points"  : isodepart,           # tenseur uniquement avec les colonnes utiles par point
                "tmini"   : 10000,
                "distmini": 1000,
                "nptmini" : 0,
                "idxptmini": 0
              }



        else:
            point=session.isoglobal[npt_proche]
            points= point.unsqueeze(0)      # on rajoute une dimension pour transformer en isochrone 
            iso_record = {
                "numiso"  : numiso,
                "ecart"   : ecart,          # secondes ou minutes depuis t0
                "points"  : points,           # tenseur uniquement avec les colonnes utiles par point
                "tmini"   : tmini,
                "distmini": distmini,
                "nptmini" : nptmini,
                "idxptmini": npt_proche
              }

            
        
        session.initialiseRoutagePartiel(indiceroutage,iso_record )        
        rwp              = session.rwp
        tmini            = 10000
        distmini         = 1000       
        n                = 0
         
        
        while distmini > rwp:      
            iso_record = session.isoplusun(iso_record,mode) 
            distmini    = iso_record['distmini']  
            # n+=1
            # print (n)          
           # print ('n  {} tmini {:6.0f} distmini {:6.1f} nptmini{} '.format(n,tmini,distmini,nptmini))              
          
        # Recuperation des donnees d arrivee sur le wp
        numiso     = iso_record['numiso']
        ecart      = iso_record['ecart']
        nptmini    = iso_record['nptmini']
        tmini      = iso_record['tmini']
        distmini   = iso_record['distmini']
        npt_proche = iso_record['idxptmini']
        session.wp_passages[indiceroutage] = {
                                                "numiso": numiso,
                                                "ecart": float(ecart),
                                                "npt": int(nptmini),
                                                "rwp":   rwp,    
                                                "distmini": distmini,
                                                "tmini":tmini            
                                            }
      

    print(' session.wp_passages ', session.wp_passages)
        
    # tac=time.time()    
    # print ('temps exe SIMPLE  {:6.0f}s pour {} isos ,par iso {:6.5f}s '.format(tac-tic,n,(tac-tic)/n))   
    session.isoglobal         = session.isoglobal[:session.kglobal].clone()
    chemin_indices            = session.reconstruire_chemin_rapide(session.isoglobal, nptmini)  
    routage                   = session.construit_routage_depuis_chemin(chemin_indices)
    dico_isochrones           = session.construit_dico_isochrones_1sur6( 0.05)
    
    arrayroutage              = routage.cpu().tolist()                      # passage de torch a tableau python
    routage_np                = np.array(arrayroutage,dtype=np.float64)
    tabpointslisses           = session.lissage_routage_np( routage_np) 
    routage_lisse             = session.reconstruit_colonnes_routage_lisse(tabpointslisses, mode="gfs")
    routage_actions           = session.routage_actions(routage_lisse)
          
    routage_list              = routage_actions.cpu().tolist()
    return  waypoints,exclusions,routage_list,dico_isochrones
 







@app.route('/calculeroutage', methods=["GET", "POST"])
# c 'est celui qui est utilise par index nouveau modèle
def calculeroutage():
    print('\nDemande de routage sur url calculeroutage\n***************************************')
    

    username            = request.args.get('username')          # recupere les donnees correspondant a valeur dans le site html
    print ('username',username)
    course              = request.args.get('course')              # recupere les donnees correspondant a valeur dans le site html
    user_id             = request.args.get('user_id')  
    aristr              = request.args.get('ari')                # on a besoin de ari pour savoir sur quels waypoints on va 
    isMe                = request.args.get('isMe')
    tolerancehvmg       = float(request.args.get('tolerancehvmg'))
    retardpeno          = float(request.args.get('retardpeno'))
    y0                  = float(request.args.get('y0'))                    
    x0                  = float(request.args.get('x0'))
    t0                  = float(request.args.get('t0'))
    optionroutage       = float(request.args.get('optionroutage'))
    ari                 = json.loads(aristr)
    mode                = request.args.get('mode')
   
    #username            = findUsername(user_id)
    #username            =  users[user_id]
    print ('course                                  ',course)
    print ('username                                ',username)
    print ('user_id                                 ',user_id)
    print ('isMe                                    ',isMe)
    print ('ari                                     ',ari)
    print ('y0 routage                               {:8.4f}'.format(y0))
    print ('x0 routage                               {:8.4f}'.format(x0))
    print ('t0vr                                     {}'.format(time.strftime("%d %b %H:%M ",time.localtime(t0))))
    print ('tolerancehvmg                           ',tolerancehvmg  )
    print ('retardpeno                              ',retardpeno  )
    print ('optionroutage                           ',optionroutage)                           # si option 0 c'est le routage depuis la position vr  si 1 position y0,x0, heure t0 si 2 position y0,x0, heure depart
    print ('mode                                    ',mode     )
    
  

    
   
    tic=time.time()
    waypoints,exclusions,routage_list,dico_isochrones=routageGlobal(course, user_id,isMe,ari,y0,x0,t0,tolerancehvmg,optionroutage,mode)  
    tac=time.time()
    nombreisos=len(routage_list)
    print ('temps exe {} par iso {}  ',tac-tic,(tac-tic)/nombreisos)

    # print (waypoints )
    # print (exclusions )
    # print (routage_list)
    # print (dico_isochrones)

    dico              = {'message':'Routage OK','waypoints':waypoints,'exclusions':exclusions,'arrayroutage':routage_list,'isochrones':dico_isochrones}
    
    eta = routage_list[-1][4]
   
    print('************************************************************************************************\n')
    print ('Routage EFFECTUÉ pour {} sur course {} le {} - {} - ETA {} \n*******************************************************************************************'.format(username,course,format_time(t0),'Routage OK',format_time(eta)))
   
    
    insert_historoutages(  username,course, user_id,dico['message'], t0, y0, x0, eta)
    print ('insere dans historoutage')
    return dico




@app.route('/chargecarte3', methods=["GET", "POST"])

def chargecarte3():
   #print('on est dans chargecarte3')

    # on va recuperer la pos 
    lat      = float(request.args.get('lat'))  
    lon      = float(request.args.get('lon')) 
   
    # angle superieur gauche calcule dans get_carte3 
    # lat0=int(10*(lat//10) +10)
    # lon0=int(10*(lon//10))
   
    tic=time.time()
    
    try:
        cartenp = get_carte3(lat,lon)
        cartepy =cartenp.tolist()           #on transforme en tableau pour transmettre
    
    except:
        print ('la carte demandee est vide \n**********************************************\n')
        cartepy=[] 
    #print ('temps de chargement de la carte ',time.time()-tic)
    
    response=make_response(jsonify({'message':'Cartes envoyees par Serveur','carte':cartepy}))

    # print()
    # print(' Taille de la carte en octets '.sys.getsizeof(cartepy))
    # print()
    response.headers.add('Access-Control-Allow-Origin', '*')  # Autorise toutes les origines
    return response








@app.route('/calculepolairespourjs', methods=["GET", "POST"])
def calculepolaires():
    # recupere la meteo sur le point les vmg et les recouvrements  
    tws      = float(request.args.get('tws'))
    twaini      = float(request.args.get('twa'))
    polar_id = int(request.args.get('polar_id'))
    print()
    #print ('ligne 3932 twa',twaini)

     # on charge le tableau des polaires 

    filenamelocal1='polairesglobales10_'+str(polar_id)+'.npy'
    filename1=basedirnpy+filenamelocal1
    polairesglobales10=get_fichier(filename1)       # version avec cache

#    tws10      = float(request.args.get('tws10'))

# Donnees pour enchainement de manoeuvres    
    tws10      = float(request.args.get('tws10'))
    twa11      = float(request.args.get('twa11'))
    stam10     = float(request.args.get('stam10'))
    man10      = request.args.get('man10')

    tws20      = float(request.args.get('tws20'))
    temps20    = float(request.args.get('temps20'))
    ali20      = request.args.get('ali20')

    tws21      = float(request.args.get('tws21'))
    twa21      = float(request.args.get('twa21'))
    man20      = (request.args.get('man20'))
    tws30      = float(request.args.get('tws30'))
    temps30    = float(request.args.get('temps30'))
    ali30      = request.args.get('ali30')

    tws31      = float(request.args.get('tws31'))
    twa31      = float(request.args.get('twa31'))
    man30      = (request.args.get('man30'))

    v1         = float(request.args.get('v1'))
    v2         = float(request.args.get('v2'))
    pena1      = float(request.args.get('pena1'))
    pena2      = float(request.args.get('pena2'))


    print('Parametres transmis tws10 {} stam10 {} man10 {} tws20 {} temps20 {} ali20 {} tws21 {} man20 {} tws30 {} temps30 {} ali30 {} tws31 {} man30 {}'\
          .format(tws10, stam10 ,man10, tws20, temps20 ,ali20 ,tws21 ,man20 ,tws30, temps30 ,ali30, tws31, man30) )
    staminaini=stam10
    Tws1010=round(tws10*10)
    Twa1011=round(twa11*10)
    Tws1021=round(tws21*10)
    Twa1021=round(twa21*10)
    Tws1031=round(tws31*10)
    Twa1031=round(twa31*10)



    print('man10 ',man10)

    if man10=='Voile' :
        print('cas1')
        Chgt=1
        twam1,twa=90,90
        
    elif man10=='Gybe':
        print('cas2')
        Chgt=0
        twam1,twa=-120,+120
    
    elif man10=='Tack':
        print('cas3')
        Chgt=0
        twam1,twa=-60,+60

    elif man10=='Combo':
        print('cas4')
        Chgt=1
        twam1,twa=-60,+60

    else :    
        print('cas probleme')

    if man20=='Voile' :
        print('cas1')
        Chgt2=1
        twam12,twa2=90,90
        
    elif man20=='Gybe':
        print('cas2')
        Chgt2=0
        twam12,twa2=-120,+120
    
    elif man20=='Tack':
        print('cas3')
        Chgt2=0
        twam12,twa2=-60,+60

    elif man20=='Combo':
        print('cas4')
        Chgt2=1
        twam12,twa2=-60,+60

    else :    
        print('cas probleme')
    

    if man30=='Voile' :
        print('Chgt Voile')
        Chgt3=1
        twam13,twa3=90,90
        
    elif man30=='Gybe':
        print('Gybe')
        Chgt3=0
        twam13,twa3=-120,+120
    
    elif man30=='Tack':
        print('Tack')
        Chgt3=0
        twam13,twa3=-60,+60

    elif man20=='Combo':
        print('Combo')
        Chgt3=1
        twam13,twa3=-60,+60

    else :    
        print('cas probleme')

    print ('Chgt ',Chgt)

    carabateau=cherchecarabateau(polar_id)
    lwtimer =carabateau['lwtimer']
    hwtimer =carabateau['lwtimer']
    lwtimerGybe =carabateau['lwtimerGybe']
    hwtimerGybe =carabateau['hwtimerGybe']
    coeffboat=carabateau['Coeffboat']
    furler=0.8

    TGybe= ((twam1*twa)<0)*1
    # on calcule le a premiere perte de stamina et la premiere penalite 
    peno_chgt = Chgt   *peno_np(lwtimer, hwtimer, tws10, stam10)* furler
    peno_gybe = TGybe  *peno_np(lwtimerGybe, hwtimerGybe,tws10, stam10)   
    peno11=peno_chgt+peno_gybe

    perte_stamina   = calc_perte_stamina_np(tws10, TGybe, Chgt,coeffboat)
    
    stam11         = min((stam10 - perte_stamina ),100)
     

   
    speed= polairesglobales10[7,Tws1010,Twa1011]
    dist11=(speed*peno11/3600)*.3
    peno11=float(peno11)
    
    # recuperation avant manoeuvre2
    recup20=frecupstamina_np(temps20*60,tws20)  
    stam20=stam11
    if ali20=='Barre':
        stam20+=20   
    if ali20=='Repas':
        stam20=100    
    if ali20=='Cafe':
        max=110
    else:
        max=100     

    stam20+=recup20
    stam20=min(max,stam20)

    peno_chgt = Chgt *peno_np(lwtimer, hwtimer, tws, stam20)* furler
#       peno_gybe = Tgybe * *peno_np(lwtimer, hwtimer, Tws, Stamina)   


    # on calcule la penalite de la 2eme manoeuvre
    stam21,peno21=0,0
    speed2= polairesglobales10[7,Tws1021,Twa1021]
    dist21=(speed2*peno21/3600)*.3
    cumul21=dist11+dist21
    peno21=float(peno21)
    # recuperation avant la troisieme manoeuvre
    recup30=frecupstamina_np(temps30*60,tws30)
    stam30=stam21
    if ali30=='Barre':
        stam30+=20   
    if ali30=='Repas':
        stam30=100    
    if ali30=='Cafe':
        max=110
    else:
        max=100     

    stam30+=recup30
    stam30=min(max,stam30)


  

 # on calcule la penalite de la 3eme manoeuvre
    stam31,peno31=0,0
    speed3= polairesglobales10[7,Tws1031,Twa1031]

    peno31=float(peno31)
   
    dist31=float((speed3*peno31/3600)*.3)
    cumul31=cumul21+dist31
    
    print ('ali20:                ',ali20 )
    print ('stam11                 {:6.2f}'.format(stam11))
    print ('penalite               {:6.2f}'.format(peno11))
    print ('speed                  {:6.2f}'.format(speed))
    print ('distance perdue        {:6.2f}'.format(dist11))
    print ('recuperation de points {:6.2f}'.format(recup20))
    print ('stam20                 {:6.2f}'.format(stam20))
    print ('stam21                 {:6.2f}'.format(stam21))
    print ('penalite2              {:6.2f}'.format(peno21))
    print ('speed2                 {:6.2f}'.format(speed2))
    print ('distance perdue        {:6.2f}'.format(dist21))
    print ('cumul distance perdue  {:6.2f}'.format(cumul21))
    print ('recuperation de points {:6.2f}'.format(recup30))
    print ('stam30                 {:6.2f}'.format(stam30))
    print ('stam21                 {:6.2f}'.format(stam21))
    print ('penalite2              {:6.2f}'.format(peno21))
    print ('speed2                 {:6.2f}'.format(speed2))
    print ('distance perdue        {:6.2f}'.format(dist21))
    print ('cumul distance perdue  {:6.2f}'.format(cumul21))
    print ('recuperation de points {:6.2f}'.format(recup30))
    print ('stam30                 {:6.2f}'.format(stam30))
    print ('stam31                 {:6.2f}'.format(stam31))
    print ('penalite3              {:6.2f}'.format(peno31))
    print ('speed3                 {:6.2f}'.format(speed3))
    print ('distance perdue        {:6.2f}'.format(dist31))
    print ('cumul distance perdue  {:6.2f}'.format(cumul31))
    print ()


# on calcule l amortissement en temps pour changement de voile    
    amort=0.3*v2*(pena1+pena2)/abs(v2-v1+0.00001)


    # on charge le tableau des vmg

    filenamelocal2='vmg10_'+str(polar_id)+'.npy'
    filename2=basedirnpy+filenamelocal2
    tabvmg10 = get_fichier(filename2)
    tws10 = round(tws*10)
    twa10 = abs(round(twaini*10))
    # on extrait le tableau des vmg pour la tws et le calcul des voiles pour la twa 
    
    # print ()
    # print ('twa',twaini)
    # print ('twa10',twa10)
    # print('************************************')
    
    vmgpourjs= tabvmg10[tws10,:]
    polairespourjs=polairesglobales10[:,tws10,twa10]  
    
    tabrecouvrements=[]
    vmgpourjs     =  [arr.tolist() for arr in vmgpourjs]
    polairespourjs=  [arr.tolist() for arr in polairespourjs]
    # print('vmgpourjs',vmgpourjs)
    # print('polairespourjs',polairespourjs)

   
    response=make_response(jsonify({"message":"Tout va bien","polairespourjs":polairespourjs,"vmgpourjs":vmgpourjs  ,
                                     "tabrecouvrements":tabrecouvrements,
                                     "stam11":stam11,"peno11":peno11,"dist11":dist11,
                                     "recup20":recup20,"stam20":stam20,
                                     "stam21":stam21,"peno21":peno21,"dist21":dist21,"cumul21":cumul21,
                                     "recup30":recup30,"stam30":stam30,
                                     "stam31":stam31,"peno31":peno31,
                                     "dist31":dist31,
                                    "cumul31":cumul31,
                                     "amort":amort   }))
                                                         
                                    
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response





def cherchecarabateau(polar_id):
    updated,polar_id,polairesjsonstr=rechercheTablePolaires(polar_id)    # nouvelle version
    polairesjson=json.loads(polairesjsonstr) 
    polairesjson=json.loads(polairesjson) 
    #print(' en 3753', polairesjson)
                      
    nbvoiles       = len(polairesjson['sail'])
    print('nbvoiles',nbvoiles)
    typevoile          = []
    for i in range(nbvoiles) :
        typevoile.append( polairesjson['sail'][i]['name'])

    polar_id                    = polairesjson['_id']
    label                       = polairesjson['label']                               # type du bateau
    globalSpeedRatio            = polairesjson['globalSpeedRatio']
    foilSpeedRatio              = polairesjson['foil']['speedRatio']
    autoSailChangeTolerance     = polairesjson["autoSailChangeTolerance"]
    badSailTolerance            = polairesjson["badSailTolerance"]
    iceSpeedRatio               = polairesjson['iceSpeedRatio']
    maxSpeed                    = polairesjson['maxSpeed']
    twaMin                      = polairesjson['foil']['twaMin']
    twaMax                      = polairesjson['foil']['twaMax']
    twaMerge                    = polairesjson['foil']['twaMerge']
    twsMin                      = polairesjson['foil']['twsMin']
    twsMax                      = polairesjson['foil']['twsMax']
    twsMerge                    = polairesjson['foil']['twsMerge']
    hull                        = polairesjson['hull']['speedRatio']
    lws                         = polairesjson['winch']['lws']
    hws                         = polairesjson['winch']['hws']
    lwtimer                     = polairesjson['winch']['sailChange']['pro']['lw']['timer']
    hwtimer                     = polairesjson['winch']['sailChange']['pro']['hw']['timer']
    lwratio                     = polairesjson['winch']['sailChange']['pro']['lw']['ratio']
    hwratio                     = polairesjson['winch']['sailChange']['pro']['hw']['ratio']
    tackprolwtimer              = polairesjson['winch']['tack']['pro']['lw']['timer']
    tackprolwratio              = polairesjson['winch']['tack']['pro']['lw']['ratio']
    tackprohwtimer              = polairesjson['winch']['tack']['pro']['hw']['timer']
    tackprohwratio              = polairesjson['winch']['tack']['pro']['hw']['ratio']
    gybeprolwtimer              = polairesjson['winch']['gybe']['pro']['lw']['timer']
    gybeprolwratio              = polairesjson['winch']['gybe']['pro']['lw']['ratio']
    gybeprohwtimer              = polairesjson['winch']['gybe']['pro']['hw']['timer']
    gybeprohwratio              = polairesjson['winch']['gybe']['pro']['hw']['ratio']

    coeffboat                   = tabCoeffboat[polar_id]['stamina']

    carabateau={ "polar_id":polar_id,'typevoile':typevoile, "label":label,"globalSpeedRatio":globalSpeedRatio,"foilSpeedRatio":foilSpeedRatio,"coeffboat":coeffboat,\
            "iceSpeedRatio":iceSpeedRatio,"autoSailChangeTolerance":autoSailChangeTolerance,"badSailTolerance":badSailTolerance,\
            "maxSpeed":maxSpeed,"twaMin":twaMin,"twaMax":twaMax,"twaMerge":twaMerge,"twsMin":twsMin,"twsMax":twsMax,"twsMerge":twsMerge,\
            "hull":hull,'lws': lws ,'hws':hws, "lwtimer":lwtimer,"hwratio": hwratio ,"hwtimer":hwtimer,"lwratio":lwratio,\
            "lwtimer":lwtimer,"tackprolwtimer":tackprolwtimer,"tackprolwratio":tackprolwratio,"tackprohwtimer":tackprohwtimer,\
            "tackprohwratio":tackprohwratio,"gybeprolwtimer":gybeprolwtimer,"gybeprolwratio":gybeprolwratio,"gybeprohwtimer":gybeprohwtimer,"gybeprohwratio":gybeprohwratio,'typevoile':typevoile}

    return carabateau






@app.route('/rechercheflotte', methods=["GET", "POST"])
def rechercheflotte():
    # print ('On est dans rechercheflotte')
    id_user    = request.args.get('id_user')                                 # recupere les donnees correspondant a valeur dans le site html
    course     = request.args.get('course')                                 # recupere les donnees correspondant a valeur dans le site
  
    # print ('id_user {} course {} '.format(id_user,course))
    fleetinfos=rechercheTableFleetinfos(id_user, course)

    return jsonify({"message": 'Recuperation de la flotte effectué avec succes '  ,"fleetinfos":fleetinfos }), 200








if __name__ == "__main__":
    #socketio.run(app, host='127.0.0.1', port=5000, debug=False)
    # if socketioOn:
    #     print("socketio async_mode =", socketio.async_mode)
    #     socketio.run(
    #     app,
    #     host='0.0.0.0' if IS_PRODUCTION else '127.0.0.1',
    #     port=5000,
    #     debug=False
    #     )
    # else:
    #     app.run(host='0.0.0.0', port=5000, debug=False)
    
    socketio.run(app, host="127.0.0.1", port=5000, debug=True)
   