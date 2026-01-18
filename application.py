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
import requests
import xarray as xr
import copy
import logging
import torch
import gc

from datetime import datetime,timezone
from matplotlib.path import Path
from flask import (Flask, flash, jsonify, redirect, render_template,make_response, request,session, url_for)
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_caching import Cache
from scipy.ndimage import label, generate_binary_structure

from numba import njit
from numba import prange
from flask_cors import CORS
from global_land_mask import globe
from shapely.geometry import (LineString, MultiLineString, MultiPoint, Point,Polygon)
from shapely.strtree import STRtree
from apscheduler.schedulers.background import BackgroundScheduler
from websocket import create_connection
from scipy.signal import savgol_filter
from cachetools import TTLCache
from threading import Lock
from collections import defaultdict, OrderedDict, deque     # necessaire pour la carte 


from fonctions2024 import *
from fonctions2025 import *

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


# 80 Gestion du websocket 
# 166 Gestion du grib
# 223 Cache pour les fichiers
# 380 class CpuGpuCache:



# test pour commit dans tests modifs



socketioOn=True
# if socketioOn:
#socketio = SocketIO(app, logger=True, engineio_logger=True)
IS_PRODUCTION = os.environ.get("FLASK_ENV") == "production"
hostname = socket.gethostname()


ECM_actif=True

print ('is PRODUCTION',IS_PRODUCTION)
print("ENV =", os.getenv("VROUTEUR_ENV"))
################################################################################################
################      Gestion du websocket     #################################################
################################################################################################


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


socketio = SocketIO(app, cors_allowed_origins="*",logger=True, engineio_logger=True)   # ou 'gevent'
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



###############      Gestion du grib          #################################################
#################################################################################################
################################################################################################

#________________________________________________________________________
#  Chargement GFS
#________________________________________________________________________



#__________________________________________________________________________________
# Partie GFS
#_________________________________________________________________________________



GRGFS      = None
GRGFS_cpu  = None
GRGFS_gpu  = None
tigGFS     = None
heure      = None
gfs_interp = None




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


def gribFileName(basedir):
    ''' cherche le dernier grib complet disponible au temps en secondes '''
    ''' temps_secondes est par defaut le temps instantané '''
    ''' Cherche egalement le dernier grib chargeable partiellement'''
    ''' Change le nom du fichier a 48 '''

    temps_secondes=time.time()
    date_tuple       = time.gmtime(temps_secondes) 
    date_formatcourt = time.strftime("%Y%m%d", time.gmtime(temps_secondes))
    dateveille_tuple = time.gmtime(temps_secondes-86400) 
    dateveille_formatcourt=time.strftime("%Y%m%d", time.gmtime(temps_secondes-86400))
    mn_jour_utc =date_tuple[3]*60+date_tuple[4]
   
    if (mn_jour_utc <3*60+43):                          #avant 3h 48 UTC le nom de fichier est 18 h de la veille 
        filename=basedir+"gfs_"+dateveille_formatcourt+"-18.npy"
    elif (mn_jour_utc<9*60+43):   
        filename=basedir+"gfs_"+date_formatcourt+"-00.npy"
    elif (mn_jour_utc<15*60+43): 
        filename=basedir+"gfs_"+date_formatcourt+"-06.npy"
    elif (mn_jour_utc<21*60+43):   
        filename=basedir+"gfs_"+date_formatcourt+"-12.npy"
    else:                                              # entre 21h 48UTC  et minuit    
        filename=basedir+"gfs_"+date_formatcourt+"-18.npy" 
    date,heure,tig =dateheure(filename) 
    return filename,tig  





def chargement_grib():
    global GRGFS,GRGFS_cpu,GRGFS_gpu,tigGFS,heure,gfs_interp
    fileName,tigGFS=gribFileName(basedirGribs025)
    heure= datetime.fromtimestamp(tigGFS, tz=timezone.utc).hour
    
    with open(fileName, 'rb') as f:
            GRGFS = np.load(f)
    print('Le grib 025 {} h+ {:3.0f}h            {}      a été chargé sur l ordi local  '.format(heure,GRGFS[0,0,0,1]*3,fileName))
    print ('test sur tigGFS dans Chargement grib ',tigGFS)
    tig=tigGFS
    GRGFS_cpu = torch.from_numpy(GRGFS)
    GRGFS_gpu = GRGFS_cpu.to('cuda', non_blocking=True) 
    build_gfs_interp()

    return tig
   
      
       
def majgrib():
    print("\nRecherche d'une mise à jour du grib")
    global GRGFS, GRGFS_cpu, GRGFS_gpu, tigGFS, heure, gfs_interp

    filename, derniertig = gribFileName(basedirGribs025)
    heure = datetime.fromtimestamp(derniertig, tz=timezone.utc).hour

    print("derniertig =", derniertig, " tigGFS =", tigGFS)
    print("Dernier indice chargé :", GRGFS[0,0,0,1]*3, "h")
    print("filename:", filename)

    if os.path.exists(filename):
        # CORRECTION ICI :
        if (derniertig != tigGFS) or (GRGFS[0,0,0,1] < 120):
            print("Rechargement du grib necessaire\n******************************")
            chargement_grib()  # <-- suffit (ça refait cpu/gpu + build_gfs_interp)
            print("Indice chargé :", GRGFS[0,0,0,1]*3, "h")
            print("tigGFS apres chargement_grib:", tigGFS)
        else:
            print("Pas de rechargement necessaire.")
    else:
        print(f"Le nouveau fichier {filename} n'existe pas encore")

    return




class GFS025InterpGPU_721:
    """
    GFS 0.25°: GRGFS (n_steps, 721, 1440, 2), pas temps 3h.
    Interpolation temps + bilinéaire espace, puis vitesse (kn) + direction FROM (0..360, nord horaire).
    """

    def __init__(self, GRGFS: torch.Tensor, clamp_min_kn=1.0, clamp_max_kn=70.0, dt_step_h=3.0):
        assert GRGFS.ndim == 4 and GRGFS.shape[-1] == 2, "GRGFS doit être (n_steps, 721, 1440, 2)"
        self.GRGFS = GRGFS
        self.device = GRGFS.device
        self.dtype = GRGFS.dtype

        self.n_steps, self.ny, self.nx, _ = GRGFS.shape
        assert self.ny == 721 and self.nx == 1440, f"attendu (721,1440), reçu ({self.ny},{self.nx})"

        self.dt_step_h = float(dt_step_h)
        self.scale = 4.0  # 0.25° => 4 points/deg

        self.clamp_min = float(clamp_min_kn)
        self.clamp_max = float(clamp_max_kn)

        self._one = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self._rad2deg = 180.0 / math.pi

    @torch.no_grad()
    def __call__(self, dtig, lat0, lon0, return_uv=False):
        GRGFS = self.GRGFS

        lat = torch.as_tensor(lat0, dtype=torch.float32, device=self.device).reshape(-1)
        lon = torch.as_tensor(lon0, dtype=torch.float32, device=self.device).reshape(-1)
        dt  = torch.as_tensor(dtig, dtype=torch.float32, device=self.device).reshape(-1)

        # broadcast minimal
        if dt.numel() == 1 and lat.numel() > 1:
            dt = dt.expand_as(lat)
        if lat.numel() == 1 and dt.numel() > 1:
            lat = lat.expand_as(dt)
            lon = lon.expand_as(dt)

        # ----- indices espace -----
        # y_f in [0..720] si lat in [90..-90]
        y_f = (90.0 - lat) * self.scale
        x_f = torch.remainder(lon, 360.0) * self.scale  # périodique

        iy = torch.floor(y_f).to(torch.long)
        ix = torch.floor(x_f).to(torch.long)

        # pour accéder iy+1, on clip iy dans [0..ny-2] = [0..719]
        iy = torch.clamp(iy, 0, self.ny - 2)
        # pour ix on fait wrap périodique, mais ix doit être dans [0..nx-1]
        ix = torch.remainder(ix, self.nx)

        iy1 = iy + 1
        ix1 = torch.remainder(ix + 1, self.nx)

        dy = torch.clamp(y_f - iy.to(torch.float32), 0.0, 1.0)
        dx = torch.clamp(x_f - ix.to(torch.float32), 0.0, 1.0)

        ax = self._one - dx
        ay = self._one - dy

        # ----- temps -----
        t = (dt / 3600.0) / self.dt_step_h  # pas de 3h
        it = torch.floor(t).to(torch.long)
        wt = torch.clamp(t - it.to(torch.float32), 0.0, 1.0)
        it = torch.clamp(it, 0, self.n_steps - 2)

        # ----- 4 coins (temps interpolé), u/v séparés -----
        u00 = GRGFS[it,   iy,  ix,  0] + wt * (GRGFS[it+1, iy,  ix,  0] - GRGFS[it,   iy,  ix,  0])
        v00 = GRGFS[it,   iy,  ix,  1] + wt * (GRGFS[it+1, iy,  ix,  1] - GRGFS[it,   iy,  ix,  1])

        u01 = GRGFS[it,   iy,  ix1, 0] + wt * (GRGFS[it+1, iy,  ix1, 0] - GRGFS[it,   iy,  ix1, 0])
        v01 = GRGFS[it,   iy,  ix1, 1] + wt * (GRGFS[it+1, iy,  ix1, 1] - GRGFS[it,   iy,  ix1, 1])

        u10 = GRGFS[it,   iy1, ix,  0] + wt * (GRGFS[it+1, iy1, ix,  0] - GRGFS[it,   iy1, ix,  0])
        v10 = GRGFS[it,   iy1, ix,  1] + wt * (GRGFS[it+1, iy1, ix,  1] - GRGFS[it,   iy1, ix,  1])

        u11 = GRGFS[it,   iy1, ix1, 0] + wt * (GRGFS[it+1, iy1, ix1, 0] - GRGFS[it,   iy1, ix1, 0])
        v11 = GRGFS[it,   iy1, ix1, 1] + wt * (GRGFS[it+1, iy1, ix1, 1] - GRGFS[it,   iy1, ix1, 1])

        # bilinéaire
        u = u00 * ax * ay + u01 * dx * ay + u10 * ax * dy + u11 * dx * dy
        v = v00 * ax * ay + v01 * dx * ay + v10 * ax * dy + v11 * dx * dy

        # ----- knots + direction FROM -----
        vit_kn = torch.sqrt(u*u + v*v) * KNOTS
        vit_kn = torch.clamp(vit_kn, min=self.clamp_min, max=self.clamp_max)

        ang_math = torch.atan2(v, u) * self._rad2deg
        ang_from  = torch.remainder(270.0 - ang_math, 360.0)  # towards, nav
        #ang_tow = torch.remainder(ang_tow + 180.0, 360.0)   # from

        vit_kn = vit_kn.to(self.dtype)
        ang_from = ang_from.to(self.dtype)

        if return_uv:
            return vit_kn, ang_from, u.to(self.dtype), v.to(self.dtype)
        return vit_kn, ang_from



def build_gfs_interp():
    global gfs_interp, GRGFS_gpu

    gfs_interp = GFS025InterpGPU_721(GRGFS_gpu)
    gfs_interp.__call__ = torch.compile(gfs_interp.__call__, mode="reduce-overhead")

    # warmup
    lat512 = torch.empty(512, device="cuda", dtype=torch.float32)
    lon512 = torch.empty(512, device="cuda", dtype=torch.float32)
    dtig   = torch.tensor(3600.0, device="cuda")
    for _ in range(10):
        gfs_interp(dtig, lat512, lon512)
    torch.cuda.synchronize()



tigGFS=chargement_grib()      # Chargement initial
print ('test1 sur tigGFS',tigGFS)
majgrib()                         # met a jour egalement les fichiers torch GRGFS_gpu et GRGFS_cpu   
print ('test2 sur tigGFS',tigGFS)


#________________________________________________________________________
#  Chargement ECMMWF
#________________________________________________________________________

GRECM      = None
GRECM_cpu  = None
GRECM_gpu  = None
tigECM     = None
heure      = None
ecm_interp     = None


steps_3h_ecmwf = list(range(0, 147, 3))      # 0,3,6,...,144
steps_6h_ecmwf = list(range(150, 361, 6))    # 150,156,...,360
steps_h        = steps_3h_ecmwf + steps_6h_ecmwf    # heures de prévision

def chargement_ECM():
    global GRECM,GRECM_cpu,GRECM_gpu,tigECM,ecm_interp
    
    filename, tigECM  = ecmwfFileNamenpy(basedirGribsECMWF) 
    GRECM = np.load(filename) 
    GRECM_cpu = torch.from_numpy(GRECM)
    GRECM_gpu  = GRECM_cpu.to('cuda', non_blocking=True)  
    build_ecm_interp()
    # ecm_interp     = ECMWFInterpGPU(GRECM_gpu, steps_h)   # Une seule fois au chargement du run ECMWF (ou après maj)
    return GRECM,tigECM 

def maj_ecm():
    print('\nRecherche d\'une mise à jour du grib ECM')
    global GRECM,GRECM_cpu,GRECM_gpu,tigECM,ecm_interp
    filename, derniertigECM  = ecmwfFileNamenpy(basedirGribsECMWF)
    print('Dernier Indice chargé ',GRECM[0,0,0,1]*3,'h\n')
    if os.path.exists(filename) == True:
         if (derniertigECM !=tigECM   or int(GRECM[0,0,0,1])<85 ):
            print('Rechargement du grib ECM necessaire\n******************************')
            GRECM,tigECM = chargement_ECM()
            print('Indice chargé',GRECM[0,0,0,1]*3,'h\n')
            GRECM[0,0,0,0]=0
            GRECM_cpu = torch.from_numpy(GRECM)
            GRECM_gpu = GRECM_cpu.to('cuda', non_blocking=True)   
            GRECM[0,0,0,0]=int(tigECM)/100
            build_ecm_interp()
            # ecm_interp     = ECMWFInterpGPU(GRECM_gpu, steps_h)   # Une seule fois au chargement du run ECMWF (ou après maj)
            return 
    else:
        print('Le nouveau fichier GRECM {}  n existe pas encore'.format(filename))
    return    
 
class ECMWFInterpGPU:
    """
    Interpolation (temps + bilinéaire) sur GRECM ECMWF (n_steps, ny, nx, 2)
    + conversion directe en (vitesse_kn, dir_from_deg).
    steps_h constant: stocké une fois sur GPU.
    """

    def __init__(self, GRECM: torch.Tensor, steps_h, clamp_min_kn=1.0, clamp_max_kn=70.0):
        assert GRECM.ndim == 4 and GRECM.shape[-1] == 2, "GRECM doit être (n_steps, ny, nx, 2)"
        self.GRECM = GRECM
        self.device = GRECM.device
        self.dtype = GRECM.dtype

        self.steps = torch.as_tensor(steps_h, dtype=torch.float32, device=self.device)
        self.clamp_min = float(clamp_min_kn)
        self.clamp_max = float(clamp_max_kn)

        _, ny, nx, _ = GRECM.shape
        self.ny = ny
        self.nx = nx

        # constantes grille
        self.dlat = 180.0 / (ny - 1)
        self.dlon = 360.0 / nx
        self.lat_max = 90.0

        # constantes torch pour éviter recréation
        self._one = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self._rad2deg = 180.0 / math.pi

    @torch.no_grad()
    def __call__(self, dtig, lat0, lon0, return_uv=False):
        GRECM = self.GRECM
        steps = self.steps

        lat = torch.as_tensor(lat0, dtype=torch.float32, device=self.device).reshape(-1)
        lon = torch.as_tensor(lon0, dtype=torch.float32, device=self.device).reshape(-1)
        dt  = torch.as_tensor(dtig, dtype=torch.float32, device=self.device).reshape(-1)

        # broadcast minimal (comme ta version numpy)
        if dt.numel() == 1 and lat.numel() > 1:
            dt = dt.expand_as(lat)
        if lat.numel() == 1 and dt.numel() > 1:
            lat = lat.expand_as(dt)
            lon = lon.expand_as(dt)

        # --- TEMPS ---
        t_hours = dt / 3600.0
        idx = torch.searchsorted(steps, t_hours, right=True) - 1
        idx = torch.clamp(idx, 0, steps.numel() - 2).to(torch.long)

        step0 = steps[idx]
        step1 = steps[idx + 1]
        dt_step = step1 - step0
        dt_step = torch.where(dt_step == 0, torch.ones_like(dt_step), dt_step)
        w = torch.clamp((t_hours - step0) / dt_step, 0.0, 1.0)  # poids temporel

        # --- ESPACE ---
        lat_idx_f = (self.lat_max - lat) / self.dlat
        lon360 = torch.remainder(lon, 360.0)
        lon_idx_f = lon360 / self.dlon

        iy = torch.floor(lat_idx_f).to(torch.long)
        ix = torch.floor(lon_idx_f).to(torch.long)

        iy = torch.clamp(iy, 0, self.ny - 2)
        ix = torch.clamp(ix, 0, self.nx - 2)

        iy1 = iy + 1
        ix1 = ix + 1

        dx = lon_idx_f - ix.to(torch.float32)
        dy = lat_idx_f - iy.to(torch.float32)
        ax = self._one - dx
        ay = self._one - dy

        # --- 4 coins (temps interpolé), puis bilinéaire ---
        # coin (iy,ix)
        u00 = GRECM[idx,   iy,  ix,  0] + w * (GRECM[idx+1, iy,  ix,  0] - GRECM[idx,   iy,  ix,  0])
        v00 = GRECM[idx,   iy,  ix,  1] + w * (GRECM[idx+1, iy,  ix,  1] - GRECM[idx,   iy,  ix,  1])

        # coin (iy,ix1)
        u01 = GRECM[idx,   iy,  ix1, 0] + w * (GRECM[idx+1, iy,  ix1, 0] - GRECM[idx,   iy,  ix1, 0])
        v01 = GRECM[idx,   iy,  ix1, 1] + w * (GRECM[idx+1, iy,  ix1, 1] - GRECM[idx,   iy,  ix1, 1])

        # coin (iy1,ix)
        u10 = GRECM[idx,   iy1, ix,  0] + w * (GRECM[idx+1, iy1, ix,  0] - GRECM[idx,   iy1, ix,  0])
        v10 = GRECM[idx,   iy1, ix,  1] + w * (GRECM[idx+1, iy1, ix,  1] - GRECM[idx,   iy1, ix,  1])

        # coin (iy1,ix1)
        u11 = GRECM[idx,   iy1, ix1, 0] + w * (GRECM[idx+1, iy1, ix1, 0] - GRECM[idx,   iy1, ix1, 0])
        v11 = GRECM[idx,   iy1, ix1, 1] + w * (GRECM[idx+1, iy1, ix1, 1] - GRECM[idx,   iy1, ix1, 1])

        # bilinéaire
        u = u00 * ax * ay + u01 * dx * ay + u10 * ax * dy + u11 * dx * dy
        v = v00 * ax * ay + v01 * dx * ay + v10 * ax * dy + v11 * dx * dy

        # --- vitesse + direction FROM ---
        vit_kn = torch.sqrt(u*u + v*v) * KNOTS
        vit_kn = torch.clamp(vit_kn, min=self.clamp_min, max=self.clamp_max)


        
        # ang_math = torch.atan2(v, u) * self._rad2deg               # 0=Est, CCW
        # ang_tow  = torch.remainder(270.0 - ang_math, 360.0)        # 0=N, CW (towards)
        # ang_from = torch.remainder(ang_tow + 180.0, 360.0)         # from

        ang_math = torch.atan2(v, u) * (180.0 / math.pi)
        ang_from = torch.remainder(270.0 - ang_math, 360.0)


        
        # même dtype que GRECM si tu veux homogénéité
        vit_kn = vit_kn.to(self.dtype)
        ang_from = ang_from.to(self.dtype)

        if return_uv:
            return vit_kn, ang_from, u.to(self.dtype), v.to(self.dtype)
        return vit_kn, ang_from


def build_ecm_interp():
    global ecm_interp, GRECM_gpu,steps_h

    ecm_interp = ECMWFInterpGPU(GRECM_gpu,steps_h)
    ecm_interp.__call__ = torch.compile(ecm_interp.__call__, mode="reduce-overhead")

    # warmup
    lat512 = torch.empty(512, device="cuda", dtype=torch.float32)
    lon512 = torch.empty(512, device="cuda", dtype=torch.float32)
    dtigECM   = torch.tensor(3600.0, device="cuda")
    for _ in range(10):
        ecm_interp(dtigECM, lat512, lon512)
    torch.cuda.synchronize()


# pas initie en local pour les tests car sature la memoire 
# ECM_Actif
#if (IS_PRODUCTION):
chargement_ECM()
maj_ecm()

#--------------------------------------------------------------------------------------------------
#    Test de quelques échéances
#--------------------------------------------------------------------------------------------------




# lat = -4
# lon = -93
# heures_test =  [ 0, 12, 24, 48,96,120,144]   # T+0h, 6h, 24h, 72h
# print ('heure ECM  : {} '.format(time.strftime("%d %b %H:%M ",time.localtime(float(tigECM)))))

# for h in heures_test:
#     heure = tigECM + h * 3600.0
#     dtig = heure - tigECM
#     vit, angle = prevision_ecmwf_dtig( GRECM, dtig,lat, lon)  
   
    
#     print(' Test ECMWF à {} localtime pour lat {:6.4f} lon {:6.4f}    Vitesse {:6.3f} angle : {:6.3f}° '.format(time.strftime("%d %b %H:%M ",time.localtime(float(heure))),lat,lon,vit,angle))




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


# def get_carte(pos, zoom_x, zoom_y):
#     lat, lon = int(pos[0]), int(pos[1])
#     cle = (lat, lon)

#     with cache_lock:
#         if cle in cache_cartes:
#             return cache_cartes[cle]

#     # Si non trouvé (ou expiré), on recharge la carte
#     carte = fcarte((lat, lon), zoom_x, zoom_y)

#     with cache_lock:
#         cache_cartes[cle] = carte

#     return carte


def get_carte(lat, lon):
    # Centre sur les multiples de 5 (carte 10x10)
    lat_centre = (int(lat) // 5) * 5
    lon_centre = (int(lon) // 5) * 5

    cle = (lat_centre, lon_centre)

    with cache_lock:
        if cle in cache_cartes:
            return cache_cartes[cle]

    # Calcul ou chargement de la carte (taille fixe de 10x10)
    carte = fcarte2(lat_centre, lon_centre)

    with cache_lock:
        cache_cartes[cle] = carte

    return carte



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



def updateUsers(users, username, user_id):
    users[user_id] = username
    #print ('************************',users,'/n****************************************')


# @socketio.on('connect')
# def handle_connect():        
#     client_id = request.args.get('client_id')  # ID unique du client (passé en paramètre)
#     print ('client_id',client_id)
#     if client_id:
#         clients[client_id] = request.sid  # Associer l'ID client à la session WebSocket
#         join_room(client_id)  # Joindre une "room" unique pour ce client
#     print(f"Client {client_id} connecté avec session {request.sid}")




# @socketio.on('disconnect')
# def handle_disconnect():
#     client_id = None
#     for key, value in clients.items():


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
            return False  # Enregistrement ignoré, trop ancien
        
        #ATTENTION IL FAUT AUSSI LES RECALCULER
        #""""""""""""""""""""""""""""""""""""""
        
    finally:
        cursor.close()
        pg_pool.putconn(conn)




def insert_historoutages(  username,course, user_id, status, heuredepart, lat, lon, eta):
    ''' Sert pour garder l historique des routages realises '''
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
     
    # try:                          #recherche dans la table 
    print ('Recherche dans la table')
    print('course',course)
    leginfostr  = rechercheTableLegInfos(course)
    leginfos    = json.loads(leginfostr)
    # except:                      #recherche sur vrouteur 
    #     leginfostr  = rechercheleginfos(course)
    #     leginfos    = json.loads(leginfostr)
    
    
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


    # Recuperation des zones d exclusion VR
    #**************************************
    try:
        zones=leginfos['restrictedZones']
        #print ('exclusions1 l 1179 ************\n',zones)
        # print()
        tabexclusions={}
        # for zone in zones:
        #     name = zone["name"]
        #     print (name)
        #     vertices = [[pt["lat"], pt["lon"]] for pt in zone["vertices"]]
        #     print (vertices)
        #     tabexclusions[name] = vertices    
        for i, zone in enumerate(zones, start=1):
            base = zone.get("name", "zone")
            name = f"{base}_{i}"
            vertices = [[pt["lat"], lon_to_360( pt["lon"])] for pt in zone["vertices"]]
            tabexclusions[name] = vertices
        # try:
        #     for zone in zones:
        #         try :
        #             name = zone["name"]
        #         except:
        #             name=time.time()
        #         vertices = [[pt["lat"], pt["lon"]] for pt in zone["vertices"]]
        #         tabexclusions[name] = vertices

        # except KeyError as e:
        #     print(f"Clé manquante dans la zone {zone}: {e}")    

        # print ('tabexclusions l 1203 ************\n',tabexclusions)        
  
    except:   
         print ('on est dans le except de leginfos  (Pas d\'exclusions\n**********************************************************')
         tabexclusions={}       
    # print(result)

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

    twa=55
    tws=12.1
    typeVoiles = ['jib', 'Spi', 'Staysail', 'LightJib', 'Code0', 'HeavyGnk', 'LightGnk']
    voile=typeVoiles[int(polairesglobales10[8,int(tws*10),int(twa*10)])]


    print ('\n*************************************************************************************************')   
    print ('Resultats calculés pour {:25} twa= {} tws= {} voile {} vitessemax = {:6.3f} '.format(label,twa,tws,voile,polairesglobales10[7,int(tws*10),int(twa*10)]))
    print ('\n*************************************************************************************************')
    print ('Resultats attendus pour Ocean 50            twa= 55 tws= 12.1 voile LightJib vitessemax = 11.956 ')
    print ('Resultats attendus pour Imoca               twa= 55 tws= 12.1 voile LightJib vitessemax = 13.450 ')
    print ('Resultats attendus pour Figaro3             twa= 55 tws= 12.1 voile LightJib vitessemax =  6.7101 ')
    print ('Resultats attendus pour Ultim BP XI         twa= 55 tws= 12.1 voile Jib      vitessemax = 16.202 ')
    print ('Resultats attendus pour Class40             twa= 55 tws= 12.1 voile LightJib vitessemax =  8.656 ')
    print ('Resultats attendus pour CruiserRacer        twa= 55 tws= 12.1 voile Jib      vitessemax =  8.197 ')
    print ('Resultats attendus pour Volvo65             twa= 55 tws= 12.1 voile Jib      vitessemax = 10.402')
    print ('Resultats attendus pour Mini6.5             twa= 55 tws= 12.1 voile Jib      vitessemax =  6.1')
    print ('Resultats attendus pour Super Maxi 100      twa= 55 tws= 12.1 voile LightJib vitessemax = 15.457')
    print ('Resultats attendus pour OffshoreRacer       twa= 55 tws= 12.1 voile LightJib vitessemax =  8.730')
    print ('**************************************************************************************************\n')


    # polairesglobales10to = torch.from_numpy(polairesglobales10).to('cuda')
    # tabvmg10to           = torch.from_numpy(tabvmg10).to('cuda')      
    return leginfos,tabexclusions,tabicelimits,carabateau,polairesglobales10,tabvmg10 





# recherche a partir du cache 
class DonneesCourse:
    def __init__(self, leginfos, tabexclusions, tabicelimits, carabateau,
                 polaires_np, vmg_cpu, polaires_gpu, vmg_gpu):
        self.leginfos = leginfos
        self.tabexclusions = tabexclusions
        self.tabicelimits  = tabicelimits
        self.carabateau    = carabateau
        self.polaires_np   = polaires_np
        self.vmg_cpu       = vmg_cpu
        self.polaires_gpu  = polaires_gpu
        self.vmg_gpu       = vmg_gpu


cache_donnees = CpuGpuCache(maxsize=10, ttl=3*3600)                          # definition du cache 





def rechercheDonneesCourseCache(course):
    cached = cache_donnees.get(course)

    if cached:
        print(f"[CACHE HIT] Course {course}")
        cpu_data, gpu_data = cached

        return DonneesCourse(
            *cpu_data,  # leginfos, tabexclusions, tabicelimits, carabateau, polaires_cpu, vmg_cpu
            polaires_gpu=gpu_data[4],
            vmg_gpu=gpu_data[5]
        )

    else:
        print(f"[CACHE MISS] Course {course}")
        leginfos, tabexclusions, tabicelimits, carabateau, polaires_np, vmg_np = charger_donnees(course)
        # polaires_cpu = torch.tensor(polaires_np, dtype=torch.float32, device='cpu')
        vmg_cpu      = torch.tensor(vmg_np     , dtype=torch.float32, device='cpu')
       
      
        polaires_cpu = torch.tensor(polaires_np, dtype=torch.float32, device='cpu')
        polaires_gpu   = polaires_cpu.clone().cuda()
        vmg_gpu        = vmg_cpu.clone().cuda()

        cache_donnees.set(course,
                          cpu_data=(leginfos, tabexclusions, tabicelimits, carabateau, polaires_np, vmg_cpu),
                          gpu_data=(leginfos, tabexclusions, tabicelimits, carabateau, polaires_gpu, vmg_gpu))

        return DonneesCourse(
            leginfos, tabexclusions, tabicelimits, carabateau,polaires_np,
            vmg_cpu, polaires_gpu, vmg_gpu
        )












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
    
    # print ('state',state)
    
    if state=='waiting':
        # si le state est waiting on a seulement un heading et legstartdate     
        lastCalcDate         = boatinfosbs['legStartDate']   # normalement deja recupere en dessus 
        t0=legStartDate/1000
        twsto,twdto = prevision025to(GRGFS_cpu,t0,y0vr,x0vr)
        capto=torch.tensor([headingvr])
        twato=ftwato(capto,twdto)
        twsvr=twsto.item()
        twdvr= twdto.item()
        twavr   =twato.item()
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

def calculePosDepart(posStartVR,polairesglobales,carabateau,dt=60): 

    ''' donne la position de depart au bout de 60 s'''
    ''' Sous la meme forme que posStartVR'''
    
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
        sail       = posStartVR['voile']
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
        peno_chgt = Chgt * spline(lw, hw, lwtimer, hwtimer, tws) * MF * Cstamina
#       peno_gybe = Tgybe * spline(lw, hw, lwtimerGybe, hwtimerGybe, tws) * Cstamina
        peno_globale= soldepeno + peno_chgt
        peno_fin    = max(0,peno_globale-dt)
    
#       # Stamina
        perte_stamina = calc_perte_stamina_np(tws, int(Tgybe), Chgt, coeffboat)
        recup = frecupstamina(dt, tws)
        stamina = stamina - perte_stamina + recup
        staminafin   =  max (0.0, min(100.0, stamina))
    
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
        twsf,twdf= prevision025(GRGFS, t0+dt, y1, x1)

        # print ('dans calcul isodepart twsf {},twdf {} t0+dt {} '.format(twsf,twdf ,time.strftime(" %d %b %H:%M %S ",time.localtime(t0+dt)) ))  
        dtig=   t0+dt-tigGFS
        # dtig=torch.tensor(dtig, device='cuda') 
        twsf1,twdf1=  prevision025dtig(GRGFS, dtig, y1, x1)
        # print ('dans calcul isodepart twsf1  {},twdf1 {} t0+dt {} '.format(twsf1,twdf1 ,time.strftime(" %d %b %H:%M %S ",time.localtime(t0+dt)) ))  
    
       
        posStart['tws']=twsf
        posStart['twd']=twdf
        posStart['twa']=twa
        posStart['y0'] =y1
        posStart['x0'] =x1
        posStart['t0'] =t1
        posStart['heading']=cap
        posStart['speed']  = meilleurevitesse
        posStart['voile']  = meilleurevoile +voileAuto
        posStart['stamina'] = staminafin
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
    x0=lon_to_360(x0)
    # isovr 22 elements  en 12  c'est l ecart par rapport a l iso precedent 
    isodepart       = torch.tensor ([[numisoini,npt,nptmere,y0,x0,voile,twa,stamina,soldepeno,tws,twd,cap,0,0,0,0,speed,0,boost,0,0,0]], dtype=torch.float32, device='cuda')
   
    return isodepart
    



def deplacement(dep,polairesglobales10to,carabateau):
    '''positionvr=torch.tensor([0,t0,dt,option,valeur,y0,x0,voile,twa,cap,speed,stamina,soldepenovr,twd,tws,voileAuto,boost],dtype=torch.float64)'''

    lwtimer             = carabateau['lwtimer']
    hwtimer             = carabateau['hwtimer']
    lw                  = carabateau['lws']
    hw                  = carabateau['hws']
    lwtimerGybe         = carabateau['gybeprolwtimer']
    hwtimerGybe         = carabateau['gybeprohwtimer']
    coeffboat           = carabateau['coeffboat']
    MF                  = 0.8


    
    dt=dep[2]
    ari = dep.clone()
    option=dep[3]
    ari[0]=dep[0]+1
    ari[1]=dep[1]+dep[2]
    ari[2]=dep[2]            # l intervalle de temps est reconduit pour le prochain
    dep[7]=dep[7]%10
    # on calcule la tws et twd au point initial
    ari[14],ari[13]= prevision025to(GRGFS_gpu,dep[1],dep[5],dep[6])     # calcul de tws et twd
    tws10=torch.round(ari[14]*10).int().item()

    if option==1:
       # print('option =1 twa')
       twa=dep[4]
       ari[8]=twa 
       cap= fcap(twa,ari[13])
       ari[9]=cap 
    else:
       # print('option =0 cap')
       twa= ftwato(dep[4],ari[13])
       ari[6]=twa  
       ari[9]= dep[9]
       cap=dep[4]

    # print('twa ',twa)
    # print('cap ',cap)
 
    twa10=  torch.round(torch.abs(twa)*10).int().item()
    vitessevoileini   = polairesglobales10to[dep[7].int().item(), tws10, twa10]                                    # vitesse voileini[voileini,tws10,twa10
    Chgt=0                                                                                                         # par defaut
    if dep[15]==1:                                                                                                 # si voile auto 
        meilleurevitesse  = polairesglobales10to[  7         , tws10, twa10]                                       # vitesse meilleure voile[voileini,tws10,twa10
        meilleurevoile    = polairesglobales10to[  8         , tws10, twa10]                                       # meilleure voile
        ari[16]           = meilleurevitesse/(vitessevoileini+0.0001)                                              # boost         
        ari[7]            = torch.where(ari[16]>1.014,meilleurevoile,dep[7])                                                 # la voile = nouvelle voile si boost>1.014    sinon ancienne voile 
        ari[10]           = meilleurevitesse                                                                                  # la vitesse est celle de la meilleure voile  
        if ari[16]>1.014:
            Chgt=1
            
    else :
        ari[7] = dep[7]                                                                                           # si voile manuelle la voile reste inchangee
        ari[10]=vitessevoileini                                                                                   # la vitesse est celle de l ancienne voile  chgt reste = 0 car on ne change pas de voile 
    Tgybe=((twa*dep[8])<0)*1
  
    # on calcule la penalites de temps et de stamina 
    Cstamina      = 2 - 0.015 * dep[11]                                                                                        # coefficient de stamina en fonction de la staminaini
    tempspenochgt = Chgt *splineto(lw, hw, lwtimer, hwtimer,dep[14]) * MF * Cstamina                                        # tempspenochgt=chgt*spline(-,-,-,-,tws)*MF*Cstamina le tws est le tws du depart  
    tempspenoTG   = Tgybe*splineto(lw, hw, lwtimerGybe, hwtimerGybe,dep[14])  * Cstamina                                    # pas de magicfurler pour le tackgybe
    ari[12]       = dep[12]+tempspenochgt+tempspenoTG                                                                         #*** cumul peno  on rajoute les penalites sur l iteration le -dt sera applique a la fin avec un clamp
    stamina       = dep[11] - calc_perte_stamina_to(dep[14],Tgybe.item(),Chgt, coeffboat)  +   frecupstaminato(dt,dep[14])         # la stamina est egale a l ancienne (col4 )-perte (tws,TG,Chgt,coeff,MF)  + frecupstaminato(dt,Tws,pouf=0.8):
    ari[11]       = torch.clamp(stamina,min=0,max=100)                                                                          #*** le max pourra eventuellement etra passse a 110 avec une boisson 
    
     # calcul des nouvelles coordonnees
    dt_ite  = dt-0.3*torch.clamp(ari[12],max=dt )                                                                           # dt remplace boost en colonne 17
    capradians=torch.deg2rad(cap)
    y0radians=torch.deg2rad(dep[5])

    ari[5]             = dep[5] + ari[10]*dt_ite/3600/60*math.cos(capradians)                                   # latitude après deplacement
    ari[6]             = dep[6] + ari[10]*dt_ite/3600/60*math.sin(capradians)/math.cos(y0radians)               # longitude après deplacement
    ari[12]            = torch.clamp (ari[12]-dt,min=0)                                                         # la penalite a dimue de dt avec un mini de 0 
    
    return ari



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

@app.route('/majgrib', methods=["GET", "POST"])
# recherche les coursesactives pour le username
def majgribajax():
    
    majgrib()
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
        if not valeur:
            return jsonify({"error": "Missing valeur for modification"}), 400
       
        
        if typeinfo == "tolerancehvmg":
            try:
                valeur = float(valeur)  # ou int(valeur) si tu veux un entier
            except ValueError:
                return jsonify({"error": "valeur invalide pour tolerancehvmg"}), 400
            infos["tolerancehvmg"] = valeur

        else:
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
    global GRGFS,tigGFS,indicemajgrib,heure
  
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


    majgrib()                                        # on renvoie aussi la date de majgrib
    #tigGFS= GRGFS[0,0,0,0].astype(np.float64)*100       # permet d afficher la date du grib
    indicemajgrib=int(GRGFS[0,0,0,1])                # permet d afficher son indice
    # print ('Dans rechercheboatinfos ligne 3346 indice de mise a jour du grib  ',indicemajgrib) 
    # tigGFS vient directement de majgrib et chargegrib
   
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



    
    
    
    
    
    print ('*****************************************************************')
    
    return None 






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



class DeplacementEngine:
    def __init__(self, donneesCourse):
        c                  = donneesCourse.carabateau
        self.lwtimer       = c['lwtimer']
        self.hwtimer       = c['hwtimer']
        self.lw            = c['lws']
        self.hw            = c['hws']
        self.lwtimerGybe   = c['gybeprolwtimer']
        self.hwtimerGybe   = c['gybeprohwtimer']
        self.coeffboat     = c['coeffboat']
        self.polaires_np   = donneesCourse.polaires_np
      

    def deplacement_np(self, dep):
         # position_np=([0)0, 1)t0,2)dt,3)option,4)valeur,5)y0,6)x0,7)voile,8)twa,9)cap,10)speed,11)stamina,12)soldepeno,13)twd,14)tws,15)voileAuto,16)boost],dtype=torch.float64)'''
        ari=np.zeros(len(dep))
        
        dt          = dep[2]
        option      = int(dep[3])
        valeur      = dep[4]
        y0          = dep[5]
        x0          = dep[6]
        if dep[7]==9 :                       # arrive si waiting la voile est 10 ce qui amene la voile jp a 9 
           dep[7]=0
        voile_ini   = int(dep[7]%10)
        heading     = dep[8]
        twd         = dep[9]
        vitesse     = dep[10]
        stamina_ini = dep[11]
        peno_ini    = dep[12]
        auto_voile  = 1                       #  int(dep[15])
        MF=0.8

        ari[0] = dep[0] + 1
        ari[1]=dep[1]+dep[2]
        ari[2] = dt
        
        
        tws, twd = prevision025(GRGFS, dep[1], y0, x0)
        
        if option == 1:
            twa = valeur
            cap = fcap(twa, twd)
            # print ('Option twa : twa et twd ',twa ,' ', twd)

        else:          
            cap = valeur
            twa = ftwato(cap, twd)
            # print ('Option cap : cap et twd ',cap ,' ', twd)


            
        # valeurs en debut d iteration pour calculer le deplacement 
        ari[3]=dep[3]           # on conserve l option 
        ari[4]=valeur 
        ari[8]=twa
        ari[9]=cap
        
        ari[14]=tws
        ari[13]=twd
        # print ('test ',dep[1], y0, x0)
        tws10 = round(tws * 10)
        twa10 = round(abs(twa * 10))
        # print ('twa' ,twa)
        # print ('twa10' ,twa10)


        vitesse_voile_ini = self.polaires_np[voile_ini, tws10, twa10]
        chgt_voile = 0

        if auto_voile == 1:
            meilleure_vitesse = self.polaires_np[7, tws10, twa10]
            meilleure_voile = int(self.polaires_np[8, tws10, twa10])
            boost = meilleure_vitesse / (vitesse_voile_ini + 1e-4)
            ari[16] = boost
            if boost > 1.014:
                ari[7] = meilleure_voile
                vitesse_voile_ini = meilleure_vitesse
                chgt_voile = 1
            else:
                ari[7] = voile_ini
        else:
            ari[7] = voile_ini

        ari[10] = vitesse_voile_ini

        Tgybe = 1 if (twa * heading < 0) else 0
        Cstamina = 2 - 0.015 * stamina_ini

        peno_chgt = chgt_voile * spline(self.lw, self.hw, self.lwtimer, self.hwtimer, tws) * MF * Cstamina
        peno_gybe = Tgybe * spline(self.lw, self.hw, self.lwtimerGybe, self.hwtimerGybe, tws) * Cstamina
        ari[12] = peno_ini + peno_chgt + peno_gybe

        perte_stamina = calc_perte_stamina_np(tws, Tgybe, chgt_voile, self.coeffboat)
        recup = frecupstamina(dt, tws)
        stamina = stamina_ini - perte_stamina + recup
        ari[11] = np.clip(stamina, 0.0, 100.0)

        dt_ite = dt - 0.3 * min(ari[12], dt)
        cap_rad = np.deg2rad(cap)
        y0_rad = np.deg2rad(y0)
        ari[5] = dep[5]   + vitesse_voile_ini * dt_ite / 3600.0 / 60.0 * math.cos(cap_rad)
        ari[6] = dep[6]   + vitesse_voile_ini * dt_ite / 3600.0 / 60.0 * math.sin(cap_rad) / math.cos(y0_rad)
       

       
        ari[12] = max(0.0, ari[12] - dt)        # c est le solde de penalite a la fin de l iteration 

        # on regarde si on traverse une terre 

        # A = (y0,x0)
        # B = (ari[5], ari[6])

        # print(intersect_segment_polyline(A, B, carte))  # ➜ True
        # if segment_intersecte_carte_np(y0, x0, ari[5], ari[6], self.carte):
        #     ari[15] = 1.0  # marque que le segment traverse une barrière
        # else:
        #     ari[15] = 0.0
        # if intersecte_segment_multipolyline_np(dep[5],dep[6], ari[5], ari[6], self.carte):
        #     ari[15] = 1  # Marque une pénalité ou un rejet (ex. : 1 = traverse une barrière)
        # else:
        #     ari[15] = 0

        return ari




class DeplacementEngine2:
    def __init__(self, course):
        donneesCourse1     = rechercheDonneesCourseCache(course)   # c 'est un objet de la classe DonneesCourse
        c                  = donneesCourse1.carabateau
        self.lwtimer       = c['lwtimer']
        self.hwtimer       = c['hwtimer']
        self.lw            = c['lws']
        self.hw            = c['hws']
        self.lwtimerGybe   = c['gybeprolwtimer']
        self.hwtimerGybe   = c['gybeprohwtimer']
        self.coeffboat     = c['coeffboat']
        self.polaires_np   = donneesCourse1.polaires_np
        self.polaires_gpu  = donneesCourse1.polaires_gpu
        self.tabvmg        = donneesCourse1.vmg_cpu
        self.MF            = 0.8  
   


    # def posplus(self, Position,dt,dt_it,option,valeur,dtig0GFS):

    #     ''' dans cette hypothese de deplacement on ne connait que la position 
    #         l intervalle de temps
    #         l option 
    #         le cap ou la twa suivant l option 
    #         la stamina initiale 
    #         le solde de penalite 

    #     '''
    #     Positionfin=np.copy(Position)
        
    #     numero      = Position[0]
    #     y0          = Position[2]
    #     x0          = Position[3]
    #     twa         = Position[5]   # c est la twa pour le deplacement que je veux faire apparaitre dans le tableau 
    #     cap         = Position[6]
    #     vitesse     = Position[9]
    #     voile       = int(Position[10])
    #     tws0         = Position[12]
    #     twd         = Position[13]
    #     stamina_ini = Position[14]
    #     soldepeno   = max((Position[15]),0)
    #     dt_ite      = dt_it    - 0.3 * min(soldepeno, dt)        #normalement dt 

    #     Positionfin[0]=Position[0]+1
    #     Positionfin[1]=dt
        
    #     if option==0 : 
    #         cap = valeur
    #         twa1= ftwaos(cap,twd) 
    #     if option==1 : 
    #         twa1= valeur
    #         cap= fcap(twa,twd) 

        
    #     # print (' ite {} option {} cap {:4.2f} '.format(i,option ,cap))
    
            
    #     cap_rad = np.deg2rad(cap)
       
    #     y0_rad  = np.deg2rad(y0) 
    #     y1      = y0   + vitesse * dt_ite / 3600.0 / 60.0 * math.cos(cap_rad)
    #     x1      = x0   + vitesse * dt_ite / 3600.0 / 60.0 * math.sin(cap_rad) / math.cos(y0_rad)


    #     Positionfin[2] = y1
    #     Positionfin[3] = x1

        
    #     tws,twd =  prevision025dtig(GRGFS, dtig0GFS+dt , y1, x1)                        # Previsions au point de depart  
    #     Positionfin[12]=tws
    #     Positionfin[13]=twd
    #     Positionfin[6]=fcap(twa1,twd) 
        
    #   # on va faire le calcul de voile et de vitesse 
    #     tws10 = round(tws*10)
    #     twa10 = abs(round(twa1*10))
    #     vitesseVoileIni       = polairesglobales10[voile, tws10, twa10]                                           # vitesse voileini[voileini,tws10,twa10
    #     meilleureVitesse      = polairesglobales10[7    ,  tws10, twa10]                                          # vitesse meilleure voile[voileini,tws10,twa10
    #     meilleureVoile        = polairesglobales10[8,  tws10, twa10]                                              # meilleure voile
    #     Boost                 = meilleureVitesse/(vitesseVoileIni+0.0001)                                    # Boost 


    #     #  # # calcul des penalites
    #     if Boost >1.014 :
    #         Chgt=1
    #         voilefinale=meilleureVoile
    #     else:
    #         Chgt=0
    #         voilefinale=voile
    #                                                                                                               # on remplit la colonne chgt a la place de voiledef
    #     Tgybe  = ((twa*twa1)<0)*1                                                                                 # on remplit la colonne 16 Tgybe a la place de boost  (signe de twam1*twa10

    #     Positionfin[5]=twa1                                                        # on ne peut pas ecrasere la twa tant que la twa anterieure n a pas ete utilisee 
       
        
                                                                                                                 
    #     Cstamina        = 2 - 0.015 * stamina_ini                                                                       # coefficient de stamina en fonction de la staminaini
    #     peno_chgt       = Chgt * spline(self.lw, self.hw, self.lwtimer, self.hwtimer, tws) * self.MF * Cstamina
    #     peno_gybe       = Tgybe * spline(self.lw, self.hw, self.lwtimerGybe, self.hwtimerGybe, tws) * Cstamina
        
    #     perte_stamina   = calc_perte_stamina_np(tws, Tgybe, Chgt, self.coeffboat)
    #     recup           = frecupstamina(dt_it, tws)
    #     # print ('dans recup stamina dt {} tws {} '.format(dt, tws))
    #     stamina         = min((stamina_ini - perte_stamina + recup),100)
    #     soldepeno       = max((soldepeno-dt),0)         # la trajectoire a ete calculee en debut d ite avec la peno complete maintenant on retire dt puis on rajoute les nouvelles penos 
    #     soldepeno1      = max((soldepeno+peno_chgt+peno_gybe ),0)   
        
    #     Positionfin[9]=meilleureVitesse
    #     Positionfin[10]=voilefinale
    #     Positionfin[14] =stamina
    #     Positionfin[15] = soldepeno1
    #     Positionfin[16] = dt_ite        
    #     return   Positionfin    

   






@app.route('/rechercheprogsvr', methods=["GET", "POST"])
def rechercheprogsvr():
    # print()
    

    # print ('On est dans rechercheprogsvr suite à la demande de rafraichissement generee par boataction ')
    user_id    = request.args.get('user_id')                                 # recupere les donnees correspondant a valeur dans le site htmlstrf
    course     = request.args.get('course')                                  # recupere les donnees correspondant a valeur dans le site
    t0routage      = int(float(request.args.get('t0routage')))
    isMe='yes'
    ari=['WP1']   # ne sert a rien dans le cas present mais est necessaire pour que ari soit defini

    session         = RoutageSession(course, user_id,isMe,ari)        # permet de recuperer les polaires et autres 

    positionvr      = session.positionvr
    boatinfos       = session.boatinfos
    positionvr_np   = positionvr.detach().cpu().numpy()
    lastCalcDateBI= int(boatinfos['bs']['lastCalcDate']/1000)
    # print ('boatinfos dans rechercheprogsvr\n ',boatinfos)

    date,origine,programmations = extraitprogrammations (boatinfos,user_id,course)      # va chercher dans la table 

    #  print  ('lastCalcDateBA',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBA)),lastCalcDateBA)
    # print  ('lastCalcDateBI',time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBI)),lastCalcDateBI)
    

    print('\n****************************************************************************************************************************************************************')
    print ('Programmations retenues heure Boatinfos {} BAction  :{}  ,origine : {} ,programmations {}\n'.format(time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDateBI)),time.strftime(" %d %b %H:%M ",time.localtime(date)),origine,programmations))
    print('******************************************************************************************************************************************************************')
    Position_np =np.zeros((24*60,17),dtype=np.float64)   
    Position_np[0]=positionvr_np
    donneesCourse    = rechercheDonneesCourseCache(course)
    engine = DeplacementEngine(donneesCourse)
    sections = []

    if "progs" in programmations:
        typeprogs='progs'
        print('On a affaire a des programmations')
        tic=time.time()
        prog0=programmations['progs']
        prog1 = copy.deepcopy(prog0)
        applied_prog = [False] * len(prog1)
        
        Position_np[0, 3] = prog1[0][1]
        Position_np[0, 4] = prog1[0][2]

        applied_prog = [False] * len(prog1)

        tic =time.time()
       
        for i in range(len(Position_np) - 1):
            t = Position_np[i, 1]  # temps courant
            
            for j in range(len(prog1)):
                if not applied_prog[j] and t >= prog1[j][0]:
                    Position_np[i, 3] = prog1[j][1]  # option
                    Position_np[i, 4] = prog1[j][2]  # valeur
                    applied_prog[j] = True          # marquer comme appliquée

           
            Position_np[i + 1] = engine.deplacement_np(Position_np[i])
     # position_np=([0)0, 1)t0,2)dt,3)option,4)valeur,5)y0,6)x0,7)voile,8)twa,9)cap,10)speed,11)stamina,12)soldepeno,13)twd,14)tws,15)voileAuto,16)boost],dtype=torch.float64)'''   
    
    
        # segments_par_voile=segmenter_par_voile_routage_np(Position_np)
    # print (Position_np)


        
            
        routeprogs = Position_np[:,[1,3,4,5,6,7,8,9,10,13,14]]      # ensemble des
        routeprogs =[arr.tolist() for arr in routeprogs]

        # on transforme en tableau pour la transmission 
        # impressionpointdep(Position_np)

        

    if "wps" in programmations:
        tic=time.time()
        print ('programmations',programmations)
        progswaypoints=programmations['wps']
        typeprogs='wps'
        print('On a affaire a des wps')
        
        y0       = Position_np[0, 5]
        x0       = Position_np[0, 6]
        tini     = Position_np[0,1]         # issu de la position de depart
        

        WP=[[round(y0,5),round(x0,5)]]     #Wp du point de depart
        for elt in progswaypoints:
            WP.append([elt['lat'],elt['lon']])

        print ('tableau des wp avec le point de depart',WP )
        capini   = calcul_cap_loxodromique    (WP[0][0], WP[0][1], WP[1][0], WP[1][1])
        distance = distance_haversine         (WP[0][0], WP[0][1], WP[1][0], WP[1][1])
        
        optionini = 0   # pour les waypoints on suit un cap constant 
        valeurini = capini

        sections = []
        i = 0
        idx          = 0  # Index dans Position_np
        tempsrestant = 1000

       
        print ('lenWP',len(WP))
        
        tic=time.time()

        
        print()
    
        tic=time.time()
        for j in range(len(WP) - 1):
            section = { "y0": WP[0][0], "x0": WP[0][1],  "t_start": tini,"twa": Position_np[idx, 8], "cap": Position_np[idx, 9],"stamina": Position_np[idx, 11],  "voile": Position_np[idx, 7],\
            "soldepeno": Position_np[idx, 12],"twd": Position_np[idx, 13], "tws": Position_np[idx, 14]      }
        
            # on initialise le trajet *
            lat0=WP[j][0]
            lon0=WP[j][1]
            lat1=WP[j+1][0]
            lon1=WP[j+1][1]
            cap =calcul_cap_loxodromique(lat0, lon0, lat1, lon1)  
            
            distanceRestante = distance_haversine(WP[j][0], WP[j][1], WP[j+1][0], WP[j+1][1])        # C est la distance au depart
            print ('WP depart {}  WP visé {}  ,capini {},distance {}'.format(WP[j],WP[j+1],cap,distanceRestante))
            Position_np[idx, 3] =0     # option cap pur 
            Position_np[idx, 4] = cap  # heading vers le nouveau wp
            Position_np[idx, 2] = 60   # on reinitialise le dt 
            tempsrestant=1000

            while idx < Position_np.shape[0] - 1:
                # cap = calcul_cap_loxodromique_tensor(Positionto[idx][5], Positionto[idx][6], WP[j+1][0], WP[j+1][1])
                # Positionto[idx][4] = cap
            
                distWP = distance_haversine(Position_np[idx][5], Position_np[idx][6], WP[j+1][0], WP[j+1][1])
                speed = Position_np[idx][10]
            
                tempsrestant = (distWP / (speed + 1e-6)) * 3600  # évite la division par zéro
            
                if tempsrestant < 60:  # On approche du waypoint
                    print(f'tempsrestant {tempsrestant:.1f}s, distance au WP {distWP:.3f} NM, vitesse {speed:.2f} kn')                    
                    Position_np[idx, 2] = tempsrestant  # ajuste dt AVANT le dernier déplacement
                    Position_np[idx + 1] = engine.deplacement_np(Position_np[idx])
                    Position_np[idx + 1][2] =60
                    print ('Position du WP suivant ', lat1,' ', lon1)
                    Position_np[idx + 1][5] =lat1
                    Position_np[idx + 1][6] =lon1
                    # on va forcer la position 
                    idx += 1
                    break  # on termine ce tronçon 5 et  6 , si il y a changement de voile la derniere position n 'est pas bonne 
                else:
                    Position_np[idx, 2] = 60  # sinon, dt standard
                    Position_np[idx + 1] = engine.deplacement_np(Position_np[idx])
                    idx += 1
            section.update({
                "y1": Position_np[idx, 5],
                "x1": Position_np[idx, 6],
                "t_end": Position_np[idx, 1],
                "stam_end": Position_np[idx, 11],
                "voile_end": Position_np[idx, 7],
                "peno_end": Position_np[idx, 12],
                "twd_end": Position_np[idx, 13],
                "tws_end": Position_np[idx, 14],
                "twa_end": Position_np[idx, 8],
                "duree": Position_np[idx, 1] - section["t_start"]
            })
            
            sections.append(section)








        Position_np=Position_np[Position_np[:, 0] != 0] 
        routeprogs = Position_np[:,[1,3,4,5,6,7,8,9,10,13,14]] 
        routeprogs =[arr.tolist() for arr in routeprogs]
        print('temps de calcul des programmations',time.time()-tic)

        impressionpointdep(Position_np)



    polyline=[]
    timestamp= int(time.time() )
   
  

    response=make_response(jsonify({'polyline':polyline,'origine':origine,'date':date, 'routeprogs':routeprogs,  'timestamp':timestamp,'typeprogs' :typeprogs, 'programmations':programmations,'sections':sections}))
    response.headers.add('Access-Control-Allow-Origin', '*')                # Autorise toutes les origines
    return response








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
    # print('y0vr {}, x0vr {} ,t0vr {} ,twsvr {} polar_id {}'.format(y0vr,x0vr,t0vr,tws,polar_id))

    twscalc,twdcalc=prevision025(GRGFS,t0vr,y0vr,x0vr)
    filenamelocal2='vmg10_'+str(polar_id)+'.npy'
    filename2=basedirnpy+filenamelocal2
    tabvmg10=get_fichier(filename2)
    # with open(filename2,'rb')as f:
    #      tabvmg10 = np.load(f)

    #on calcule les vmg pour tws
    tws10        = int(round(tws*10))
    tabvmg       = tabvmg10[tws10]
    tabvmg       = tabvmg.tolist()

    tabrecouvrements = plagevoiles2(polar_id,tws)
    tabrecouvrements = [arr.tolist() for arr in tabrecouvrements]

    # tableau numpy transforme en array
    #print('tabrecou',tabrecouvrements)

    response=make_response(jsonify({"twscalc":twscalc, "twdcalc":twdcalc,"tabvmg":tabvmg,"tabrecouvrements":tabrecouvrements }))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response



@app.route('/recherchevoilessimple', methods=["GET", "POST"])
def recherchevoilessimple():
    # recupere la meteo sur le point les vmg et les recouvrements  
   
    tws      = float(request.args.get('tws'))
    polar_id = int(request.args.get('polar_id'))
    # print('y0vr {}, x0vr {} ,t0vr {} ,twsvr {} polar_id {}'.format(y0vr,x0vr,t0vr,tws,polar_id))

   
    filenamelocal2='vmg10_'+str(polar_id)+'.npy'
    filename2=basedirnpy+filenamelocal2
    tabvmg10=get_fichier(filename2)
    # with open(filename2,'rb')as f:
    #      tabvmg10 = np.load(f)

    #on calcule les vmg pour tws
    tws10        = int(round(tws*10))
    tabvmg       = tabvmg10[tws10]
    tabvmg       = tabvmg.tolist()

    tabrecouvrements = plagevoiles2(polar_id,tws)
    tabrecouvrements = [arr.tolist() for arr in tabrecouvrements]

    # tableau numpy transforme en array
    #print('tabrecou',tabrecouvrements)

    response=make_response(jsonify({"tabvmg":tabvmg,"tabrecouvrements":tabrecouvrements }))
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







# @app.route("/index", methods=["GET", "POST"])
# def index():
#     '''SErt uniquement a donner les indications de base a index '''

#     client     = request.remote_addr
#     host       = request.host
#     origine    = request.url_root
#     print()
#     print('Client {} host {} origin {}'.format(client,host,origine))

#     try : 
#         #si l origine de la demande est le dash il y a une source de definie et une posStart 
#                                                                                                    # pour l instant ne sert a rien
#         source                  = request.args.get('source')      
#         posStart                = request.args.get('posStart')  
#         posStart                = json.loads(posStart)
#         print('sail : ',posStart['sail'])
    
#     except:
#         #print ('Pas de reception de valeurs par le dash  on indique que la source est "direct" \n')
#         source='direct'
#         posStart={"username": 'Takron-BSP','source':source}
#         print('\nDemande accès direct au site par : ',origine)
#         #print ('\nLigne 1808  Demande de routage direct sans transmission par l URL\n source : {} \n posStart :{}   \n '.format(source,posStart))

#     return render_template("index.html",origine=origine,source=source,posStart=posStart)







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

        reponse ='Données bien reçues par le serveur '  # reponse envoyee au dash pour test
        print ("\nMessage reçu : {}".format(typeinfos) )
       # print ('Type du Message : {} \n*********************************************************'.format(typeinfos))
        print(' message ' , )
        print ('Message         : {} \n***********************************'.format( message))


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
                    state       = message['bs']['state']
                    tini0       = message['bs']['lastCalcDate']/1000


                    updateUsers(users, username, user_id)
                    print('users',users)
                    # print ('\n boatinfos pour  {}  le  {} \n********************************************************\n {} \n'.format(username,time.strftime(" %d %b %Y %H:%M ",time.localtime(tini0)), message))
                    # print()    
                except:
                    print (' toutes les infos ne sont pas disponibles')
                    state       =  message['bs']['state']
                    #print    ('\nstate :',state) 

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
                print()
                






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
                    timestamp =time.time()


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
                    envoi = {'Reception':typeinfos ,'course':course}
                    print ('tentative d envoi au client   ',user_id ,'  envoi ' ,envoi )
                    send_update_to_client( user_id, envoi)         # envoi par websocket de l info que l on a recu une actualisation  de boatActions 
                        
                except:
                    print ('On est dans le except de boatactions')
                




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
                updated,polar_id,polaires=rechercheTablePolaires(_id)
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
          

             





        if typeinfos =='getfleet':
            
                # print(message)
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
########### Fonctions de routage                       ########################
###############################################################################

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






def calcul_caps(tabpoints):
    """
    tabpoints : ndarray de forme (N, 2)  -> [ [lat1, lon1], [lat2, lon2], ... ]
    Retourne un tableau (N,) des caps (azimuts) en degrés [0..360), 
    la dernière valeur étant 0.
    """
    lat = np.radians(tabpoints[:, 0])
    lon = np.radians(tabpoints[:, 1])

    # Différences successives
    dlon = lon[1:] - lon[:-1]

    # Formule de cap initial (azimut) sur sphère
    x = np.sin(dlon) * np.cos(lat[1:])
    y = np.cos(lat[:-1]) * np.sin(lat[1:]) - np.sin(lat[:-1]) * np.cos(lat[1:]) * np.cos(dlon)

    caps = np.degrees(np.arctan2(x, y)) % 360

    # Même dimension : on ajoute une valeur finale à 0
    caps = np.append(caps, 0.0)

    return caps


def lissage(course,routage_np,t0,posStartVR,posStart):
    tabpoints       = routage_np[:,[2,3]]
    tabtwa          = routage_np[:,5]
    tabdt           = routage_np[:,1]
    tabpointslisses = lissagepoints_dt_twa(tabpoints, tabtwa, tabdt)
    tabpointslisses = lissagepoints_dt_twa(tabpointslisses, tabtwa, tabdt)   # on fait 2 lissages successifs
    caps            = calcul_caps(tabpointslisses)
    
    
    #initialisation de engine pour avoir les differentes valeurs  
    deplacement=DeplacementEngine2(course)
    polairesglobales10=deplacement.polaires_np
    tabvmg10           = deplacement.tabvmg
    
    routagelisse            = np.zeros_like(routage_np)
    routagelisse[:,[0,1,4]] = routage_np[:,[0,1,4]]
    routagelisse[:,[2,3]]   = tabpointslisses
    routagelisse[:,6]       = caps
    routagelisse[:,8]       = tabtwa
    dtig0GFS                   = t0-tigGFS

    Tws,Twd                 = prevision025dtig(GRGFS, dtig0GFS+routagelisse[:,1] , routagelisse[:,2],routagelisse[:,3])     # calcul de tws et twd 
    routagelisse[:,12]      = Tws 
    routagelisse[:,13]      = Twd
    Twa                     = np.round(ftwao(routagelisse[:,6],routagelisse[:,13]))                # twa=ftwao(cap,twd
    routagelisse[:,5]       = Twa
    
    # on va recalculer les differents autres elements 0
    lenroutage        = len (routagelisse)
    tws10             = np.rint(Tws*10).astype(int)
    twa10             = np.abs(np.rint(Twa*10)).astype(int)
                                                            # on calcule vmgmin et max a partir de tws et twd 
    routagelisse[:,7] = tabvmg10[tws10,2]
    routagelisse[:,8] = tabvmg10[tws10,4]
    
    meilleureVitesse  = polairesglobales10[7  ,  tws10, twa10]                                          # vitesse meilleure voile[voileini,tws10,twa10]
    routagelisse[:,9] = meilleureVitesse   
    meilleureVoile    = polairesglobales10[8,  tws10, twa10] 
    
    
    
    for i in range (0,lenroutage):
        twsi = Tws[i]
        if i==0:
            voileini    = posStartVR['voile']%10
            stamina_ini = posStart['stamina']
            twaini      = posStart['twa'] 
            dt_ini    = 60
            solde_peno  =  posStart['penovr']
            dt_it       = 0

        else:    
            voileini    =int(routagelisse[i-1,10]) 
            stamina_ini  = routagelisse[i-1,14]
            twaini      = routagelisse[i-1,5]
            dt_ini      = routagelisse[i-1,1]
            dt_it       =  routagelisse[i,1] -dt_ini 
            
        Tgybe  = ((routagelisse[i,5]*twaini)<0)*1      
        vitessevoileini=polairesglobales10[voileini,  tws10[i], twa10[i]]
        boost  =  routagelisse[i,9]/(vitessevoileini +0.0001)
       
        routagelisse[i,11]=boost
        if boost >1.014:
            routagelisse[i,10]=meilleureVoile[i]
            Chgt=1                                   #Changement de voile 
          
        else:
            routagelisse[i,10]=voileini
            Chgt=0
            
        # on va calculer les penalites et les changements de stamina
        Cstamina        = 2 - 0.015 * stamina_ini                                                                       # coefficient de stamina en fonction de la staminaini
        peno_chgt       = Chgt * spline(deplacement.lw, deplacement.hw, deplacement.lwtimer, deplacement.hwtimer, twsi) * deplacement.MF * Cstamina
        peno_gybe       = Tgybe * spline(deplacement.lw, deplacement.hw, deplacement.lwtimerGybe, deplacement.hwtimerGybe, twsi) * Cstamina
        perte_stamina   = calc_perte_stamina_np(twsi, Tgybe, Chgt, deplacement.coeffboat)
        recup           = frecupstamina(dt_it, twsi)
        stamina         = min((stamina_ini - perte_stamina + recup),100)
       
        solde_peno       = max((solde_peno-dt_it),0)         # la trajectoire a ete calculee en debut d ite avec la peno complete maintenant on retire dt puis on rajoute les nouvelles penos 
        solde_peno1      = max((solde_peno+peno_chgt+peno_gybe ),0) 
        
         
        if i==0:
            solde_peno1= posStart['penovr']
            stamina    = posStart['stamina']
            
       

        routagelisse[i,14]= stamina
        routagelisse[i,15]=solde_peno1
    
    return routagelisse









def reconstruire_chemin_rapide(isoglobal: torch.Tensor, nptmini: int) -> torch.Tensor:
    """
    Reconstruit le chemin depuis le point d’ID nptmini jusqu’au point d’ID 0 (inclus).
    """
    # Création des dictionnaires pour remonter efficacement
    point_ids = isoglobal[:, 1].to(torch.int32).tolist()
    parent_ids = isoglobal[:, 2].to(torch.int32).tolist()
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
    return isoglobal[chemin_indices]







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





class RoutageSession:                                                  # version au 8 aout


    def __init__(self, course, user_id,isMe,ari):
        tic=time.time()
        
        # Initialisation globale
        ########################
        self.n3=512
        self.isoglobal=torch.zeros((1000*self.n3,15), device='cuda', dtype=torch.float32)    # isoglobal est destiné a recevoir 850 isochrones avec leurs 512 points  
        
        # print ('course',course)
        # print ('user_id',user_id)        
        
        self.course  = course
        self.user_id = user_id
        self.ari     = ari
  
        
        # Chargement des données
        DonneesCourse1                                                      = rechercheDonneesCourseCache(course)   # c 'est un objet de la classe DonneesCourse
        self.boatinfos,self.posStartVR,self.exclusionsVR2  ,self.positionvr = rechercheDonneesCourseUser( user_id,course)                  # self.exclusionsVR2 est un dictionnaire avec les exclusions VR qui ne sert pas 
        self.personalinfos,self.ari,self.waypoints,self.exclusionsperso,self.barrieres,self.trajets,self.tolerancehvmg,retardpeno = rechercheDonneesPersoCourseUser( user_id,course)
     
        # print('self.barrieres ',self.barrieres)

        self.leginfos              = DonneesCourse1.leginfos
        self.exclusionsVR          = DonneesCourse1.tabexclusions                     # ce sont les exclusions VR
        self.tabicelimits          = DonneesCourse1.tabicelimits
        self.carabateau            = DonneesCourse1.carabateau
        self.polairesglobales10to  = DonneesCourse1.polaires_gpu
        self.tabvmg10to            = DonneesCourse1.vmg_gpu
        self.polaires_np           = DonneesCourse1.polaires_np                               #ce sont les polaires10
        self.tabvmg                = DonneesCourse1.vmg_cpu


        
        
        # pour l instant posStartVR  est systematiquement ma position 

        self.posStart              = calculePosDepart(self.posStartVR,self.polaires_np,self.carabateau,dt=60)    # Position de depart du routage au bout de 60 s


        # si ce n est pas moi , va donner une mauvaise indication 
        self.isodepart             = calculeisodepart2(self.posStart)    # Transformation en iso de depart 
        
        try :
        
            self.retardpeno            = retardpeno['retardpeno']
        except: 
            self.retardpeno=0


        # print()
        # print ('Ligne 3567 retardpeno',retardpeno )
        # print()
        
        # print()
        # print ('selfexclusionsVR',self.exclusionsVR) 

        # pos=(self.posStartVR['y0'],self.posStartVR['x0'])
        # self.carte=fcarte(pos)      # renvoie une multipolyline      
        self.exclusionsVR.update(self.exclusionsperso)          # on rajoute exclusionsperso à exclusions VR 

        # print ('selfexclusionsVR apres integration des exclusions perso ',self.exclusionsVR)        
        # # conversion en tenseur 

        # print()
        # # print ('selfexclusionsVR2 ',self.exclusionsVR2) 
        # print()
        # print ('selfexclusionsperso ',self.exclusionsperso)
       
        try:
            self.exclusions = {  nom: torch.tensor([[lon, lat] for lat, lon in coords], device='cuda')   for nom, coords in self.exclusionsVR.items()  }    # transformation en tensor de self.exclusionsVR
        except : 
            self.exclusions = {}

            

        self.segments = prepare_segments(self.exclusions)  # [S, 4] segments [x1, y1, x2, y2]        
        # except:
        #     self.segments=[]
        # self.isodepart                    = calculeIsoDepart(self.posStartVR,self.polairesglobales10to,self.carabateau) # tient compte du state waiting si necessaire
        self.t0vr                           = self.posStartVR['t0']
        self.t0                             = self.posStart['t0']


        # print ('\ntemps initial T0 au point VR dans posStartVR', time.strftime(" %d %b %H:%M %S",time.localtime(self.t0vr)))
        # print   ('temps initial T0 au point 1  dans posStart  ', time.strftime(" %d %b %H:%M %S",time.localtime(self.t0)))       
        # print()
        # print ('temps chargement des donnees pour la classe Routage Session ',time.time()-tic) 
        

    
    def initialiseRoutagePartiel(self, posStartPartiel,ari, indiceroutage , ouverture=220, pas=1, cocheexclusions=1):
        '''Initialisation des données nécessaires au routage pour chaque waypoint'''
        
        isoglobal   = self.isoglobal 
        waypoints   = self.waypoints
        retardpeno  = self.retardpeno
        waypoint = next((ligne for ligne in waypoints if ligne[1] == ari[indiceroutage]), None)    # on extrait le waypoint correspondant à ari[indiceroutage]
            
        
        numisoini   = posStartPartiel['numisoini']
        npt         = posStartPartiel['npt']
        nptmere     = posStartPartiel['nptmere']
        nptini      = posStartPartiel['nptini']
        y0          = posStartPartiel['y0']
        x0          = posStartPartiel['x0']
        voile       = posStartPartiel['voile']
        voileAuto   = posStartPartiel['voileAuto']
        tws         = posStartPartiel['tws']
        twd         = posStartPartiel['twd']
        twa         = posStartPartiel['twa']
        twaAuto     = posStartPartiel['twaAuto']
        cap         = posStartPartiel['heading']
        speed       = posStartPartiel['speed']
        stamina     = posStartPartiel['stamina']
        soldepeno   = posStartPartiel['penovr']
        boost       = posStartPartiel['boost']


        x0=lon_to_360(x0)

        # print ( '\n Dans initialiseroutagepartiel y0 = ', posStartPartiel['y0'])
        # print ( '\n Dans initialiseroutagepartiel x0 = ', posStartPartiel['x0'])
        # print ('*************************************************************')
        
        try :
            t0       = posStartPartiel['t0']
            # print()
            # print ('Heure de depart du routage partiel t0 =',time.strftime(" %d %b %H:%M %S",time.localtime(t0)))
            # print()
            
        except:    
            ecart       = posStartPartiel['ecart']
            t0          = self.t0 + ecart
    
        y1          = waypoint[2]
        x1          = waypoint[3]
        rwp         = waypoint[4]
            
        
        depto       = torch.tensor([y0, x0], device='cuda', dtype=torch.float32)
        arito       = torch.tensor([y1, x1], device='cuda', dtype=torch.float32)  
    
        cap_ar      = round((450 - math.atan2(y1 - y0, x1 - x0) * 180 / math.pi) % 360)
        range_caps  = torch.arange(cap_ar - ouverture / 2, cap_ar + ouverture / 2, pas) % 360
        range_capsR = range_caps.deg2rad()
    
        m_ar        = (y1 - y0) / (x1 - x0)
        centre      = [(y0 + y1) / 2, (x0 + x1) / 2]
        rayon       = (((y1 - y0)**2 + (x1 - x0)**2)**.5) / 2 * 1.10
    
        deptoR      = torch.deg2rad(depto)
        aritoR      = torch.deg2rad(arito)
        dti         = 60 
        # print('dist',dist)
        distDepAri  = dist(deptoR[0], deptoR[1], aritoR[0], aritoR[1])
        rayonRoutage   = (distDepAri / 2)
        centreRoutageR = (deptoR + aritoR) / 2
       

        if indiceroutage==0:
            iso          = self.isodepart 
            seuils = [[144, 300],[108,600], [672, 1800], [240, 3600]]       # 14h a 5 mn = 120 --  18h a 10 mn =108 -- 14 jours a 30mn =24*2*14=672
            tabdt = construire_dt(seuils, taille=1000)
            # print('shape isoglobal ',isoglobal.shape) 
            # print('shape iso ',iso.shape) 
            isoglobal[0] = iso[0,0:15]                  # il est necessaire de remplir le premier terme de isoglobal
           
    
        else:
            seuils = [[30, 60], [20, 600], [10, 1800]]
            tabdt = construire_dt(seuils, taille=1000)
            
            iso=torch.tensor ([[numisoini,npt,nptmere,y0,x0,voile,twa,stamina,soldepeno,tws,twd,cap, ecart,0,0,0,speed,0,boost,0,0,0]], dtype=torch.float32, device='cuda')  
            # print( 'isoglobal du numero de point mini', self.isoglobal[nptini])                  #  isoglobal est deja rempli
       

        paramRoutage={
            "posStartPartiel": posStartPartiel,
            "indiceroutage" :indiceroutage,
            "numisoini":numisoini, 
            "deptoR": deptoR,
            "aritoR": aritoR,
            "y0": y0,
            "x0": lon_to_360(x0),
            "y1": y1,
            "x1": x1,
            "rwp": rwp,
            "dti": dti,
            "cap_ar": cap_ar,
            "range_caps": range_caps,
            "range_capsR": range_capsR,
            "m_ar": m_ar,
            "centreRoutageR": centreRoutageR,
            "rayonRoutage": rayonRoutage,
            "distDepAri": distDepAri,
            "cocheexclusions": cocheexclusions,
            "dtglobal":tabdt,                                  # c est le tableau des dt 
            "retardpeno":retardpeno
          
        }
      
        #print ('dans initialiseroutagepartiel paramroutage',paramRoutage)


        return paramRoutage,iso
    

    def isoplusun(self,iso,tmini,paramRoutage,mode ):
        ''' Donnees a charger propres au routage'''
        ''' polairesglobales10to, lw, hw, lwtimer, hwtimer,MF, coeffboat, rayonRoutage     '''
        ''' necessite comme donnneees externes  GRGFS_gpu ,polairesglobales10to,range_caps,range_capsR,carabateau,lw,....                                   '''
        ''' Les penalites sont affichees sur l iteration et ne sont diminuees du temps de l iteration que sur l iteration suivante '''

        MF          = 0.8
        furler      = 0.8
        # tolerancehvmg = self.tolerancehvmg         #tolerance de hvmg
        numisom1=int(iso[0,0].item())                                                 # numero de l'iso precedent 
        numiso=numisom1+1    
        t0          = self.t0

        dtig0GFS       = t0-tigGFS
        if mode!='gfs':
            dtig0ECM       = t0-tigECM
        # print('t0 ',time.strftime(" %d %b %H:%M %S",time.localtime(t0)))
        # print('tigGFS ',time.strftime(" %d %b %H:%M %S",time.localtime(tigGFS)))
        # print ('dtig0GFS en h ',dtig0GFS/3600)
        
        ecartprecedent=iso[0,12].item()         # c est l ecart de temps de a la fin de l iso precedent par rapport a t0 , c est le temps ou sont les points de depart de calcul de l iso 
        #print ('ecartprecedent /t0',ecartprecedent)
       
        

        # print('iso.shape',iso.shape)
        n3=512
        range_caps     = paramRoutage["range_caps"]
        range_capsR    = paramRoutage["range_capsR"]
        centreRoutageR = paramRoutage['centreRoutageR']
        rayonRoutage   = paramRoutage['rayonRoutage']
        aritoR         = paramRoutage ["aritoR"]
        deptoR         = paramRoutage ["deptoR"]
        m_ar           = paramRoutage ["m_ar"]
        numisoini      = paramRoutage ["numisoini"]
        indiceroutage  = paramRoutage ["indiceroutage"]
        dtglobal       = paramRoutage ["dtglobal"]
        retardpeno     = paramRoutage ["retardpeno"]


        #print ('retardpeno et tolerance hvmg', retardpeno,self.tolerancehvmg)
        
        latcR, loncR   = centreRoutageR
        
        lw             = self.carabateau["lws"]
        hw             = self.carabateau["hws"]
        lwtimer        = self.carabateau["lwtimer"]
        hwtimer        = self.carabateau["hwtimer"]
        gybeprolwtimer = self.carabateau['gybeprolwtimer']
        gybeprohwtimer = self.carabateau['gybeprohwtimer']
        coeffboat      = self.carabateau['coeffboat']
        polairesglobales10to = self.polairesglobales10to
        tabvmg10to           = self.tabvmg10to
        
        ecartprecedent=iso[0,12].item()                                               # c est l ecart de temps de l iso precedent par rapport a t0
     
        n=len (iso)                                                                   # Longueur de l iso precedent sert a dupliquer 
        p=len(range_caps)                                                             # nombre de caps du rangecaps 
        dernier=iso[-1,1].item()                                                      # dernier point de l iso precedent   
        
       
        # if numiso==1: 
        #     print()
        #     print ('dans iso+1 tiso pour prev meteo du  deuxieme isochrone ', time.strftime(" %d %b %H:%M %S",time.localtime(tiso.item())))
        #     print()
      
        ordre=numiso-numisoini -1                                                       # ordre par rapport a numisoini pour calculer le dt 
      
        if tmini>3600:                                                                # cas normal a plus de 1h  de l objectif  
            dt=dtglobal[ordre]
       
        else :
            dt = torch.tensor(60.0, dtype=torch.float32, device='cuda:0')
         
        iso[:,0]  = numiso                                                              # on ajoute 1 au numero d iso sur toute la colonne  
        ecart=ecartprecedent+dt     # c est l ecart de temps de l iso  par rapport a t0
  
        # print('iso {} tmini {} '.format(numiso,tmini ))
      
    
        
        # if (numiso!=1):       # Pour le premier isochrone la penalite appliquable au trajet a deja ete amputee du dt  de 60 secondes dans  isodepart  
        
        iso[:,8]=torch.clamp(iso[:,8]-dt,min=0)                                     # pour le calcul de la penalite appliquable au trajet on defalque le dt qui permet de passer 
                                                                                        # de l iso precedent a celui que l on est en train de calculer et on bloque a 0 min 
        
        iso[:,11]=torch.round(iso[:,9]*10)
       
        iso            = iso.repeat((p, 1))                                                                # répète p fois les lignes (nb de caps)
        tws10          = iso[:,11] .to(torch.int)                                                          # nombre de points de l iso precedent pour pouvoir dupliquer les caps 
        caps_expanded  = range_caps.repeat_interleave(n)                                                   # on duplique les caps (cap1,cap1,cap1,cap2,cap2,cap2,cap3,cap3.......)
        capsR_expanded = range_capsR.repeat_interleave(n)
        iso[:,12]      = caps_expanded                                                                     # caps dupliques en col12
        iso[:,13]      = capsR_expanded                                                                    # caps en radians dupliqués en col13
        iso[:,2]       = iso[:,1]                                                                          # on enregistre les numeros de points comme des numeros de point mere 
       

        # calcul suivant des twa entieres au lieu de caps entiers
        iso[:,14] = torch.round(ftwato(iso[:,12],iso[:,10]) )                                                           # twa    arrondies  
        iso[:,12] = fcapto (iso[:,14],iso[:,10])
        iso[:,13]  =iso[:,12].deg2rad()

   ###############     
        #print ('numiso ordre : ',ordre,'   ', ecart)




        # # calcul de la voile
        iso[:,14] = ftwato(iso[:,12],iso[:,10])                                                            # twa                                                     
        # iso[:,18] = torch.round(abs(  iso[:,14])* 10  ) 
        twa10=torch.round(torch.abs(iso[:,14]) * 10).to(torch.int)
        # la je peux directement reduire le horsvmg 

        # print('iso.shape avant reduction vmg',iso.shape)
                                                               # on calcule vmgmin et max a partir de tws et twd 
        vmg_min = tabvmg10to[tws10,2]-self.tolerancehvmg
        vmg_max = tabvmg10to[tws10,4]+self.tolerancehvmg

        vmg = torch.abs(iso[:, 14])
        mask = (vmg >= vmg_min) & (vmg <= vmg_max)

        iso = iso[mask]
        
        tws10          = iso[:,11] .to(torch.int)   #on est oblige de redefinir la taaille de tws10 et twa10   # si on avait pris iso ce serait pas necessaire
        twa10=torch.round(torch.abs(iso[:,14]) * 10).to(torch.int)
        # print('iso.shape apres reduction vmg',iso.shape)


        
        #twa10 ftwato (cap,twd)*10)     (non signee)
        iso[:,15] = polairesglobales10to[iso[:,5].int(), tws10, twa10.int()]                               # vitesse voileini[voileini,tws10,twa10
        iso[:,16] = polairesglobales10to[7,  tws10, twa10]                                                 # vitesse meilleure voile[voileini,tws10,twa10
        iso[:,17] = polairesglobales10to[8,  tws10, twa10]                                                 # meilleure voile
        iso[:,18] = iso[:,16]/(iso[:,15]+0.0001)                                                                               # Boost remplace twa10
        iso[:,19] = torch.where(iso[:,18]>1.014,iso[:,17],iso[:,5])                                                            # voile definitive 
        iso[:,5]  = iso[:,19]                                                                                                  #*** on met la nouvelle voile dans la colonne 5 a la place de l ancienne
       
        
        # calcul des penalites
        iso[:,19] = torch.where(iso[:,18]>1.014,1,0)                                                                            # on remplit la colonne chgt a la place de voiledef
        iso[:,18] = (iso[:,6]*iso[:,14])<0                                                                                      # on remplit la colonne 16 Tgybe a la place de boost  (signe de twam1*twa10
        
        iso[:,6]  = iso[:,14]                                                                                                  # on met la nouvelle twa a la place de l ancienne  
        Cstamina  = 2 - 0.015 * iso[:,7]                                                                                        # coefficient de stamina en fonction de la staminaini
        
    
        # calcul seulement pour les éléments où nécessaire
        # on va passer avec la nouvelle version 

# Nouveau calculmodele 
# Calcul des penalites suite aux manoeuvres 
# mask_voile = tab[:,12] != 0
# mask_tack  = tab[:,11] != 0
# PenaliteChgt = np.zeros(N, dtype=np.float32)
# PenaliteTackGybe = np.zeros(N, dtype=np.float32)
# # formule globalisee
# Stamina = tab[:,7]
# Tws     = tab[:,8]
   
# PenaliteChgt  [mask_voile]    = peno_np(lwtimer,hwtimer,Tws[mask_voile] ,Stamina[mask_voile] )*furler
# PenaliteTackGybe[mask_tack]   = peno_np(gybeprolwtimer,gybeprohwtimer,Tws[mask_tack],Stamina[mask_tack])
# tab[:,13]=PenaliteChgt+PenaliteTackGybe



        mask_voile = iso[:,19] != 0
        mask_tack  = iso[:,18] != 0
        PenaliteChgt     = torch.zeros_like(iso[:,19])   
        PenaliteTackGybe = torch.zeros_like(iso[:,19])   
        # formule globalisee
        Stamina = iso[:,7]
        Tws     = iso[:,9]    # le vent est en colonne 9 de iso 
        
           
        PenaliteChgt  [mask_voile]    = peno_torch(lwtimer,hwtimer,iso[mask_voile,9] ,iso[mask_voile,7] )*furler
        PenaliteTackGybe[mask_tack]   = peno_torch(gybeprolwtimer,gybeprohwtimer,iso[mask_tack,9],iso[mask_tack,7])
        iso[:,8]=iso[:,8]+PenaliteChgt+PenaliteTackGybe      # on ajoute a la penalite restant eventuellement 





# Ancien modèle création de masques
        # mask_chgt = iso[:,19] != 0
        # mask_gybe = iso[:,18] != 0
    
        # # initialisation des résultats à 0
        # tempspenochgt = torch.zeros_like(iso[:,19])                               
        # tempspenoTG   = torch.zeros_like(iso[:,18])

        # if mask_chgt.any():       
        #     tempspenochgt[mask_chgt] = ( splineto(lw, hw, lwtimer, hwtimer, iso[mask_chgt,9]) * MF * Cstamina[mask_chgt]   )
        
        # if mask_gybe.any():
        #     tempspenoTG[mask_gybe]   = ( splineto(lw, hw, lwtimerGybe, hwtimerGybe, iso[mask_gybe,9]) * Cstamina[mask_gybe]  )
       
        # iso[:,8]  = iso[:,8]+tempspenochgt+tempspenoTG                                                                           #***  on rajoute les penalites sur l iteration le -dt sera applique a la fin avec un clamp

 # iso[:,7]  = iso[:,7] - calc_perte_stamina_to(iso[:,9], iso[:,18],iso[:,19], coeffboat)  +   frecupstaminato(dt,iso[:,9]) 
 # stamina                                   Tws        gybe     chgt                                          Tws


        iso[:,7]  = iso[:,7] - calc_perte_stamina_to(iso[:,9], iso[:,18],iso[:,19], coeffboat)  +   frecupstaminato(dt,iso[:,9])    # la stamina est egale a l ancienne (col4 )-perte (tws,TG,Chgt,coeff,MF)  + frecupstaminato(dt,Tws,pouf=0.8):
        iso[:,7]  = torch.clamp(iso[:,7],min=0,max=100)                                                                          #*** le max pourra eventuellement etra passse a 110 avec une boisson 
        
        # # # calcul des nouvelles coordonnees

        # print(type(dt))
        iso[:,17]=dt-0.3*torch.clamp(iso[:,8],max=dt )                                                                           # dt remplace boost en colonne 17    
        
        iso[:,18]=iso[:,3]                                                                                      # on copie la latitude initiale en 18 pour les barrieres 
        iso[:,19]=iso[:,4]                                                                                      # on copie la longitude initiale en 19 pour calculer les barrieres 
        # nouvelles cvoordonnees 
        iso[:,3]= iso[:,18] + iso[:,17] * iso[:,16] / 3600 / 60 * torch.cos(iso[:,13])                                         #  
        iso[:,4]= iso[:,19] + iso[:,17] * iso[:,16] / 3600 / 60 * torch.sin(iso[:,13])/torch.cos(iso[:,3].deg2rad())                      #    (le cos utilise pour la lat est avec la latitude deja calculee ?)
        iso[:,11]= iso[:,3].deg2rad()                                # on stocke la lat en rad pour le calcul de distance a l arrivee
        iso[:,12]= iso[:,4].deg2rad() 
        
        # on elimine les points hors cercle                          la distance par rapport au centre est calculee en colonne 15 anciennement colonne twa           latcR, loncR                                                     
        iso[:,17]=dist(iso[:,11],iso[:,12],latcR, loncR) 
        maskDiCentreRoutage   = iso[:,17]<(rayonRoutage*1.15)
        iso    = iso[maskDiCentreRoutage] 
    
        # calcul de distar, du point le plus proche et du temps estime vers l arrivee     
        iso[:,9]=dist(iso[:,11],iso[:,12],aritoR[0], aritoR[1]) *1.852       # Calcul de distar en mN en colonne 9 on utilise les valeurs en radians deja calculees pour les points et pour l arrivee 


        distmini, idx_min = torch.min(iso[:, 9], dim=0)                # idx_min est l’indice de la ligne où la distance est minimale   
        vitesse = iso[idx_min, 16]                                     # Vitesse en nœuds (milles nautiques par heure) au point de distance mini
        tmini   = ((distmini * 3600) / (vitesse * 1.852)).item()       # la distance est en km avec ma fonction dist et vitesse en noeuds pour les polaires 
        nptmini = int(iso[idx_min, 1].item())                          # transformation de idxmin en indice entier   
    
        # on va eliminer les points qui sont a plus de 3 fois la distance mini de l arrivee 
        maskDiMini   = iso[:,9]<(distmini*3)
        iso          = iso[maskDiMini]
     
        # calcul de ordoar et arrondi pour avoir le nb de points voulu a la fin 
        iso[:,10]= iso[:,11] - m_ar * iso[:,12]                                  # calcul ordoar :  ordonnee a  l origine  
        ordomini, ordomaxi = torch.min(iso[:,10]), torch.max(iso[:,10])          # min et max sur la colonne 10
                                                                   
        coeff         = (n3-1)/ (ordomaxi-ordomini)                              # coefficient pour ecremer et garder n3 points
        iso[:,10]  = (iso[:,10]*coeff).int() 
    
        # tris et elimination 
        indices_9 = torch.argsort(iso[:, 9], stable=False)                       # indice des elements tries    
        iso = iso[indices_9]                                                     # on reconstitue iso avec le tableau des elements triés sur l indice 9
        indices_10 = torch.argsort(iso[:, 10], stable=True)                      # indices des elements triés de la colonne 10
        iso = iso[indices_10]                                                    # on reconstitue iso avec le tableau des elements triés sur l indice 10
        torch.cuda.synchronize()  # on attend que le GPU ait fini pour mesurer le temps 
        iso[1:,14]= iso[1:,10] - iso[:-1,10]                                     # on fait l ecart entre 2 points successifs sur la colonne 10 ordoar dans la colonne 11
        mask   = iso[:,14]!=0
        iso    = iso[mask]                                                       # on elimine les points pour lesquels l ecart est egal a zero   
        #  on recalcule le boost sur 512 points pour l avoir a la fin 
        iso[:,14]= iso[:,16]/(iso[:,15]+0.0001)




        
        # print (' shape avant masques ',iso.shape)
        # elimination des points terre 
        iso[:,12]=terremer(iso[:,3],iso[:,4])                                   #on stocke le resultat terre mer en 12   
        mask   = iso[:,12]!=0
        iso    = iso[mask]                                                      # on garde les points pour lesquels iso[:,12]==1 
        try:
            # elimination des points dans les zones d exclusion 
            maskexclu = points_in_any_polygon_vectorized(iso[:,3],iso[:,4], self.segments)
            iso    = iso[~maskexclu] 
        except:
            None
        
        try:
            # calcul des barrieres et elimination des trajets traversant les barrieres
            mask=detect_barrier_crossings(iso[:,18],iso[:,19],iso[:,3],iso[:,4] , self.barrieres)
            iso=iso[~mask]
        except:
            None
        # recalcul du vent sur les points pour transmission vent ini a chemin
        # a voir suivant dash    tisovent= t_c[numisom1-1]
        # on calcule le vent pour la prochaine iteration 
   ### Calcul meteo   
        dtigGFS = dtig0GFS + ecart  
        
       
        #iso[:,9],iso[:,10]   = prevision025todtig(GRGFS_gpu,dtigGFS, iso[:,3],iso[:,4])    # correction erreur le vent est calcule pour le prochain point avec les nuvelles coordonnees
        
        if mode=='mixte':
            dtigECM = dtig0ECM + ecart
            if ecart<9*3600 :
                iso[:,9],iso[:,10]   = gfs_interp(dtigGFS,iso[:,3],iso[:,4] )    
            else :
                iso[:,9],iso[:,10]   = ecm_interp(dtigECM,iso[:,3],iso[:,4] ) 
                
        else:  
            iso[:,9],iso[:,10]   = gfs_interp(dtigGFS,iso[:,3],iso[:,4] ) 
            #print ('shape tws ',iso[:,9].shape)

                
        #print ('iso 9',iso[:9,])
        
        iso[:,11]=torch.rad2deg(iso[:,13])            # on remet le cap initial en 11 a partir du cap en radian que l on a toujours   
        iso[:,14] = iso[:,16]/(iso[:,15]+0.0001)      # on va remettre le boost en colonne 14 a la place de la twa   
        iso[:,13] = iso[:,16]   # copie de la vitesse max en 13 pour la passer a isoglobal
        iso[:,12] = ecart   
        
        # renumerotation 
        iso[:,1]= dernier+torch.arange(len(iso)) +1                         # on va renumeroter les points 
      
        # Copie des points de l isochrone dans isoglobal
        # print ('premier  ' ,iso[0,1])
       
        torch.cuda.synchronize()           # Pour attendre que la synchronisation soit complete et essayer d eviter des erreurs 
        premier= int(iso[0,1].item())
        dernier= int(iso[-1,1].item())
        # print ('numiso {} dt {} premier {} dernier {} shape {} isoglobal.shape {} '.format(numiso ,dt,premier,dernier,iso.shape,self.isoglobal.shape))
       
        
        self.isoglobal[premier:dernier+1, :] = iso[:, 0:15]
        
        return iso, tmini,distmini,nptmini
    

    
    # Version avec les retards peno 
    # def isoplusun(self,iso,tmini,paramRoutage ):
    #     ''' Donnees a charger propres au routage'''
    #     ''' polairesglobales10to, lw, hw, lwtimer, hwtimer,MF, coeffboat, rayonRoutage     '''
    #     ''' necessite comme donnneees externes  GRGFS_gpu ,polairesglobales10to,range_caps,range_capsR,carabateau,lw,....                                   '''
    #     ''' Les penalites sont affichees sur l iteration et ne sont diminuees du temps de l iteration que sur l iteration suivante '''

    #     MF          = 0.8
    #     # tolerancehvmg = self.tolerancehvmg         #tolerance de hvmg
    #     numisom1=int(iso[0,0].item())                                                 # numero de l'iso precedent 
    #     numiso=numisom1+1    
    #     t0          = self.t0
    #     dtig0GFS       = t0-tigGFS        
    #     ecartprecedent=iso[0,12].item()         # c est l ecart de temps de a la fin de l iso precedent par rapport a t0 , c est le temps ou sont les points de depart de calcul de l iso 
  
    #     n3=512
    #     range_caps     = paramRoutage["range_caps"]
    #     range_capsR    = paramRoutage["range_capsR"]
    #     centreRoutageR = paramRoutage['centreRoutageR']
    #     rayonRoutage   = paramRoutage['rayonRoutage']
    #     aritoR         = paramRoutage ["aritoR"]
    #     deptoR         = paramRoutage ["deptoR"]
    #     m_ar           = paramRoutage ["m_ar"]
    #     numisoini      = paramRoutage ["numisoini"]
    #     indiceroutage  = paramRoutage ["indiceroutage"]
    #     dtglobal       = paramRoutage ["dtglobal"]         # dt global , c ece sont les intervalles 
    #     retardpeno     = paramRoutage ['retardpeno']
        
    #     latcR, loncR   = centreRoutageR
        
    #     lw             = self.carabateau["lws"]
    #     hw             = self.carabateau["hws"]
    #     lwtimer        = self.carabateau["lwtimer"]
    #     hwtimer        = self.carabateau["hwtimer"]
    #     lwtimerGybe    = self.carabateau['gybeprolwtimer']
    #     hwtimerGybe    = self.carabateau['gybeprohwtimer']
    #     coeffboat      = self.carabateau['coeffboat']
    #     polairesglobales10to = self.polairesglobales10to
    #     tabvmg10to           = self.tabvmg10to
        
    #     ecartprecedent=iso[0,12].item()                                               # c est l ecart de temps de l iso precedent par rapport a t0
     
    #     n=len (iso)                                                                   # Longueur de l iso precedent sert a dupliquer 
    #     p=len(range_caps)                                                             # nombre de caps du rangecaps 
    #     dernier=iso[-1,1].item()                                                      # dernier point de l iso precedent   
        
    #     ordre=numiso-numisoini -1                                                       # ordre par rapport a numisoini pour calculer le dtiso 
    #     if tmini>3600:                                                                # cas normal a plus de 1h  de l objectif  
    #         dtiso=dtglobal[ordre]
       
    #     else :
    #         dtiso = torch.tensor(60.0, dtype=torch.float32, device='cuda:0')
         
    #     iso[:,0]  = numiso                                                              # on ajoute 1 au numero d iso sur toute la colonne  
    #     ecart=ecartprecedent+dtiso                                                      # c est l ecart de temps de l iso  par rapport a t0        
    #     iso[:,8]=torch.clamp(iso[:,8]-dtiso,min=0)                                     # pour le calcul de la penalite appliquable au trajet on defalque le dtiso qui permet de passer 
    #                                                                                     # de l iso precedent a celui que l on est en train de calculer et on bloque a 0 min 
    #     iso[:,11]=torch.round(iso[:,9]*10)
       
    #     iso            = iso.repeat((p, 1))                                                                # répète p fois les lignes (nb de caps)
    #     tws10          = iso[:,11] .to(torch.int)                                                          # nombre de points de l iso precedent pour pouvoir dupliquer les caps 
    #     caps_expanded  = range_caps.repeat_interleave(n)                                                   # on duplique les caps (cap1,cap1,cap1,cap2,cap2,cap2,cap3,cap3.......)
    #     capsR_expanded = range_capsR.repeat_interleave(n)
    #     iso[:,12]      = caps_expanded                                                                     # caps dupliques en col12
    #     iso[:,13]      = capsR_expanded                                                                    # caps en radians dupliqués en col13
    #     iso[:,2]       = iso[:,1]                                                                          # on enregistre les numeros de points comme des numeros de point mere 
       

    #     # calcul suivant des twa entieres au lieu de caps entiers
    #     iso[:,14] = torch.round(ftwato(iso[:,12],iso[:,10]) )                                                           # twa    arrondies  
    #     iso[:,12] = fcapto (iso[:,14],iso[:,10])
    #     iso[:,13]  =iso[:,12].deg2rad()


    #     # # calcul de la voile
    #     iso[:,14] = ftwato(iso[:,12],iso[:,10])                                                            # twa                                                     
    #     twa10=torch.round(torch.abs(iso[:,14]) * 10).to(torch.int)
     

    #     # print('iso.shape avant reduction vmg',iso.shape)
                                                             
    #     vmg_min = tabvmg10to[tws10,2]-self.tolerancehvmg       # on calcule vmgmin et max a partir de tws et twd 
    #     vmg_max = tabvmg10to[tws10,4]+self.tolerancehvmg

    #     vmg = torch.abs(iso[:, 14])
    #     mask = (vmg >= vmg_min) & (vmg <= vmg_max)              # la je reduit directement  le horsvmg 
    #     iso = iso[mask]                                         # Si necessaire print('iso.shape apres reduction vmg',iso.shape)

        
    #     tws10          = iso[:,11] .to(torch.int)   #on est oblige de redefinir la taille de tws10 et twa10   # si on avait pris iso ce serait pas necessaire
    #     twa10          = torch.round(torch.abs(iso[:,14]) * 10).to(torch.int)
           
  
    #     iso[:,15] = polairesglobales10to[iso[:,5].int(), tws10, twa10.int()]                               # vitesse voileini[voileini,tws10,twa10
    #     iso[:,16] = polairesglobales10to[7,  tws10, twa10]                                                 # vitesse meilleure voile[voileini,tws10,twa10
    #     iso[:,17] = polairesglobales10to[8,  tws10, twa10]                                                 # meilleure voile
    #     iso[:,18] = iso[:,16]/(iso[:,15]+0.0001)                                                           # Boost remplace twa10
    #     iso[:,19] = torch.where(iso[:,18]>1.014,iso[:,17],iso[:,5])                                        # voile definitive 
    #     iso[:,5]  = iso[:,19]                                                                              #*** on met la nouvelle voile dans la colonne 5 a la place de l ancienne
       
        
    #     # calcul des penalites
    #     #**********************
    #     iso[:,19] = torch.where(iso[:,18]>1.014,1,0)                                                       # on remplit la colonne chgt a la place de voiledef
    #     iso[:,18] = (iso[:,6]*iso[:,14])<0                                                                 # on remplit la colonne 16 Tgybe a la place de boost  (signe de twam1*twa10
        
    #     iso[:,6]  = iso[:,14]                                                                              # on met la nouvelle twa a la place de l ancienne  
    #     Cstamina  = 2 - 0.015 * iso[:,7]                                                                   # coefficient de stamina en fonction de la staminaini
    #     # création de masques
    #     mask_chgt = iso[:,19] != 0
    #     mask_gybe = iso[:,18] != 0
    
    #     # initialisation des résultats à 0
    #     tempspenochgt = torch.zeros_like(iso[:,19])                               
    #     tempspenoTG   = torch.zeros_like(iso[:,18])
    #     peno          = torch.zeros_like(iso[:,18])
    #     # calcul seulement pour les éléments où nécessaire
    #     if mask_chgt.any():       
    #         tempspenochgt[mask_chgt] = ( splineto(lw, hw, lwtimer, hwtimer, iso[mask_chgt,9]) * MF * Cstamina[mask_chgt]   )
        
    #     if mask_gybe.any():
    #         tempspenoTG[mask_gybe]   = ( splineto(lw, hw, lwtimerGybe, hwtimerGybe, iso[mask_gybe,9]) * Cstamina[mask_gybe]  )

            
    #     peno =   tempspenochgt+ tempspenoTG
    #     # on diminue de l ecart de temps  sur la colonne decenchement peno qui est un ecart par rappport au moment de la peno
    #     iso[:,21]=torch.clamp(iso[:,21]-dtiso,min=0,max=retardpeno)  

    #      # Application des anciennes penalites applicables sur le dti en colonne 6 (qui sont en  colonne 9)    
    #     #*****************************************************
    #     # si l ecart est egal a zero et que l on a une penalite en attente, on applique la penalite 
    #     mask4=(iso[:,21]==0)
    #     iso[:,17]= (dtiso) -(mask4* torch.clamp(0.3*iso[:,20],max=dtiso*0.3))   # la on met directement dans le dti  en plafonnant la penalite applicable a dtiso *0.3
       
       
         
    #     # si Des nouvelles penalites viennent en plus des penalites existantes                 
    #     mask5 = (peno != 0) & (iso[:, 20] != 0)
    #     #si oui on applique alors les penalites de la colonne 20 independamment du temps restant   sur le dti   
    #     iso[mask5, 17] = iso[mask5, 17] - ( torch.clamp(iso[mask5,20]*0.3,max=dtiso*0.3)) 
        
    #     # on remet a zero les valeurs de penalites qui ont ete utilisees et les valeurs de peno qui ont ete utilisees 
    #     #*************************************************
    #     iso[mask4, 20]  = torch.clamp(iso[mask4,20]-dtiso,min=0)
    #     iso[mask5, 20]  = torch.clamp(iso[mask5,20]-dtiso,min=0)
    #     iso[mask4, 21] = 0
    #     iso[mask5, 21] = 0

          
    #     # on applique les nouvelles penalites et les nouveaux decalages sur des cases qui sont toutes egales a zero 
    #     maskpeno=(peno != 0)
    #     iso[maskpeno, 20] = peno[maskpeno]    
    #     iso[maskpeno, 21] = retardpeno
            
    #     # ancien calcul de penalite on rajoutait a la peno precedente 
    #     #iso[:,8]  = iso[:,8]+tempspenochgt+tempspenoTG  
    #     #iso[:,17]=dtiso-0.3*torch.clamp(iso[:,8],max=dtiso )   
        
    #     # calcul des stamina 
    #     #******************
    #     #***  on rajoute les penalites sur l iteration le -dtiso sera applique a la fin avec un clamp
    #     iso[:,7]  = iso[:,7] - calc_perte_stamina_to(iso[:,9], iso[:,18],iso[:,19], coeffboat)  +   frecupstaminato(dtiso,iso[:,9])    # la stamina est egale a l ancienne (col4 )-perte (tws,TG,Chgt,coeff,MF)  + frecupstaminato(dtiso,Tws,pouf=0.8):
    #     iso[:,7]  = torch.clamp(iso[:,7],min=0,max=100)                                                                          #*** le max pourra eventuellement etra passse a 110 avec une boisson 
        
              
    #     # Calcul des nouvelles coordonnees
    #     #**************************************
        
    #     iso[:,18]=iso[:,3]                                        # on copie la latitude initiale en 18 pour les barrieres 
    #     iso[:,19]=iso[:,4]                                        # on copie la longitude initiale en 19 pour calculer les barrieres 
    #     # nouvelles coordonnees 
    #     iso[:,3]= iso[:,18] + iso[:,17] * iso[:,16] / 3600 / 60 * torch.cos(iso[:,13])                                         #  
    #     iso[:,4]= iso[:,19] + iso[:,17] * iso[:,16] / 3600 / 60 * torch.sin(iso[:,13])/torch.cos(iso[:,3].deg2rad())                      #    (le cos utilise pour la lat est avec la latitude deja calculee ?)
    #     iso[:,11]= iso[:,3].deg2rad()                                # on stocke la lat en rad pour le calcul de distance a l arrivee
    #     iso[:,12]= iso[:,4].deg2rad() 
        
    #     # on elimine les points hors cercle                          la distance par rapport au centre est calculee en colonne 15 anciennement colonne twa           latcR, loncR                                                     
    #     iso[:,17]=dist(iso[:,11],iso[:,12],latcR, loncR) 
    #     maskDiCentreRoutage   = iso[:,17]<(rayonRoutage*1.15)
    #     iso    = iso[maskDiCentreRoutage] 
    
    #     # calcul de distar, du point le plus proche et du temps estime vers l arrivee     
    #     iso[:,9]=dist(iso[:,11],iso[:,12],aritoR[0], aritoR[1]) *1.852       # Calcul de distar en mN en colonne 9 on utilise les valeurs en radians deja calculees pour les points et pour l arrivee 


    #     distmini, idx_min = torch.min(iso[:, 9], dim=0)                # idx_min est l’indice de la ligne où la distance est minimale   
    #     vitesse = iso[idx_min, 16]                                     # Vitesse en nœuds (milles nautiques par heure) au point de distance mini
    #     tmini   = ((distmini * 3600) / (vitesse * 1.852)).item()       # la distance est en km avec ma fonction dist et vitesse en noeuds pour les polaires 
    #     nptmini = int(iso[idx_min, 1].item())                          # transformation de idxmin en indice entier   
    
    #     # on va eliminer les points qui sont a plus de 3 fois la distance mini de l arrivee 
    #     maskDiMini   = iso[:,9]<(distmini*3)
    #     iso          = iso[maskDiMini]
     
    #     # calcul de ordoar et arrondi pour avoir le nb de points voulu a la fin 
    #     iso[:,10]= iso[:,11] - m_ar * iso[:,12]                                  # calcul ordoar :  ordonnee a  l origine  
    #     ordomini, ordomaxi = torch.min(iso[:,10]), torch.max(iso[:,10])          # min et max sur la colonne 10
                                                                   
    #     coeff         = (n3-1)/ (ordomaxi-ordomini)                              # coefficient pour ecremer et garder n3 points
    #     iso[:,10]  = (iso[:,10]*coeff).int() 
    
    #     # tris et elimination 
    #     indices_9 = torch.argsort(iso[:, 9], stable=False)                       # indice des elements tries    
    #     iso = iso[indices_9]                                                     # on reconstitue iso avec le tableau des elements triés sur l indice 9
    #     indices_10 = torch.argsort(iso[:, 10], stable=True)                      # indices des elements triés de la colonne 10
    #     iso = iso[indices_10]                                                    # on reconstitue iso avec le tableau des elements triés sur l indice 10
    #     torch.cuda.synchronize()  # on attend que le GPU ait fini pour mesurer le temps 
    #     iso[1:,14]= iso[1:,10] - iso[:-1,10]                                     # on fait l ecart entre 2 points successifs sur la colonne 10 ordoar dans la colonne 11
    #     mask   = iso[:,14]!=0
    #     iso    = iso[mask]                                                       # on elimine les points pour lesquels l ecart est egal a zero   
    #     #  on recalcule le boost sur 512 points pour l avoir a la fin 
    #     iso[:,14]= iso[:,16]/(iso[:,15]+0.0001)
    
        
    #     # elimination des points terre 
    #     iso[:,12]=terremer(iso[:,3],iso[:,4])                                   #on stocke le resultat terre mer en 12   
    #     mask   = iso[:,12]!=0
    #     iso    = iso[mask]                                                      # on garde les points pour lesquels iso[:,12]==1 
    #     try:
    #         # elimination des points dans les zones d exclusion 
    #         maskexclu = points_in_any_polygon_vectorized(iso[:,3],iso[:,4], self.segments)
    #         iso    = iso[~maskexclu] 
    #     except:
    #         None
        
    #     try:
    #         # calcul des barrieres et elimination des trajets traversant les barrieres
    #         mask=detect_barrier_crossings(iso[:,18],iso[:,19],iso[:,3],iso[:,4] , self.barrieres)
    #         iso=iso[~mask]
    #     except:
    #         None
    #     # recalcul du vent sur les points pour transmission vent ini a chemin
    #     #a voir suivant dash    tisovent= t_c[numisom1-1]

    #     # on calcule le vent pour la prochaine iteration 
       
    #     dtigiso= dtig0GFS+ecart
          
    #     #print ('ecart grib en h ',ecart/3600)


    #     iso[:,9],iso[:,10]= prevision025todtig(GRGFS_gpu,dtigiso, iso[:,3],iso[:,4])    # correction erreur le vent est calcule pour le prochain point avec les nuvelles coordonnees
      
    #     iso[:,11]=torch.rad2deg(iso[:,13])            # on remet le cap initial en 11 a partir du cap en radian que l on a toujours   
    #     iso[:,14] = iso[:,16]/(iso[:,15]+0.0001)      # on va remettre le boost en colonne 14 a la place de la twa   
    #     iso[:,13] = iso[:,16]   # copie de la vitesse max en 13 pour la passer a isoglobal
    #     iso[:,12] = ecart   
        
    #     # renumerotation 
    #     iso[:,1]= dernier+torch.arange(len(iso)) +1                         # on va renumeroter les points 
      
    #     # Copie des points de l isochrone dans isoglobal
    #    # print ('premier iso ' ,iso[0])

    #     torch.cuda.synchronize()           # Pour attendre que la synchronisation soit complete et essayer d eviter des erreurs 
    #     premier= int(iso[0,1].item())
    #     dernier= int(iso[-1,1].item())
    #     # print ('numiso {} dtiso {} premier {} dernier {} shape {} isoglobal.shape {} '.format(numiso ,dtiso,premier,dernier,iso.shape,self.isoglobal.shape))
       
        
    #     self.isoglobal[premier:dernier+1, :] = iso[:, 0:15]
        
    #     return iso, tmini,distmini,nptmini
   
    




    # VERSION AVEC NOUVELLES FORMULES DE PENO DU 15 JANVIER 2026
    # ****************


    # def isoplusun(self,iso,tmini,paramRoutage,mode ):
    #     ''' Donnees a charger propres au routage'''
    #     ''' polairesglobales10to, lw, hw, lwtimer, hwtimer,MF, coeffboat, rayonRoutage     '''
    #     ''' necessite comme donnneees externes  GRGFS_gpu ,polairesglobales10to,range_caps,range_capsR,carabateau,lw,....                                   '''
    #     ''' Les penalites sont affichees sur l iteration et ne sont diminuees du temps de l iteration que sur l iteration suivante '''

    #     MF          = 0.8
    #     # tolerancehvmg = self.tolerancehvmg         #tolerance de hvmg
    #     numisom1=int(iso[0,0].item())                                                 # numero de l'iso precedent 
    #     numiso=numisom1+1    
    #     t0          = self.t0

    #     dtig0GFS       = t0-tigGFS
    #     if mode=='mixte':
    #         dtig0ECM       = t0-tigECM
    #     # print('t0 ',time.strftime(" %d %b %H:%M %S",time.localtime(t0)))
    #     # print('tigGFS ',time.strftime(" %d %b %H:%M %S",time.localtime(tigGFS)))
    #     # print ('dtig0GFS en h ',dtig0GFS/3600)
        
    #     ecartprecedent=iso[0,12].item()         # c est l ecart de temps de a la fin de l iso precedent par rapport a t0 , c est le temps ou sont les points de depart de calcul de l iso 
       
       
        

    #     # print('iso.shape',iso.shape)
    #     n3=512
    #     range_caps     = paramRoutage["range_caps"]
    #     range_capsR    = paramRoutage["range_capsR"]
    #     centreRoutageR = paramRoutage['centreRoutageR']
    #     rayonRoutage   = paramRoutage['rayonRoutage']
    #     aritoR         = paramRoutage ["aritoR"]
    #     deptoR         = paramRoutage ["deptoR"]
    #     m_ar           = paramRoutage ["m_ar"]
    #     numisoini      = paramRoutage ["numisoini"]
    #     indiceroutage  = paramRoutage ["indiceroutage"]
    #     dtglobal       = paramRoutage ["dtglobal"]
    #     retardpeno     = paramRoutage ["retardpeno"]


    #     #print ('retardpeno et tolerance hvmg', retardpeno,self.tolerancehvmg)
        
    #     latcR, loncR   = centreRoutageR
        
    #     lw             = self.carabateau["lws"]
    #     hw             = self.carabateau["hws"]
    #     lwtimer        = self.carabateau["lwtimer"]
    #     hwtimer        = self.carabateau["hwtimer"]
    #     lwtimerGybe    = self.carabateau['gybeprolwtimer']
    #     hwtimerGybe    = self.carabateau['gybeprohwtimer']
    #     coeffboat      = self.carabateau['coeffboat']
    #     polairesglobales10to = self.polairesglobales10to
    #     tabvmg10to           = self.tabvmg10to
        
    #     ecartprecedent=iso[0,12].item()                                               # c est l ecart de temps de l iso precedent par rapport a t0
     
    #     n=len (iso)                                                                   # Longueur de l iso precedent sert a dupliquer 
    #     p=len(range_caps)                                                             # nombre de caps du rangecaps 
    #     dernier=iso[-1,1].item()                                                      # dernier point de l iso precedent   
        
       
    #     # if numiso==1: 
    #     #     print()
    #     #     print ('dans iso+1 tiso pour prev meteo du  deuxieme isochrone ', time.strftime(" %d %b %H:%M %S",time.localtime(tiso.item())))
    #     #     print()
        
    #     ordre=numiso-numisoini -1                                                       # ordre par rapport a numisoini pour calculer le dt 
    #     if tmini>3600:                                                                # cas normal a plus de 1h  de l objectif  
    #         dt=dtglobal[ordre]
       
    #     else :
    #         dt = torch.tensor(60.0, dtype=torch.float32, device='cuda:0')
         
    #     iso[:,0]  = numiso                                                              # on ajoute 1 au numero d iso sur toute la colonne  
    #     ecart=ecartprecedent+dt                                                         # c est l ecart de temps de l iso  par rapport a t0
        
    #     # print('iso {} tmini {} '.format(numiso,tmini ))
      
    
        
    #     # if (numiso!=1):       # Pour le premier isochrone la penalite appliquable au trajet a deja ete amputee du dt  de 60 secondes dans  isodepart  
        
    #     iso[:,8]=torch.clamp(iso[:,8]-dt,min=0)                                     # pour le calcul de la penalite appliquable au trajet on defalque le dt qui permet de passer 
    #                                                                                     # de l iso precedent a celui que l on est en train de calculer et on bloque a 0 min 
        
    #     iso[:,11]=torch.round(iso[:,9]*10)
       
    #     iso            = iso.repeat((p, 1))                                                                # répète p fois les lignes (nb de caps)
    #     tws10          = iso[:,11] .to(torch.int)                                                          # nombre de points de l iso precedent pour pouvoir dupliquer les caps 
    #     caps_expanded  = range_caps.repeat_interleave(n)                                                   # on duplique les caps (cap1,cap1,cap1,cap2,cap2,cap2,cap3,cap3.......)
    #     capsR_expanded = range_capsR.repeat_interleave(n)
    #     iso[:,12]      = caps_expanded                                                                     # caps dupliques en col12
    #     iso[:,13]      = capsR_expanded                                                                    # caps en radians dupliqués en col13
    #     iso[:,2]       = iso[:,1]                                                                          # on enregistre les numeros de points comme des numeros de point mere 
       

    #     # calcul suivant des twa entieres au lieu de caps entiers
    #     iso[:,14] = torch.round(ftwato(iso[:,12],iso[:,10]) )                                                           # twa    arrondies  
    #     iso[:,12] = fcapto (iso[:,14],iso[:,10])
    #     iso[:,13]  =iso[:,12].deg2rad()






    #     # # calcul de la voile
    #     iso[:,14] = ftwato(iso[:,12],iso[:,10])                                                            # twa                                                     
    #     # iso[:,18] = torch.round(abs(  iso[:,14])* 10  ) 
    #     twa10=torch.round(torch.abs(iso[:,14]) * 10).to(torch.int)
    #     # la je peux directement reduire le horsvmg 

    #     # print('iso.shape avant reduction vmg',iso.shape)
    #                                                            # on calcule vmgmin et max a partir de tws et twd 
    #     vmg_min = tabvmg10to[tws10,2]-self.tolerancehvmg
    #     vmg_max = tabvmg10to[tws10,4]+self.tolerancehvmg

    #     vmg = torch.abs(iso[:, 14])
    #     mask = (vmg >= vmg_min) & (vmg <= vmg_max)

    #     iso = iso[mask]
        
    #     tws10          = iso[:,11] .to(torch.int)   #on est oblige de redefinir la taaille de tws10 et twa10   # si on avait pris iso ce serait pas necessaire
    #     twa10=torch.round(torch.abs(iso[:,14]) * 10).to(torch.int)
    #     # print('iso.shape apres reduction vmg',iso.shape)


        
    #     #twa10 ftwato (cap,twd)*10)     (non signee)
    #     iso[:,15] = polairesglobales10to[iso[:,5].int(), tws10, twa10.int()]                               # vitesse voileini[voileini,tws10,twa10
    #     iso[:,16] = polairesglobales10to[7,  tws10, twa10]                                                 # vitesse meilleure voile[voileini,tws10,twa10
    #     iso[:,17] = polairesglobales10to[8,  tws10, twa10]                                                 # meilleure voile
    #     iso[:,18] = iso[:,16]/(iso[:,15]+0.0001)                                                                               # Boost remplace twa10
    #     iso[:,19] = torch.where(iso[:,18]>1.014,iso[:,17],iso[:,5])                                                            # voile definitive 
    #     iso[:,5]  = iso[:,19]                                                                                                  #*** on met la nouvelle voile dans la colonne 5 a la place de l ancienne
       
        
    #     # calcul des penalites
    #     iso[:,19] = torch.where(iso[:,18]>1.014,1,0)                                                                            # on remplit la colonne chgt a la place de voiledef
    #     iso[:,18] = (iso[:,6]*iso[:,14])<0                                                                                      # on remplit la colonne 16 Tgybe a la place de boost  (signe de twam1*twa10
        
    #     iso[:,6]  = iso[:,14]                                                                                                  # on met la nouvelle twa a la place de l ancienne  
    #     Cstamina  = 2 - 0.015 * iso[:,7]                                                                                        # coefficient de stamina en fonction de la staminaini
    #     # création de masques
    #     mask_chgt = iso[:,19] != 0
    #     mask_gybe = iso[:,18] != 0
    
    #     # initialisation des résultats à 0
    #     tempspenochgt = torch.zeros_like(iso[:,19])                               
    #     tempspenoTG   = torch.zeros_like(iso[:,18])
    
    #     # calcul seulement pour les éléments où nécessaire
    #     if mask_chgt.any():       
    #         tempspenochgt[mask_chgt] = ( splineto(lw, hw, lwtimer, hwtimer, iso[mask_chgt,9]) * MF * Cstamina[mask_chgt]   )
        
    #     if mask_gybe.any():
    #         tempspenoTG[mask_gybe]   = ( splineto(lw, hw, lwtimerGybe, hwtimerGybe, iso[mask_gybe,9]) * Cstamina[mask_gybe]  )
       
    #     iso[:,8]  = iso[:,8]+tempspenochgt+tempspenoTG                                                                           #***  on rajoute les penalites sur l iteration le -dt sera applique a la fin avec un clamp
    #     iso[:,7]  = iso[:,7] - calc_perte_stamina_to(iso[:,9], iso[:,18],iso[:,19], coeffboat)  +   frecupstaminato(dt,iso[:,9])    # la stamina est egale a l ancienne (col4 )-perte (tws,TG,Chgt,coeff,MF)  + frecupstaminato(dt,Tws,pouf=0.8):
    #     iso[:,7]  = torch.clamp(iso[:,7],min=0,max=100)                                                                          #*** le max pourra eventuellement etra passse a 110 avec une boisson 
        
    #     # # # calcul des nouvelles coordonnees

    #     # print(type(dt))
    #     iso[:,17]=dt-0.3*torch.clamp(iso[:,8],max=dt )                                                                           # dt remplace boost en colonne 17    
        
    #     iso[:,18]=iso[:,3]                                                                                      # on copie la latitude initiale en 18 pour les barrieres 
    #     iso[:,19]=iso[:,4]                                                                                      # on copie la longitude initiale en 19 pour calculer les barrieres 
    #     # nouvelles coordonnees 
    #     iso[:,3]= iso[:,18] + iso[:,17] * iso[:,16] / 3600 / 60 * torch.cos(iso[:,13])                                         #  
    #     iso[:,4]= iso[:,19] + iso[:,17] * iso[:,16] / 3600 / 60 * torch.sin(iso[:,13])/torch.cos(iso[:,3].deg2rad())                      #    (le cos utilise pour la lat est avec la latitude deja calculee ?)
    #     iso[:,11]= iso[:,3].deg2rad()                                # on stocke la lat en rad pour le calcul de distance a l arrivee
    #     iso[:,12]= iso[:,4].deg2rad() 
        
    #     # on elimine les points hors cercle                          la distance par rapport au centre est calculee en colonne 15 anciennement colonne twa           latcR, loncR                                                     
    #     iso[:,17]=dist(iso[:,11],iso[:,12],latcR, loncR) 
    #     maskDiCentreRoutage   = iso[:,17]<(rayonRoutage*1.15)
    #     iso    = iso[maskDiCentreRoutage] 
    
    #     # calcul de distar, du point le plus proche et du temps estime vers l arrivee     
    #     iso[:,9]=dist(iso[:,11],iso[:,12],aritoR[0], aritoR[1]) *1.852       # Calcul de distar en mN en colonne 9 on utilise les valeurs en radians deja calculees pour les points et pour l arrivee 


    #     distmini, idx_min = torch.min(iso[:, 9], dim=0)                # idx_min est l’indice de la ligne où la distance est minimale   
    #     vitesse = iso[idx_min, 16]                                     # Vitesse en nœuds (milles nautiques par heure) au point de distance mini
    #     tmini   = ((distmini * 3600) / (vitesse * 1.852)).item()       # la distance est en km avec ma fonction dist et vitesse en noeuds pour les polaires 
    #     nptmini = int(iso[idx_min, 1].item())                          # transformation de idxmin en indice entier   
    
    #     # on va eliminer les points qui sont a plus de 3 fois la distance mini de l arrivee 
    #     maskDiMini   = iso[:,9]<(distmini*3)
    #     iso          = iso[maskDiMini]
     
    #     # calcul de ordoar et arrondi pour avoir le nb de points voulu a la fin 
    #     iso[:,10]= iso[:,11] - m_ar * iso[:,12]                                  # calcul ordoar :  ordonnee a  l origine  
    #     ordomini, ordomaxi = torch.min(iso[:,10]), torch.max(iso[:,10])          # min et max sur la colonne 10
                                                                   
    #     coeff         = (n3-1)/ (ordomaxi-ordomini)                              # coefficient pour ecremer et garder n3 points
    #     iso[:,10]  = (iso[:,10]*coeff).int() 
    
    #     # tris et elimination 
    #     indices_9 = torch.argsort(iso[:, 9], stable=False)                       # indice des elements tries    
    #     iso = iso[indices_9]                                                     # on reconstitue iso avec le tableau des elements triés sur l indice 9
    #     indices_10 = torch.argsort(iso[:, 10], stable=True)                      # indices des elements triés de la colonne 10
    #     iso = iso[indices_10]                                                    # on reconstitue iso avec le tableau des elements triés sur l indice 10
    #     torch.cuda.synchronize()  # on attend que le GPU ait fini pour mesurer le temps 
    #     iso[1:,14]= iso[1:,10] - iso[:-1,10]                                     # on fait l ecart entre 2 points successifs sur la colonne 10 ordoar dans la colonne 11
    #     mask   = iso[:,14]!=0
    #     iso    = iso[mask]                                                       # on elimine les points pour lesquels l ecart est egal a zero   
    #     #  on recalcule le boost sur 512 points pour l avoir a la fin 
    #     iso[:,14]= iso[:,16]/(iso[:,15]+0.0001)
    # ## Calcul de la meteo

    #     dtigGFS = dtig0GFS + ecart
       
           
    #     #iso[:,9],iso[:,10]   = prevision025todtig(GRGFS_gpu,dtigGFS, iso[:,3],iso[:,4])    # correction erreur le vent est calcule pour le prochain point avec les nuvelles coordonnees
        
    #     if mode=='mixte':
    #         dtigECM = dtig0ECM + ecart
    #         if ecart<9*3600 :
    #             iso[:,9],iso[:,10]   = gfs_interp(dtigGFS,iso[:,3],iso[:,4] )    
    #         else :
    #             iso[:,9],iso[:,10]   = ecm_interp(dtigECM,iso[:,3],iso[:,4] ) 
                
    #     else:  
    #         iso[:,9],iso[:,10]   = gfs_interp(dtigGFS,iso[:,3],iso[:,4] )    

    #     #iso[:,9],iso[:,10]   = gfs_interp(dtigGFS,iso[:,3],iso[:,4] ) 
    #     # elimination des points terre 
    #     iso[:,12]=terremer(iso[:,3],iso[:,4])                                   #on stocke le resultat terre mer en 12   
    #     mask   = iso[:,12]!=0
    #     iso    = iso[mask]                                                      # on garde les points pour lesquels iso[:,12]==1 
    #     try:
    #         # elimination des points dans les zones d exclusion 
    #         maskexclu = points_in_any_polygon_vectorized(iso[:,3],iso[:,4], self.segments)
    #         iso    = iso[~maskexclu] 
    #     except:
    #         None
        
    #     try:
    #         # calcul des barrieres et elimination des trajets traversant les barrieres
    #         mask=detect_barrier_crossings(iso[:,18],iso[:,19],iso[:,3],iso[:,4] , self.barrieres)
    #         iso=iso[~mask]
    #     except:
    #         None
    #     # recalcul du vent sur les points pour transmission vent ini a chemin
    #     #a voir suivant dash    tisovent= t_c[numisom1-1]

    #     # on calcule le vent pour la prochaine iteration 
       
      
    #     #iso[:,9],iso[:,10]= prevision025todtig(GRGFS_gpu,dtigiso, iso[:,3],iso[:,4])    # correction erreur le vent est calcule pour le prochain point avec les nuvelles coordonnees
      
    #     iso[:,11]=torch.rad2deg(iso[:,13])            # on remet le cap initial en 11 a partir du cap en radian que l on a toujours   
    #     iso[:,14] = iso[:,16]/(iso[:,15]+0.0001)      # on va remettre le boost en colonne 14 a la place de la twa   
    #     iso[:,13] = iso[:,16]   # copie de la vitesse max en 13 pour la passer a isoglobal
    #     iso[:,12] = ecart   
        
    #     # renumerotation 
    #     iso[:,1]= dernier+torch.arange(len(iso)) +1                         # on va renumeroter les points 
      
    #     # Copie des points de l isochrone dans isoglobal
    #    # print ('premier iso ' ,iso[0])

    #     torch.cuda.synchronize()           # Pour attendre que la synchronisation soit complete et essayer d eviter des erreurs 
    #     premier= int(iso[0,1].item())
    #     dernier= int(iso[-1,1].item())
    #     # print ('numiso {} dt {} premier {} dernier {} shape {} isoglobal.shape {} '.format(numiso ,dt,premier,dernier,iso.shape,self.isoglobal.shape))
       
        
    #     self.isoglobal[premier:dernier+1, :] = iso[:, 0:15]
        
    #     return iso, tmini,distmini,nptmini
    

    # # FIN DE L ANCIENNE VERSION 
    # #****************************






def routageGlobal(course,user_id,isMe,ari,y0,x0,t0,tolerancehvmg,optionroutage,mode):                              # version Vrouteur 1/12/2025
    ''' Calcule le routage '''
    ''' Recupère les données et parcoure les differents waypoints '''

    session         = RoutageSession(course, user_id,isMe,ari) 

    waypoints       = session.waypoints
    tabvmg10to      = session.tabvmg10to
    iso             = session.isodepart            # la on est systematiquement sur ma pposition 
    posStartVR      = session.posStartVR
    posStart        = session.posStart
    x0=lon_to_360(x0)


    print()
    print ('(4784) waypoints ',waypoints)
    print()       

    for wp in waypoints:
        wp[3] = lon_to_360(wp[3])
    # print ('4791 x0',x0)    

    x0=lon_to_360(x0)
    # print ('\n Demande de routage global ')
    # print ('course',course)
    # print ('ari',ari)
    # print ('Option routage ',optionroutage)
    # print ('\nposStartVR\n',posStartVR)
    # print ('\nposStart  \n',posStart)
    # print()


    # suivant l option de routage, on va changer sessionposStartVR
    #positionvr=torch.tensor([0,t0vr,dt1,option,valeur,y0vr,x0vr,voile,twavr,headingvr,speedvr,staminavr,penovr,twdvr,twsvr,voileAuto,boost],dtype=torch.float64,device='cpu')

    if optionroutage==1:

        # print ('On est dans l option de routage option1 avant modif posStartVR',posStart)
        posStart['t0']= t0               # transformation de t0 en tenseur    
        posStart['y0']= y0               # transformation de t0 en tenseur
        posStart['x0']= lon_to_360(x0)
        print ('iso \n',iso )
        iso[0,3]=y0
        iso[0,4]=x0

        # print ('\n On est dans l option de routage option1 posStartVR apresmodif ',posStart)
   
    if optionroutage==2:
        session.posStartVR[1]= torch.tensor([t0], dtype=torch.float64)               # transformation de t0 en tenseur
        session.posStartVR[5]= torch.tensor([y0], dtype=torch.float64)               # transformation de t0 en tenseur
        session.posStartVR[6]= torch.tensor([x0], dtype=torch.float64)      


    if isMe=='no':
        print ('On est dans l option de routage option1 avant modif posStartVR',posStart)
        posStart['t0']= t0               # transformation de t0 en tenseur    
        posStart['y0']= y0               # transformation de t0 en tenseur
        posStart['x0']= lon_to_360(x0)
        print ('iso \n',iso )
        iso[0,3]=y0
        iso[0,4]=x0

    tic=time.time()

    for indiceroutage in range(len(ari)):     # Parcourt les differents  waypoints 
        if indiceroutage==0:
            #posStart=session.posStart   
            if isMe=='no':
               posStart['y0']=y0
               posStart['x0']=x0   

               print ()
               print ('on est dans le cas du routage d un concurrent')
               print ('posStart',posStart)     
                
        else:
            posStart=posEnd

        #print ('\n On est dans l option de routage option1 posStart avant initialise ',posStart)
        paramRoutage,iso = session.initialiseRoutagePartiel(posStart,ari,indiceroutage)  # print ('Premier iso',iso)

        impression_tensor15 (iso,titre='\niso de depart correspondant à PositionVR decalee')
       

        tmini            = 10000
        distmini         = 1000
        rwp              = paramRoutage["rwp"]
        if rwp==0:            # probleme des lignes d arrivee a gerer ulterieurement
            rwp=0.2


        numisoini        = paramRoutage['posStartPartiel']['numisoini']
        # print ('rwp',rwp)
        # print ('numisoini',numisoini)
      
        while distmini > rwp:
            iso, tmini, distmini, nptmini = session.isoplusun(iso, tmini,paramRoutage,mode)
                   
        # Dernière itération pour rentrer dans le cercle 
        try:
            iso, tmini, distmini, nptmini     = session.isoplusun(iso, tmini,paramRoutage,mode)
        except:
            print ('Limite grib atteinte 2 ')
            distmini, idx_min = torch.min(iso[:, 9], dim=0)                # idx_min est l’indice de la ligne où la distance est minimale   
            vitesse = iso[idx_min, 16]                                     # Vitesse en nœuds (milles nautiques par heure) au point de distance mini
            tmini   = ((distmini * 3600) / (vitesse * 1.852)).item()       # la distance est en km avec ma fonction dist et vitesse en noeuds pour les polaires 
            nptmini = int(iso[idx_min, 1].item())    
                 

        distmini, idx_min = torch.min(iso[:, 9], dim=0)                # idx_min est l’indice de la ligne où la distance est minimale  
      
        pointfinal=torch.zeros(20)                                      # dernier point le plus pres 
        pointfinal[:15]=session.isoglobal [nptmini]
        # impression_tensor15 ([pointfinal],titre='Point d arrivee')
        # print ('\npointfinal',pointfinal)
        posEnd= pointfinalToPosEnd(pointfinal)
        derniereligne = int(iso[-1,1].item())
        pointfinal[0]+=1           #on incremente directement le numero d iso
        pointfinal[1]= derniereligne        
   
    derniereligne = int(iso[-1,1].item())                                     # derniere ligne du dernier iso donne le dernier numero de point et donc la derniere ligne de  de isoglobal   
    session.isoglobal = session.isoglobal[:derniereligne,:]                   # on ne garde que la partie contenant les valeurs calculees dans isoglobal
            
    dico_isochrones = {}                                                      # Dictionnaire pour stocker les courbes des isos 
    numerosIso = torch.unique(session.isoglobal[:, 0])

    for iso in numerosIso:
        mask = session.isoglobal[:, 0] == iso                                 # Masque pour sélectionner les lignes de cet isochrone
        lat_lon = session.isoglobal[mask][:, [3, 4]]                          # Extraction des colonnes lat/lon (3 et 4)    
        isodecoupe=decoupe_latlon(lat_lon, seuil=0.01)
        dico_isochrones[int(iso.item())] = isodecoupe                         # Ajout au dictionnaire (clé en int pour faciliter la lecture)
   
    dernieriso=pointfinal[0]
    tempsexe =time.time()-tic
    print('_________________________________________________________________________________________________________________________________________________________________________________________\
          \nRoutage {} effectué en {:4.2f}s  {} isochrones  {:4.3f}s par Iso  \
          \n_________________________________________________________________________________________________________________________________________________________________________________________\
           '.format(mode, tempsexe,dernieriso ,tempsexe/dernieriso) )  # Ajouter iso à session.isoglobal

    return waypoints,session.isoglobal,session.posStartVR,session.posStart,nptmini,session.exclusionsVR,session.tabvmg10to,dico_isochrones




def cheminToRoutage(chemin,tabvmg10to):

    ''' Transforme un chemin (extrait de isoglobal ) en tableau de routage '''
    routage =torch.zeros((len(chemin),17) , dtype=torch.float32, device='cuda')
    # reorganisation des colonnes 
    routage[:,[0,1,2,3,5,6,9,10,11,12,13,14,15]]=chemin[:,[0,12,3,4,6,11,13,5,14,9,10,7,8]]
    
    #decalage des colonnes pour que les actions a faire correspondent au temps a venir et non au temps passé 
    decaler_colonnes(routage, colonnes=[0,5,6,9,10,11,15])     # on decale les colonnes twa cap,  voile et boost car ce sont les valeurs pour les commandes a realiser a l a prendre 
    
    #calcul des vmg 
    tws10=torch.round(routage[:,12]*10).int()                                                            # on calcule vmgmin et max a partir de tws et twd 
    routage[:,7] = tabvmg10to[tws10,2]
    routage[:,8] = tabvmg10to[tws10,4]
    
    # on garde les avants dernieres valeurs pour la twa le cap la vitesse et la voile le boost et la stamina 
    routage[-1,[5,6,9,10,11,14]]=routage[-2,[5,6,9,10,11,14]]
    return routage
    

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




@app.route('/calculeroutage', methods=["GET", "POST"])
# c 'est celui qui est utilise par index nouveau modèle
def calculeroutage():
    print('\nDemande de routage sur url routageajax3\n*************************************')
    

    # username            = request.args.get('username')          # recupere les donnees correspondant a valeur dans le site html
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
    username            = findUsername(user_id)

    x0=lon_to_360(x0)
    print ('5003 x0 ',x0)
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
    #print ('routage  pour {}'.format( username))
    # try:
    #print ('5023 x0 ',x0)
    waypoints,isoglobal,posStartVR,posStart,nptmini,exclusions,tabvmg10to,dico_isochrones=routageGlobal(course, user_id,isMe,ari,y0,x0,t0,tolerancehvmg,optionroutage,mode)  
    chemin            = reconstruire_chemin_rapide(isoglobal, nptmini)
    routage           = cheminToRoutage(chemin,tabvmg10to)
    arrayroutage      = routage.cpu().tolist()
    routage_np        = np.array(arrayroutage,dtype=np.float64)
    routagelisse      = lissage(course,routage_np,t0,posStartVR,posStart)  
    tabtwa            = routagelisse[:,5]
    twasmooth         = smooth(tabtwa)                      #    c est du smooth torch 
    twasmooth2        = smooth(twasmooth)  
    routagelisse[:,5] = twasmooth2                   # c 'est juste une substitution de facade, il faudrait recalculer le routage  
    arrayroutage2     = [arr.tolist() for arr in routagelisse] 
    dico              = {'message':'Routage OK','waypoints':waypoints,'arrayroutage':arrayroutage,'arrayroutage2':arrayroutage2,'isochrones':dico_isochrones,'t0routage':t0}
    eta = t0 + routagelisse[-1,1]

    # except:
    #     waypoints,arrayroutage,arrayroutage2,dico_isochrones,t0,eta=0,0,0,0,0,0  
    #     dico={'message':'Erreur','waypoints':waypoints,'arrayroutage':arrayroutage,'arrayroutage2':arrayroutage2,'isochrones':dico_isochrones,'t0routage':t0}    
    #     print ('\n Erreur  dico pour debug ,_n',dico)
    
    
    print ('Routage pour {} en mode {} sur course {} le {} {} ETA {} \n******************************************************************'.format(username,course,mode,format_time(t0),dico['message'],format_time(eta)))
    print ('len',len(arrayroutage))
    eta=time.time()+10000
    insert_historoutages(  username,course, user_id,dico['message'], t0, y0, x0, eta)  # pour consulter les routages qui ont ete tentes et par qui 
    print('****************************************************************\n')
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

    # on calcule le a premiere perte de stamina et la premiere penalite 
    stam11,peno11=calculpenalitesnump(Tws1010, Chgt, twam1, twa, staminaini, carabateau)
    speed= polairesglobales10[7,Tws1010,Twa1011]
    dist11=(speed*peno11/3600)*.3
    peno11=float(peno11)
    
    # recuperation avant manoeuvre2
    recup20=frecupstamina(temps20*60,tws20)  
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

  


    # on calcule la penalite de la 2eme manoeuvre
    stam21,peno21=calculpenalitesnump(Tws1021, Chgt2, twam12, twa2, stam20, carabateau)
    speed2= polairesglobales10[7,Tws1021,Twa1021]
    dist21=(speed2*peno21/3600)*.3
    cumul21=dist11+dist21
    peno21=float(peno21)
    # recuperation avant la troisieme manoeuvre
    recup30=frecupstamina(temps30*60,tws30)
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
    stam31,peno31=calculpenalitesnump(Tws1031, Chgt3, twam13, twa3, stam30, carabateau)
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




def calculpenalitesnump(Tws10, Chgt, twam1, twa, staminaini, carabateau):
    """
    Revision de la formule de la perte le 28 02 2025
    Calcule les pénalités de stamina et de temps.
    Accepte soit des scalaires soit des np.array en entrée.
    retourne staminaend et temps de peno sous forme de numpy array si necessaire
    La stamina recuperee pendant dt n'est pas prise en compte
    Le temps de penalite est un temps brut le coefficient de 0.3ou 0.5 n'est pas applique
    """
    Perte=0
    MF=0.8
    Tws10 = np.asarray(Tws10, dtype=np.float64)
    tws=Tws10/10
    Chgt = np.asarray(Chgt, dtype=np.int32)
    twam1 = np.asarray(twam1, dtype=np.float64)
    twa = np.asarray(twa, dtype=np.float64)
    staminaini = np.asarray(staminaini, dtype=np.float64)

    TackGybe = (twam1 * twa < 0).astype(int)
    Tack = np.where((TackGybe == 1) & (np.abs(twa) < 90), 1, 0)
    Gybe = np.where((TackGybe == 1) & (np.abs(twa) >= 90), 1, 0)
    lw = carabateau['lws']
    hw = carabateau['hws']
    coeffboat = carabateau['coeffboat']
    Cstamina = 2 - 0.015 * staminaini

    m = np.select(
        [tws <= 10, (tws > 10) & (tws <= 20), (tws > 20) & (tws < 30), tws >= 30],
        [0, 0.2, 0.6, 0]
    )
    p = np.select(
        [tws <= 10, (tws > 10) & (tws <= 20), (tws > 20) & (tws < 30), tws >= 30],
        [10, 8, 0, 18]
    )
    Perte = ((m * tws + p) * (TackGybe + 2 * Chgt * MF)) * coeffboat
    staminaend = staminaini - Perte 
    staminaend = np.maximum(0, staminaend)

    # Initialisation de tempspeno en float
    tempspeno = np.zeros_like(tws, dtype=np.float64)
    tempspenom = np.zeros_like(tws, dtype=np.float64)

    if np.any(Chgt == 1):
        lwtimer = carabateau['lwtimer']
        hwtimer = carabateau['hwtimer']
        tempspeno += np.where( Chgt == 1,  spline(lw, hw, lwtimer, hwtimer, tws) * MF * Cstamina, 0 )

    if np.any(Gybe == 1):
        lwtimer = carabateau['gybeprolwtimer']
        hwtimer = carabateau['gybeprohwtimer']
        tempspenom += np.where( Gybe == 1, spline(lw, hw, lwtimer, hwtimer, tws) * Cstamina,   0    )

    if np.any(Tack == 1):
        lwtimer = carabateau['tackprolwtimer']
        hwtimer = carabateau['tackprohwtimer']
        tempspenom += np.where( Tack == 1, spline(lw, hw, lwtimer, hwtimer, tws) * Cstamina,   0    )


    if np.any(Chgt == 1) and (np.any(Gybe == 1) or np.any(Tack == 1) ):
        tempspeno=np.maximum(tempspeno,tempspenom)

    elif np.any(Chgt == 0) :
        tempspeno=tempspenom

    return staminaend, tempspeno





# @app.route('/chargecarte', methods=["GET", "POST"])
# def chargecarte():
#     print('on est dans chargecarte')
#     # points=request.args.get('points')
#     # points=json.loads(points)
#     data = request.get_json()  # Récupération du JSON envoyé
#     points = np.asarray(data["points"])  # Conversion en array numpy




#    # points=np.asarray(points)

#     b=np.ceil(points[:,0]).reshape(-1,1)
#     c=np.floor(points[:,1]).reshape(-1,1)
#     cartes=np.concatenate((b,c),axis=1)
#     listecartes=np.unique(cartes,axis=0)
#     # print ('Ligne 2940 points ',listecartes )
#     cartevr=[]
#     for i in range (len(listecartes)):
#         cartevr.append(charge_carto(listecartes[i,0],listecartes[i,1]))

#     # print ('ligne 2944 cartevr \n ',cartevr)
#     response=make_response(jsonify({'message':'Cartes envoyees par Serveur','carte':cartevr}))
#     print()
#     print(sys.getsizeof(response))
#     print()
#     response.headers.add('Access-Control-Allow-Origin', '*')  # Autorise toutes les origines
#     return response




@app.route('/rechercheflotte', methods=["GET", "POST"])
def rechercheflotte():
    # print ('On est dans rechercheflotte')
    id_user    = request.args.get('id_user')                                 # recupere les donnees correspondant a valeur dans le site html
    course     = request.args.get('course')                                 # recupere les donnees correspondant a valeur dans le site
  
    # print ('id_user {} course {} '.format(id_user,course))
    fleetinfos=rechercheTableFleetinfos(id_user, course)

    # print ('Ligne 3621 Fleetinfos',fleetinfos) 

    return jsonify({"message": 'Recuperation de la flotte effectué avec succes '  ,"fleetinfos":fleetinfos }), 200








if __name__ == "__main__":

    # start_scheduler()
    if socketioOn:
        socketio.run(app, host='0.0.0.0' if IS_PRODUCTION else '127.0.0.1', port=5000, debug=not IS_PRODUCTION)
    else:
       app.run  ( host='0.0.0.0' , port=5000, debug=True)

    