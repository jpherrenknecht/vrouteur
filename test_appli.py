import sys,torch,gc
import numpy as np
import math
import sqlite3 as sql
import time
import calendar
import folium
import requests
import psycopg2
import copy
import logging
import traceback


from datetime import datetime,timezone
from cachetools import TTLCache
from threading import Lock
from websocket import create_connection
from flask_socketio import SocketIO
from scipy.signal import savgol_filter         # necessaire pour lissage
from collections import defaultdict, deque     # necessaire pour la carte 
from pathlib import Path
from psycopg2 import pool
from fonctions2026 import *

from collections import OrderedDict



print('python : ',sys.executable)
# print("CUDA dispo :", torch.cuda.is_available())
device='cuda'
torch.set_default_device(device)
print('device :',torch.get_default_device())

Pi=math.pi
R_NM = 3440.065  # rayon moyen de la Terre en milles nautiques
KNOTS = 1.9438444924406048  # m/s -> kn
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

def gribFileName(basedir):
    ''' cherche le dernier grib complet disponible au temps en secondes '''
    ''' temps_secondes est par defaut le temps instantané '''
    ''' Cherche egalement le dernier grib chargeable partiellement'''
    temps_secondes=time.time()
    date_tuple       = time.gmtime(temps_secondes) 
    date_formatcourt = time.strftime("%Y%m%d", time.gmtime(temps_secondes))
    dateveille_tuple = time.gmtime(temps_secondes-86400) 
    dateveille_formatcourt=time.strftime("%Y%m%d", time.gmtime(temps_secondes-86400))
    mn_jour_utc =date_tuple[3]*60+date_tuple[4]
    #print ('mn_jour_utc dans gribFileNames ',mn_jour_utc)
    if (mn_jour_utc <3*60+40):                          #avant 3h 48 UTC le nom de fichier est 18 h de la veille 
        filename=basedir+"gfs_"+dateveille_formatcourt+"-18.npy"
        filenameC=basedir+"gfsC_"+dateveille_formatcourt+"-18.npz"
    elif (mn_jour_utc<9*60+40):   
        filename=basedir+"gfs_"+date_formatcourt+"-00.npy"
        filenameC=basedir+"gfsC_"+date_formatcourt+"-00.npz"
    elif (mn_jour_utc<15*60+40): 
        filename=basedir+"gfs_"+date_formatcourt+"-06.npy"
        filenameC=basedir+"gfsC_"+date_formatcourt+"-06.npz"
    elif (mn_jour_utc<21*60+40):   
        filename=basedir+"gfs_"+date_formatcourt+"-12.npy"
        filenameC=basedir+"gfsC_"+date_formatcourt+"-12.npz"
    else:                                              # entre 21h 48UTC  et minuit    
        filename=basedir+"gfs_"+date_formatcourt+"-18.npy"  
        filenameC=basedir+"gfsC_"+date_formatcourt+"-18.npz" 
    tig=dateheure(filename)[2]
    return filename ,filenameC,tig



# le save est la pour memoire , il est utilise dans le chargement du grib en cron 
def save_pack_npz(fileNameC, Iu, Iv, refu, refv, step, tig, indice, bias=32768):
    np.savez_compressed(
        fileNameC,
        Iu=Iu.astype(np.uint16, copy=False),
        Iv=Iv.astype(np.uint16, copy=False),
        refu=refu.astype(np.float32, copy=False),
        refv=refv.astype(np.float32, copy=False),
        step= np.float32(step),
        bias= np.int32(bias),
        tig = np.int32(tig),
        indice=np.uint32(indice),
        )
    
def load_pack_npz(fileNameC):
    z = np.load(fileNameC)
    Iu     = z["Iu"].astype(np.uint16, copy=False)
    Iv     = z["Iv"].astype(np.uint16, copy=False)
    refu   = z["refu"].astype(np.float32, copy=False)
    refv   = z["refv"].astype(np.float32, copy=False)
    step   = float(z["step"])
    bias   = int(z["bias"])
    tig    = int(z["tig"])
    indice = int(z["indice"])
    return Iu, Iv, refu, refv, step, bias,tig,indice


def to_gpu_i16(Iu_u16, Iv_u16, refu, refv, bias=32768, device="cuda"):
    # conversion uint16 -> int16 avec bias (GPU-friendly)
    Iu_i16 = (torch.from_numpy(Iu_u16).to(torch.int32) - bias).to(device=device, dtype=torch.int16)
    Iv_i16 = (torch.from_numpy(Iv_u16).to(torch.int32) - bias).to(device=device, dtype=torch.int16)
    refu_t = torch.from_numpy(refu).to(device=device, dtype=torch.float32)
    refv_t = torch.from_numpy(refv).to(device=device, dtype=torch.float32)
    return Iu_i16, Iv_i16, refu_t, refv_t



def save_pack_meta(filemeta, tig, indice):
    meta = {
        "tig": float(np.float64(tig)),
        "indice": int(np.uint32(indice)),
    }
    Path(filemeta).write_text(json.dumps(meta, separators=(",", ":")))



def load_pack_meta(fileMeta):
    p = Path(fileMeta)
    if not p.exists():
        return None
    return json.loads(p.read_text())

def need_reload(meta, tig, indice):
    if meta is None:
        return True
    if int(meta["indice"]) != int(indice):
        return True
    # tig float : tolérance sécurité
    if not math.isclose(meta["tig"], float(tig), rel_tol=0.0, abs_tol=1e-9):
        return True
    return False
    
def drop_old_interp():
    global gfs_interpolateur
    gfs_interpolateur = None
    gc.collect()
    torch.cuda.empty_cache()



class GFS025InterpGPU_721_I16:
    """
    Version GPU-friendly:
      - Iu, Iv : int16 (n_steps, 721, 1440) contenant (uint16 - bias)
      - refu, refv : float32 (n_steps,)
      - step : float32 scalaire (m/s)
      - bias : 32768 par défaut
    """

    def __init__(
        self,
        Iu_i16: torch.Tensor,
        Iv_i16: torch.Tensor,
        refu: torch.Tensor,
        refv: torch.Tensor,
        step: float,
        bias: int = 32768,
        clamp_min_kn=1.0,
        clamp_max_kn=70.0,
        dt_step_h=3.0,
        out_dtype=torch.float32,
    ):
        assert Iu_i16.ndim == 3 and Iv_i16.ndim == 3, "Iu/Iv doivent être (n_steps, 721, 1440)"
        assert Iu_i16.shape == Iv_i16.shape, "Iu et Iv doivent avoir la même shape"
        assert Iu_i16.dtype == torch.int16 and Iv_i16.dtype == torch.int16, "Iu/Iv doivent être torch.int16"
        assert refu.ndim == 1 and refv.ndim == 1 and refu.shape[0] == Iu_i16.shape[0], "refu/refv (n_steps,)"

        self.Iu = Iu_i16
        self.Iv = Iv_i16
        self.refu = refu.to(dtype=torch.float32, device=Iu_i16.device)
        self.refv = refv.to(dtype=torch.float32, device=Iu_i16.device)

        self.device = Iu_i16.device
        self.n_steps, self.ny, self.nx = Iu_i16.shape
        assert self.ny == 721 and self.nx == 1440, f"attendu (721,1440), reçu ({self.ny},{self.nx})"

        self.dt_step_h = float(dt_step_h)
        self.scale = 4.0  # 0.25° => 4 points/deg
        self.clamp_min = float(clamp_min_kn)
        self.clamp_max = float(clamp_max_kn)
        self.out_dtype = out_dtype

        self._one = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self._rad2deg = torch.tensor(180.0 / math.pi, device=self.device, dtype=torch.float32)
        self._step = torch.tensor(float(step), device=self.device, dtype=torch.float32)
        self._bias = torch.tensor(float(bias), device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, dtig, lat0, lon0, return_uv=False):
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
        y_f = (90.0 - lat) * self.scale
        x_f = torch.remainder(lon, 360.0) * self.scale

        iy = torch.floor(y_f).to(torch.long)
        ix = torch.floor(x_f).to(torch.long)

        iy = torch.clamp(iy, 0, self.ny - 2)
        ix = torch.remainder(ix, self.nx)

        iy1 = iy + 1
        ix1 = torch.remainder(ix + 1, self.nx)

        dy = torch.clamp(y_f - iy.to(torch.float32), 0.0, 1.0)
        dx = torch.clamp(x_f - ix.to(torch.float32), 0.0, 1.0)

        ax = self._one - dx
        ay = self._one - dy

        # ----- temps -----
        t = (dt / 3600.0) / self.dt_step_h
        it = torch.floor(t).to(torch.long)
        wt = torch.clamp(t - it.to(torch.float32), 0.0, 1.0)
        it = torch.clamp(it, 0, self.n_steps - 2)

        # refs (N,)
        ru0 = self.refu[it]
        ru1 = self.refu[it + 1]
        rv0 = self.refv[it]
        rv1 = self.refv[it + 1]

        # --- dequant helper ---
        # Iu16 = I_i16 + bias
        # u = ref + Iu16*step
        def deq(I_i16, ref):
            return ref + (I_i16.to(torch.float32) + self._bias) * self._step

        # ----- 4 coins (temps interpolé), u/v séparés -----
        # 00
        u0 = deq(self.Iu[it,   iy,  ix],  ru0)
        u1 = deq(self.Iu[it+1, iy,  ix],  ru1)
        v0 = deq(self.Iv[it,   iy,  ix],  rv0)
        v1 = deq(self.Iv[it+1, iy,  ix],  rv1)
        u00 = u0 + wt * (u1 - u0)
        v00 = v0 + wt * (v1 - v0)

        # 01
        u0 = deq(self.Iu[it,   iy,  ix1], ru0)
        u1 = deq(self.Iu[it+1, iy,  ix1], ru1)
        v0 = deq(self.Iv[it,   iy,  ix1], rv0)
        v1 = deq(self.Iv[it+1, iy,  ix1], rv1)
        u01 = u0 + wt * (u1 - u0)
        v01 = v0 + wt * (v1 - v0)

        # 10
        u0 = deq(self.Iu[it,   iy1, ix],  ru0)
        u1 = deq(self.Iu[it+1, iy1, ix],  ru1)
        v0 = deq(self.Iv[it,   iy1, ix],  rv0)
        v1 = deq(self.Iv[it+1, iy1, ix],  rv1)
        u10 = u0 + wt * (u1 - u0)
        v10 = v0 + wt * (v1 - v0)

        # 11
        u0 = deq(self.Iu[it,   iy1, ix1], ru0)
        u1 = deq(self.Iu[it+1, iy1, ix1], ru1)
        v0 = deq(self.Iv[it,   iy1, ix1], rv0)
        v1 = deq(self.Iv[it+1, iy1, ix1], rv1)
        u11 = u0 + wt * (u1 - u0)
        v11 = v0 + wt * (v1 - v0)

        # bilinéaire
        u = u00 * ax * ay + u01 * dx * ay + u10 * ax * dy + u11 * dx * dy
        v = v00 * ax * ay + v01 * dx * ay + v10 * ax * dy + v11 * dx * dy

        # ----- knots + direction FROM -----
        vit_kn = torch.sqrt(u*u + v*v) * KNOTS
        vit_kn = torch.clamp(vit_kn, min=self.clamp_min, max=self.clamp_max)

        ang_math = torch.atan2(v, u) * self._rad2deg
        ang_from = torch.remainder(270.0 - ang_math, 360.0)

        vit_kn = vit_kn.to(self.out_dtype)
        ang_from = ang_from.to(self.out_dtype)

        if return_uv:
            return vit_kn, ang_from, u.to(self.out_dtype), v.to(self.out_dtype)
        return vit_kn, ang_from
    
    def build(self, device=None, sizes=(512, 1024), dtig=0.0):
        device = device or self.device
        with torch.no_grad():
            for n in sizes:
                lat = torch.linspace(-60, 60, n, device=device, dtype=torch.float32)
                lon = torch.linspace(0, 359.75, n, device=device, dtype=torch.float32)
                _ = self(dtig, lat, lon)
                _ = self(dtig, lat, lon)
        torch.cuda.synchronize()
        return self



class GFS025InterpNP_721_I16:
    """
    Version NumPy :
      - Iu, Iv : int16 (n_steps, 721, 1440) contenant (uint16 - bias)
      - refu, refv : float32 (n_steps,)
      - step : float scalaire
    """

    def __init__(
        self,
        Iu_i16: np.ndarray,
        Iv_i16: np.ndarray,
        refu: np.ndarray,
        refv: np.ndarray,
        step: float,
        bias: int = 32768,
        clamp_min_kn: float = 1.0,
        clamp_max_kn: float = 70.0,
        dt_step_h: float = 3.0,
        out_dtype=np.float32,
    ):
        assert Iu_i16.ndim == 3 and Iv_i16.ndim == 3, "Iu/Iv doivent être (n_steps, 721, 1440)"
        assert Iu_i16.shape == Iv_i16.shape, "Iu et Iv doivent avoir la même shape"
        assert Iu_i16.dtype == np.int16 and Iv_i16.dtype == np.int16, "Iu/Iv doivent être np.int16"
        assert refu.ndim == 1 and refv.ndim == 1 and refu.shape[0] == Iu_i16.shape[0], "refu/refv (n_steps,)"

        # self.Iu = Iu_i16
        # self.Iv = Iv_i16
        self.Iu = Iu_i16.astype(np.float32)
        self.Iv = Iv_i16.astype(np.float32)
        
        self.refu = np.asarray(refu, dtype=np.float32)
        self.refv = np.asarray(refv, dtype=np.float32)

        self.n_steps, self.ny, self.nx = Iu_i16.shape
        assert self.ny == 721 and self.nx == 1440, f"attendu (721,1440), reçu ({self.ny},{self.nx})"

        self.dt_step_h = float(dt_step_h)
        self.scale = 4.0
        self.clamp_min = float(clamp_min_kn)
        self.clamp_max = float(clamp_max_kn)
        self.out_dtype = out_dtype

        self._step = np.float32(step)
        self._bias = np.float32(bias)
        self._rad2deg = np.float32(180.0 / math.pi)

    def __call__(self, dtig, lat0, lon0, return_uv=False):
        lat = np.asarray(lat0, dtype=np.float32).reshape(-1)
        lon = np.asarray(lon0, dtype=np.float32).reshape(-1)
        dt  = np.asarray(dtig, dtype=np.float32).reshape(-1)

        # broadcast minimal
        if dt.size == 1 and lat.size > 1:
            dt = np.full_like(lat, dt.item(), dtype=np.float32)
        if lat.size == 1 and dt.size > 1:
            lat = np.full_like(dt, lat.item(), dtype=np.float32)
            lon = np.full_like(dt, lon.item(), dtype=np.float32)

        if not (lat.size == lon.size == dt.size):
            raise ValueError("lat, lon, dt doivent avoir la même taille (ou être scalaires broadcastables)")

        # ----- indices espace -----
        y_f = (90.0 - lat) * self.scale
        x_f = np.mod(lon, 360.0) * self.scale

        iy = np.floor(y_f).astype(np.int64)
        ix = np.floor(x_f).astype(np.int64)

        iy = np.clip(iy, 0, self.ny - 2)
        ix = np.mod(ix, self.nx)

        iy1 = iy + 1
        ix1 = np.mod(ix + 1, self.nx)

        dy = np.clip(y_f - iy.astype(np.float32), 0.0, 1.0)
        dx = np.clip(x_f - ix.astype(np.float32), 0.0, 1.0)

        ax = 1.0 - dx
        ay = 1.0 - dy

        # ----- temps -----
        t = (dt / 3600.0) / self.dt_step_h
        it = np.floor(t).astype(np.int64)
        wt = np.clip(t - it.astype(np.float32), 0.0, 1.0)
        it = np.clip(it, 0, self.n_steps - 2)

        ru0 = self.refu[it]
        ru1 = self.refu[it + 1]
        rv0 = self.refv[it]
        rv1 = self.refv[it + 1]

        # def deq(I_i16, ref):
        #     return ref + (I_i16.astype(np.float32) + self._bias) * self._step
        def deq(I, ref):
            return ref + (I + self._bias) * self._step
        # ----- 4 coins -----
        u0 = deq(self.Iu[it,   iy,  ix ], ru0)
        u1 = deq(self.Iu[it+1, iy,  ix ], ru1)
        v0 = deq(self.Iv[it,   iy,  ix ], rv0)
        v1 = deq(self.Iv[it+1, iy,  ix ], rv1)
        u00 = u0 + wt * (u1 - u0)
        v00 = v0 + wt * (v1 - v0)

        u0 = deq(self.Iu[it,   iy,  ix1], ru0)
        u1 = deq(self.Iu[it+1, iy,  ix1], ru1)
        v0 = deq(self.Iv[it,   iy,  ix1], rv0)
        v1 = deq(self.Iv[it+1, iy,  ix1], rv1)
        u01 = u0 + wt * (u1 - u0)
        v01 = v0 + wt * (v1 - v0)

        u0 = deq(self.Iu[it,   iy1, ix ], ru0)
        u1 = deq(self.Iu[it+1, iy1, ix ], ru1)
        v0 = deq(self.Iv[it,   iy1, ix ], rv0)
        v1 = deq(self.Iv[it+1, iy1, ix ], rv1)
        u10 = u0 + wt * (u1 - u0)
        v10 = v0 + wt * (v1 - v0)

        u0 = deq(self.Iu[it,   iy1, ix1], ru0)
        u1 = deq(self.Iu[it+1, iy1, ix1], ru1)
        v0 = deq(self.Iv[it,   iy1, ix1], rv0)
        v1 = deq(self.Iv[it+1, iy1, ix1], rv1)
        u11 = u0 + wt * (u1 - u0)
        v11 = v0 + wt * (v1 - v0)

        # ----- bilinéaire -----
        u = u00 * ax * ay + u01 * dx * ay + u10 * ax * dy + u11 * dx * dy
        v = v00 * ax * ay + v01 * dx * ay + v10 * ax * dy + v11 * dx * dy

        # ----- vitesse / direction -----
        vit_kn = np.sqrt(u*u + v*v) * KNOTS
        vit_kn = np.clip(vit_kn, self.clamp_min, self.clamp_max)

        ang_math = np.arctan2(v, u) * self._rad2deg
        ang_from = np.mod(270.0 - ang_math, 360.0)

        vit_kn = vit_kn.astype(self.out_dtype, copy=False)
        ang_from = ang_from.astype(self.out_dtype, copy=False)

        if return_uv:
            return vit_kn, ang_from, u.astype(self.out_dtype), v.astype(self.out_dtype)
        return vit_kn, ang_from





def majgrib():
    global basedirGribs025, tigGFS, indice, gfs_interpolateur
    if gfs_interpolateur is None:
         meta = None
    fileMeta = os.path.join(basedirGribs025, "meta.txt")
    meta = load_pack_meta(fileMeta)  # peut être None
    indicemeta= int(meta['indice'])
    tigmeta=int(meta['tig'])
    # print ('fileMeta',fileMeta)
    # print('meta',meta)
    # print ('indice de l\'ancienne version  ',indice )
    # print ('tigGFS de l\'ancienne version',tigGFS) 
    # print ('indicemeta ',indicemeta)
    # print ('tigmeta ',tigmeta)
  
    if indicemeta>indice or tigmeta>tigGFS :
        print("Necessité de recharger le grib et mettre a jour l interpolateur ")
        Iu, Iv, refu, refv, step, bias,tigGFS,indice = load_pack_npz(fileNameC) 
        Iu_i16, Iv_i16, refu_t, refv_t               = to_gpu_i16(Iu, Iv, refu, refv)
        drop_old_interp()
        gfs_interpolateur = GFS025InterpGPU_721_I16(Iu_i16, Iv_i16, refu_t, refv_t, step).build()
        print ('Nouvelles valeurs tigGFS:{}   Nouvel  indice {} '.format(time.strftime(" %d %b %H:%M ",time.gmtime(tigGFS)),indice))       
        # On sauvegarde les nouvelles valeurs (Pas necessaire )
        #save_pack_meta(fileMeta, tigGFS, indice)

                
        Iu_i16_np = (Iu.astype(np.int32) - int(bias)).astype(np.int16)
        Iv_i16_np = (Iv.astype(np.int32) - int(bias)).astype(np.int16)
        
        gfs_interpolateur_np = GFS025InterpNP_721_I16(
            Iu_i16_np,
            Iv_i16_np,
            refu.astype(np.float32, copy=False),
            refv.astype(np.float32, copy=False),
            step=float(step),
            bias=bias,
        )        
          
    else :
        print ('Le grib GFS du {}  Indice {} est à jour ainsi que l interpolateur '.format(time.strftime(" %d %b %H:%M ",time.gmtime(tigGFS)),indice))
    return tigGFS, indice    
        

fileName, fileNameC, tigGFS = gribFileName(basedirGribs025)
Iu, Iv, refu, refv, step, bias, tigGFS, indice = load_pack_npz(fileNameC)

Iu_i16, Iv_i16, refu_t, refv_t = to_gpu_i16(Iu, Iv, refu, refv)
gfs_interpolateur = GFS025InterpGPU_721_I16( Iu_i16, Iv_i16, refu_t, refv_t, step).build()

Iu_i16_np = (Iu.astype(np.int32) - int(bias)).astype(np.int16)
Iv_i16_np = (Iv.astype(np.int32) - int(bias)).astype(np.int16)

gfs_interpolateur_np = GFS025InterpNP_721_I16(
    Iu_i16_np,
    Iv_i16_np,
    refu.astype(np.float32, copy=False),
    refv.astype(np.float32, copy=False),
    step=float(step),
    bias=bias,
)

tigGFS, indice = majgrib()
print (tigGFS)
print(indice)
dtig=18240
y1=36.01804
x1=-5.632657
twsf,twdf=gfs_interpolateur_np(dtig, y1, x1) 

print (twsf,twdf)
y1_t =torch.tensor([y1], device="cuda")
x1_t =torch.tensor([x1], device="cuda")
dtig_t=torch.tensor([dtig])
dt=60

tws_t,twd_t=gfs_interpolateur(dtig_t, y1_t, x1_t) 
print (tws_t,twd_t)
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




def dequantize_uv_int16_fixed(Uq: np.ndarray, Vq: np.ndarray, scale: float) -> np.ndarray:
    """
    Reconstitue float32 (T,Y,X,2) mais pas necessaire si on envoie directement au gpu
    """
    U = Uq.astype(np.float32) * np.float32(scale)
    V = Vq.astype(np.float32) * np.float32(scale)
    return np.stack([U, V], axis=-1)


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
        print('Le grib ECM du {} est à jour '.format(time.strftime(" %d %b %H:%M ",time.gmtime(tigECM))))
        
basedirGribsECM='/home/jp/gribslocaux/ecmwf/'
fileNameECM, date_int, run_hour, tigECM =gribFileNameECM(basedirGribsECM)
print ('filenameECM',fileNameECM)
print ('tigECM : ',time.strftime(" %d %b %H:%M ",time.gmtime(tigECM)))
Uq, Vq, steps_h, scale, tigECM, run_hour, (lat0, lon0, dlat, dlon) = load_ecmwf_npz_i16(fileNameECM)
steps_t = torch.from_numpy(steps_h)  # int32
Uq_t = torch.from_numpy(Uq)  # (T,ny,nx) int16
Vq_t = torch.from_numpy(Vq)
ecm_interpolateur = ECMWFInterpGPU_I16(Uq_t, Vq_t, scale=0.01, steps_h=steps_t, device="cuda").build()



course='774.1'
#username,user_id='Sacron','5f6703a652e859ef03043bb8'
username,user_id='Takron-BSP','59c2706db395b292ed622d84'
isMe='yes'
ari=['WP4']


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

cache_donnees = CpuGpuCache(maxsize=10, ttl=3*3600) 


def charger_donnees(course):
    '''A partir de la reference de la course retourne '''
    '''leginfos,tabexclusions,tabicelimits,carabateau,polairesglobales10to,tabvmg10to '''
     
    # try:                          #recherche dans la table 
    print ('Recherche dans la table')
    print('course',course)
    leginfostr  = rechercheTableLegInfos(course)
    leginfos    = json.loads(leginfostr)
    # except:                      #recherche sur vrouteur 
    #     print('recherche sur vrouteur')
    #     leginfostr  = rechercheleginfos(course)
    #     leginfos    = json.loads(leginfostr)
    
    print(leginfos)
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

    print ('globalspeedratio ' ,globalSpeedRatio )
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
    return leginfos,tabexclusions,tabicelimits,carabateau,polairesglobales10,tabvmg10 

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



def rechercheDonneesCourseUser( user_id,course): 
    '''Recherche les donnees generales boatinfos pour le user_id sur la course '''   
    #print ('user_id,course',user_id,course)
    result    = rechercheTableBoatInfos(user_id,course)  #recherche dans la table locale  
    
    boatinfos = json.loads(result) 
    
    boatinfosbs=boatinfos['bs']

    typebateau          = boatinfosbs['boat']['name']
    user_id             = boatinfosbs['_id']['user_id']
    # print()
    # print('1199 user_id ',user_id )
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
            tabexclusions[name] = verticesGFS025InterpGPU_721
            
    except:   
         tabexclusions={}
   
    return boatinfos,posStart,tabexclusions,positionvr    


def rechercheDonneesPersoCourseUser( user_id,course): 
    '''Recherche les donnees perso waypoints exclusions perso trajets enregistres barrieres  '''
    # print (user_id, course)
    personalinfostr    = rechercheTablePersonalInfos(user_id,course)  #recherche dans la table , ce n est pas une recherche dans le serveur 
    
    # print(personalinfostr)
    personalinfos      = json.loads(personalinfostr)
    ari        = personalinfos['ari']
    
    waypoints=personalinfos['wp']
    print ('dans lable Personalinfos on extrait  personalinfosstr et personalinfos["wp"]',waypoints)
    # print()
    try:
        waypoints  = sorted(personalinfos['wp'].values(), key=lambda x: x[0])     #waypoints sous forme de tableau trié par ordre
    except:
        waypoints=[]
    try:
        exclusions = personalinfos['exclusions']
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

cache_cartes = TTLCache(maxsize=1000, ttl=5 * 24 * 60 * 60)
cache_lock = Lock()

def get_carte(pos, zoom_x, zoom_y):
    lat, lon = int(pos[0]), int(pos[1])
    cle = (lat, lon)

    with cache_lock:
        if cle in cache_cartes:
            return cache_cartes[cle]

    # Si non trouvé (ou expiré), on recharge la carte
    carte = fcarte((lat, lon), zoom_x, zoom_y)

    with cache_lock:
        cache_cartes[cle] = carte

    return carte

cache_cartes = TTLCache(maxsize=1000, ttl=5 * 24 * 60 * 60)
cache_lock = Lock()

tic=time.time()
pos=(46,-3)
carte= get_carte(pos,3,6)
carte_np=segments_to_array(carte)

print('temps chargement de la carte ',time.time()-tic)

#######################################################################################
##   Fonctions de Lissage 
#######################################################################################

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


def rechercheDonneesCourseCache(course):
    cached = cache_donnees.get(course)

    if cached:
        print(f"[CACHE HIT] Course {course}")
        cpu_data, gpu_data = cached

        return DonneesCourse(
            *cpu_data,  # leginfos, tabexclusions, tabicelimits, carabateau, polaires_cpu, vmg_cpu
            polaires_gpu = gpu_data[4],
            vmg_gpu      = gpu_data[5]
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
            leginfos, tabexclusions, tabicelimits, carabateau,polaires_np,vmg_cpu, polaires_gpu, vmg_gpu
            )


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
        self.furler        = 0.8 
   
    def posplus(self, Position,dt,dt_it,option,valeur,dtig0GFS):

        ''' dans cette hypothese de deplacement on ne connait que la position 
            l intervalle de temps
            l option 
            le cap ou la twa suivant l option 
            la stamina initiale 
            le solde de penalite 

        '''
        Positionfin=np.copy(Position)
        
        numero      = Position[0]
        y0          = Position[2]
        x0          = Position[3]
        twa         = Position[5]   # c est la twa pour le deplacement que je veux faire apparaitre dans le tableau 
        cap         = Position[6]
        vitesse     = Position[9]
        voile       = int(Position[10])
        tws0        = Position[12]
        twd         = Position[13]
        stamina_ini = Position[14]
        soldepeno   = max((Position[15]),0)
        dt_ite      = dt_it    - 0.3 * min(soldepeno, dt)        #normalement dt 

        Positionfin[0]=Position[0]+1
        Positionfin[1]=dt
        
        if option==0 : 
            cap = valeur
            twa1= ftwaos(cap,twd) 
            
        if option==1 : 
            twa1= valeur
            cap= fcap(twa,twd) 

        # il n 'y a pas d iteration on passe uniquement d'une position a la suivante 
        # print (' ite {} option {} twa {:4.2f} cap {:4.2f} '.format(numero,option ,twa,cap))
    
            
        cap_rad = np.deg2rad(cap)
        y0_rad  = np.deg2rad(y0) 
        
        y1      = y0   + vitesse * dt_ite / 3600.0 / 60.0 * math.cos(cap_rad)
        x1      = x0   + vitesse * dt_ite / 3600.0 / 60.0 * math.sin(cap_rad) / math.cos(y0_rad)


        Positionfin[2] = y1
        Positionfin[3] = x1

       
        tws,twd =  prevision025dtig(GRGFS, dtig0GFS+dt , y1, x1)                        # Previsions au point de depart  
        
        Positionfin[12]=tws
        Positionfin[13]=twd
        Positionfin[6]=fcap(twa1,twd) 
       
      # on va faire le calcul de voile et de vitesse 
        tws10 = round(tws*10)
        twa10 = abs(round(twa1*10))
        
        vitesseVoileIni       = self.polaires_np[voile, tws10, twa10]                                           # vitesse voileini[voileini,tws10,twa10
        meilleureVitesse      = self.polaires_np[7    ,  tws10, twa10]                                          # vitesse meilleure voile[voileini,tws10,twa10
        meilleureVoile        = self.polaires_np[8,  tws10, twa10]                                              # meilleure voile
        Boost                 = meilleureVitesse/(vitesseVoileIni+0.0001)                                    # Boost 


        #  # # calcul des penalites
        if Boost >1.014 :
            Chgt=1
            voilefinale=meilleureVoile
        else:
            Chgt=0
            voilefinale=voile
                                                                                                                  # on remplit la colonne chgt a la place de voiledef
        Tgybe  = ((twa*twa1)<0)*1                                                                                 # on remplit la colonne 16 Tgybe a la place de boost  (signe de twam1*twa10

        Positionfin[5]=twa1                                                        # on ne peut pas ecrasere la twa tant que la twa anterieure n a pas ete utilisee 
       
        
                                                                                                                 
        Cstamina        = 2 - 0.015 * stamina_ini                                                                       # coefficient de stamina en fonction de la staminaini
        peno_chgt       = Chgt * spline(self.lw, self.hw, self.lwtimer, self.hwtimer, tws) * self.MF * Cstamina
        peno_gybe       = Tgybe * spline(self.lw, self.hw, self.lwtimerGybe, self.hwtimerGybe, tws) * Cstamina
        
        perte_stamina   = calc_perte_stamina_np(tws, Tgybe, Chgt, self.coeffboat)
        recup           = frecupstamina(dt_it, tws)
        # print ('dans recup stamina dt {} tws {} '.format(dt, tws))
        stamina         = min((stamina_ini - perte_stamina + recup),100)
        soldepeno       = max((soldepeno-dt),0)         # la trajectoire a ete calculee en debut d ite avec la peno complete maintenant on retire dt puis on rajoute les nouvelles penos 
        soldepeno1      = max((soldepeno+peno_chgt+peno_gybe ),0)   
        
        Positionfin[9]=meilleureVitesse
        Positionfin[10]=voilefinale
        Positionfin[14] =stamina
        Positionfin[15] = soldepeno1
        Positionfin[16] = dt_ite 
        
        # print (' Dans posplus valeursfinales calculees dt {}  ',dt)

        
        return   Positionfin    




   
def lissage(course,routage_np,t0,posStartVR,posStart):
    tabpoints       = routage_np[:,[2,3]]
    tabtwa          = routage_np[:,5]
    tabdt           = routage_np[:,1]
    tabpointslisses = lissagepoints_dt_twa(tabpoints, tabtwa, tabdt)
    tabpointslisses = lissagepoints_dt_twa(tabpointslisses, tabtwa, tabdt)   # on fait 2 lissages successifs
    caps            = calcul_caps(tabpointslisses)
    
    
    #initialisation de engine pour avoir les differentes valeurs  
    deplacement        = DeplacementEngine2(course)
    
    polairesglobales10 = deplacement.polaires_np
    tabvmg10           = deplacement.tabvmg
    furler             = deplacement.furler
    
    routagelisse            = np.zeros_like(routage_np)
    routagelisse[:,[0,1,4]] = routage_np[:,[0,1,4]]
    routagelisse[:,[2,3]]   = tabpointslisses
    routagelisse[:,6]       = caps
    routagelisse[:,8]       = tabtwa
    dtig0                   = t0-tigGFS

   
    ################################################################################################################""
    #Tws,Twd                 = prevision025dtig(GR, dtig0+routagelisse[:,1] , routagelisse[:,2],routagelisse[:,3])     # calcul de tws et twd 
    Tws_t,Twd_t                 = gfs_interpolateur(dtig0+routagelisse[:,1] , routagelisse[:,2],routagelisse[:,3])
    Tws                         = Tws_t.detach().cpu().numpy() 
    Twd                         = Twd_t.detach().cpu().numpy()
    ################################################################################################################""


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
        Cstamina        = 2 - 0.015 * stamina_ini   
        
        
        # coefficient de stamina en fonction de la staminaini
        # ici on est en numpy
        ######################################################################################################################"
        #peno_chgt       = Chgt * spline(deplacement.lw, deplacement.hw, deplacement.lwtimer, deplacement.hwtimer, twsi) * deplacement.MF * Cstamina
        #peno_gybe       = Tgybe * spline(deplacement.lw, deplacement.hw, deplacement.lwtimerGybe, deplacement.hwtimerGybe, twsi) * Cstamina
        ######################################################################################################################
        peno_chgt       = Chgt * peno_np(deplacement.lwtimer, deplacement.hwtimer, twsi, stamina_ini)*furler
        peno_gybe       = Tgybe * peno_np(deplacement.lwtimerGybe, deplacement.hwtimerGybe, twsi, stamina_ini)
        ######################################################################################################################
        perte_stamina   = calc_perte_stamina_np(twsi, Tgybe, Chgt, deplacement.coeffboat)
        recup           = frecupstamina_np(dt_it, twsi)
        stamina         = min((stamina_ini - perte_stamina + recup),100)


        
        solde_peno       = max((solde_peno-dt_it),0)         # la trajectoire a ete calculee en debut d ite avec la peno complete maintenant on retire dt puis on rajoute les nouvelles penos 
        solde_peno1      = max((solde_peno+peno_chgt+peno_gybe ),0) 
        
         
        if i==0:
            solde_peno1= posStart['penovr']
            stamina    = posStart['stamina']

        routagelisse[i,14]= stamina
        routagelisse[i,15]=solde_peno1
    
    return routagelisse

DonneesCourse1                                                      = rechercheDonneesCourseCache(course)   # c 'est un objet de la classe DonneesCourse
boatinfos,posStartVR,exclusionsVR2  ,positionvr                     = rechercheDonneesCourseUser( user_id,course)                  # self.exclusionsVR2 est un dictionnaire avec les exclusions VR qui ne sert pas 
personalinfos,ari,waypoints,exclusionsperso,barrieres,trajets,tolerancehvmg,retardpeno = rechercheDonneesPersoCourseUser( user_id,course)
print ('personalinfos',personalinfos)     
print ('ari',ari)
print (ari[0])
print (posStartVR)
y0 = posStartVR['y0']
x0 = posStartVR['x0']
t0 = posStartVR['t0']
tws = posStartVR['tws']
twd = posStartVR['twd']
print()
print ('y0 : {:6.4f},x0 : {:6.4f},t0 : {},tws : {:6.2f},twd : {:6.2f}'.format(y0,x0,t0,tws,twd))
# avec interpolation 
dtig=t0-tigGFS
twsf,twdf=gfs_interpolateur_np(dtig, y0, x0) 
print (twsf,twdf)


def calculePosDepart(posStartVR,polairesglobales,carabateau,dt=60,furler=0.8): 

    ''' donne la position de depart au bout de 60 s'''
    ''' Sous la meme forme que posStartVR          '''
    
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

        peno_chgt = Chgt *peno_np(lwtimer, hwtimer, tws, stamina)* furler
#       peno_gybe = Tgybe * *peno_np(lwtimer, hwtimer, Tws, Stamina)   # pas de tackgybe dans les 60 s   
        peno_globale= soldepeno + peno_chgt
        peno_fin    = max(0,peno_globale-dt)
    
#       # Stamina
        perte_stamina = calc_perte_stamina_np(tws, int(Tgybe), Chgt, coeffboat)
        recup = pts_recuperes(dt, tws)
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
        #twsf,twdf= prevision025(GRGFS, t0+dt, y1, x1)

        # print ('dans calcul isodepart twsf {},twdf {} t0+dt {} '.format(twsf,twdf ,time.strftime(" %d %b %H:%M %S ",time.localtime(t0+dt)) ))  
        dtig=   t0+dt-tigGFS
        # dtig=torch.tensor(dtig, device='cuda') 
        lat1 = torch.tensor([y1], device="cuda")
        lon1 = torch.tensor([x1], device="cuda")
        twsf,twdf=gfs_interpolateur(dtig, lat1, lon1)

       #twsf1,twdf1=  prevision025dtig(GRGFS, dtig, y1, x1)       # ancienne interpolation 
        
        
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

donnees     = rechercheDonneesCourseCache(course)
polairesglobales           = donnees.polaires_np
carabateau                 = donnees.carabateau

print(calculePosDepart(posStartVR,polairesglobales,carabateau))


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

def cheminToRoutage(chemin,tabvmg10to):
    routage =torch.zeros((len(chemin),18))
    # reorganisation des colonnes 
    routage[:,[0,1,2,3,5,6,9,10,11,12,13,14,15]]=chemin[:,[0,12,3,4,6,11,13,5,14,9,10,7,8]]
    
    #decalage des colonnes pour que les actions a faire correspondent au temps a venir et non au temps passé 
    decaler_colonnes(routage, colonnes=[0,5,6,9,10,11,14,15])     # on decale les colonnes  0 ordre 5twa 6 cap, les valeurs :  9 vitesse, 10 voile, 11 boost ,  14 stamina  15 peno ne sont pas decales ce sont les valeurs au debut   
    
    #calcul des vmg 
    tws10=torch.round(routage[:,12]*10).int()                                                            # on calcule vmgmin et max a partir de tws et twd 
    routage[:,7] = tabvmg10to[tws10,2]
    routage[:,8] = tabvmg10to[tws10,4]
    
    # on garde les avants dernieres valeurs pour la twa le cap la vitesse et la voile le boost et la stamina 
    routage[-1,[5,6,9,10,11,14]]=routage[-2,[5,6,9,10,11,14]]
    return routage






class RoutageSession:                                                  # version au 8 aout


    def __init__(self, course, user_id, isMe ,ari):
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

        self.typevoile             = self.carabateau['typevoile']
       
        
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
        

    
    def initialiseRoutagePartiel(self, posStartPartiel,ari, indiceroutage , ouverture=200, pas=1, cocheexclusions=1):
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

        # delta longitude court
        dx = dlon_short(x1, x0)          # en degrés, dans [-180, +180]
        dy = y1 - y0

        # longitude "déroulée" de l'arrivée, cohérente avec le départ
        x1u = x0 + dx

        # coordonnées de départ et arrivée sous forme torch
        depto = torch.tensor([y0, wrap360(x0)], device='cuda', dtype=torch.float32)
        arito = torch.tensor([y1, wrap360(x1)], device='cuda', dtype=torch.float32)

            
        
        cap_ar = round((450.0 - math.degrees(math.atan2(dy, dx))) % 360.0)

        range_caps = torch.arange(
        cap_ar - ouverture / 2,
        cap_ar + ouverture / 2,
        pas,
        device='cuda',
        dtype=torch.float32
        ) % 360.0
    
        range_capsR = torch.deg2rad(range_caps)


        range_caps  = torch.arange(cap_ar - ouverture / 2, cap_ar + ouverture / 2, pas) % 360
        range_capsR = range_caps.deg2rad()
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

        # centre routage :
        centreRoutage = torch.tensor( [ (y0 + y1) / 2.0, (x0 + x1) / 2.0 ], device='cuda', dtype=torch.float32 )
        centreRoutageR = torch.deg2rad(centreRoutage)


      

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
            "x0": x0,
            "y1": y1,
            "x1": x1,
            "rwp": rwp,
            "dti": dti,
            "cap_ar": cap_ar,
            "range_caps": range_caps,
            "range_capsR": range_capsR,
            "m_ar": m_ar,
            "centreRoutageR": centreRoutageR,
            "centreRoutage": centreRoutage,
            "rayonRoutage": rayonRoutage,
            "distDepAri": distDepAri,
            "cocheexclusions": cocheexclusions,
            "dtglobal":tabdt,                                  # c est le tableau des dt 
            "retardpeno":retardpeno
          
        }
      
        #print ('dans initialiseroutagepartiel paramroutage',paramRoutage)
        return paramRoutage,iso
    
   




    # ANCIENNE VERSION SANS RETARD PENO
    # *********************************

    def isoplusun(self,iso,tmini,paramRoutage,mode ):
        ''' Donnees a charger propres au routage'''
        ''' polairesglobales10to, lw, hw, lwtimer, hwtimer,MF, coeffboat, rayonRoutage     '''
        ''' necessite comme donnneees externes  GRGFS_gpu ,polairesglobales10to,range_caps,range_capsR,carabateau,lw,....                                   '''
        ''' Les penalites sont affichees sur l iteration et ne sont diminuees du temps de l iteration que sur l iteration suivante '''
       
        MF             = 0.8
        furler         = 0.8
        tolerancehvmg  = self.tolerancehvmg         #tolerance de hvmg
        numisom1       = int(iso[0,0].item())                                                 # numero de l'iso precedent 
        numiso         = numisom1+1    
        t0             = self.t0

       ### Pour Calcul meteo
        dtig0GFS       = t0-tigGFS           # anciennement dtig0GFS=t0-tigGFS
        dtig0ECM       = t0-tigECM 
      

            # if ecart<9*3600 :
            #     iso[:,9],iso[:,10]   = gfs_interpolateur(dtigGFS,iso[:,3],iso[:,4] )    
            # else :
            #     iso[:,9],iso[:,10]   = ecm_interpolateur(dtigECM,iso[:,3],iso[:,4] ) 

        

        
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
        centreRoutage = paramRoutage['centreRoutage']
        rayonRoutage   = paramRoutage['rayonRoutage']
        aritoR         = paramRoutage ["aritoR"]
        deptoR         = paramRoutage ["deptoR"]
        m_ar           = paramRoutage ["m_ar"]
        numisoini      = paramRoutage ["numisoini"]
        indiceroutage  = paramRoutage ["indiceroutage"]
        dtglobal       = paramRoutage ["dtglobal"]
        retardpeno     = paramRoutage ["retardpeno"]


        #print ('retardpeno et tolerance hvmg', retardpeno,self.tolerancehvmg)
        
        latcR, loncR         = centreRoutageR
        latc, lonc           = centreRoutage
        lw                   = self.carabateau["lws"]
        hw                   = self.carabateau["hws"]
        lwtimer              = self.carabateau["lwtimer"]
        hwtimer              = self.carabateau["hwtimer"]
        gybeprolwtimer       = self.carabateau['gybeprolwtimer']
        gybeprohwtimer       = self.carabateau['gybeprohwtimer']
        coeffboat            = self.carabateau['coeffboat']
        polairesglobales10to = self.polairesglobales10to
        tabvmg10to           = self.tabvmg10to
        
        ecartprecedent=iso[0,12].item()                                               # c est l ecart de temps de l iso precedent par rapport a t0
     
        n=len (iso)                                                                   # Longueur de l iso precedent sert a dupliquer 
        p=len(range_caps)                                                             # nombre de caps du rangecaps 
        dernier=iso[-1,1].item()                                                      # dernier point de l iso precedent   
        
        ordre=numiso-numisoini -1                                                     # ordre par rapport a numisoini pour calculer le dt 
       
       
        
        if tmini>3600:                                                                # cas normal a plus de 1h  de l objectif  
            dt=dtglobal[ordre]
       
        else :
            dt = torch.tensor(60.0, dtype=torch.float32, device='cuda:0')
         
        iso[:,0]  = numiso                                                              # on ajoute 1 au numero d iso sur toute la colonne  
        ecart=ecartprecedent+dt     # c est l ecart de temps de l iso  par rapport a t0
  
        # print ('numiso  {} numisoini {}  ecartprecedent {} ecart {}'.format(numiso,numisoini, tmini, ecartprecedent,ecart))     
        # if (numiso!=1):                                                            # Pour le premier isochrone la penalite appliquable au trajet a deja ete amputee du dt  de 60 secondes dans  isodepart  
        
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
        
    
        mask_voile = iso[:,19] != 0
        mask_tack  = iso[:,18] != 0
        PenaliteChgt     = torch.zeros_like(iso[:,19])   
        PenaliteTackGybe = torch.zeros_like(iso[:,19])   
        # formule globalisee
        # Stamina = iso[:,7]
        # Tws     = iso[:,9]    # le vent est en colonne 9 de iso 



        ######################################################################################################################
        #   Calcul des penalites avec calculs version pour tableur 
        ######################################################################################################################
        
           
        PenaliteChgt  [mask_voile]    = peno_torch(lwtimer,hwtimer,iso[mask_voile,9] ,iso[mask_voile,7] )*furler
        PenaliteTackGybe[mask_tack]   = peno_torch(gybeprolwtimer,gybeprohwtimer,iso[mask_tack,9],iso[mask_tack,7])
        # iso[:,8]+=PenaliteChgt+PenaliteTackGybe      # on ajoute a la penalite restant eventuellement 
        iso[:,8].add_(PenaliteChgt)
        iso[:,8].add_(PenaliteTackGybe)
        
        iso[:,7]  +=  - calc_perte_stamina_to(iso[:,9], iso[:,18],iso[:,19], coeffboat)  +   frecupstaminato(dt,iso[:,9])    # la stamina est egale a l ancienne (col4 )-perte (tws,TG,Chgt,coeff,MF)  + frecupstaminato(dt,Tws,pouf=0.8):
   
    
        iso[:,7]  = torch.clamp(iso[:,7],min=0,max=100)                                                                          #*** le max pourra eventuellement etra passse a 110 avec une boisson 
        
        # # # calcul des nouvelles coordonnees

        # print(type(dt))
        iso[:,17]=dt-0.3*torch.clamp(iso[:,8],max=dt )                                                                           # dt remplace boost en colonne 17    
        iso[:,18]=iso[:,3]                                                                                      # on copie la latitude initiale en 18 pour les barrieres 
        iso[:,19]=iso[:,4]                                                                                      # on copie la longitude initiale en 19 pour calculer les barrieres 
        # nouvelles coordonnees 
      

        
              
        iso[:,3]= iso[:,18] + iso[:,17] * iso[:,16] / 3600 / 60 * torch.cos(iso[:,13])                                         #  
        iso[:,4]= iso[:,19] + iso[:,17] * iso[:,16] / 3600 / 60 * torch.sin(iso[:,13])/torch.cos(iso[:,3].deg2rad())                      #    (le cos utilise pour la lat est avec la latitude deja calculee ?)
       
        
        iso[:,11]= iso[:,3].deg2rad()                                # on stocke la lat en rad pour le calcul de distance a l arrivee
        iso[:,12]= iso[:,4].deg2rad() 
        
        # on elimine les points hors cercle                          la distance par rapport au centre est calculee en colonne 15 anciennement colonne twa           latcR, loncR                                                     
        
        iso[:,17]=dist_mn_vec(iso[:,11],iso[:,12],latcR, loncR) 


        
        # print ( 'iso[:,17] ', iso[:,17])
                
        maskDiCentreRoutage   = iso[:,17]<(rayonRoutage*1.15)
        iso    = iso[maskDiCentreRoutage] 
    
        # calcul de distar, du point le plus proche et du temps estime vers l arrivee  
        
        # iso[:,9]=dist(iso[:,11],iso[:,12],aritoR[0], aritoR[1]) *1.852       # Calcul de distar en mN en colonne 9 on utilise les valeurs en radians deja calculees pour les points et pour l arrivee 
        
        iso[:,9]=dist_mn_vec(iso[:,11],iso[:,12],aritoR[0], aritoR[1])

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
     
   ### Calcul meteo    
        if mode=='gfs':
            dtigGFS=dtig0GFS+ecart 
            iso[:,9],iso[:,10]=gfs_interpolateur(dtigGFS,iso[:,3],iso[:,4])
        
        if mode=='mixte':
            dtigGFS=dtig0GFS+ecart 
            dtigECM    = dtig0ECM + ecart
            
            if ecart<9*3600 :
                iso[:,9],iso[:,10]   = gfs_interpolateur(dtigGFS,iso[:,3],iso[:,4] )    
            else :
                iso[:,9],iso[:,10]   = ecm_interpolateur(dtigECM,iso[:,3],iso[:,4] ) 
                
                
        iso[:,11]=torch.rad2deg(iso[:,13])            # on remet le cap initial en 11 a partir du cap en radian que l on a toujours   
        iso[:,14] = iso[:,16]/(iso[:,15]+0.0001)      # on va remettre le boost en colonne 14 a la place de la twa   
        iso[:,13] = iso[:,16]   # copie de la vitesse max en 13 pour la passer a isoglobal
        iso[:,12] = ecart   
        
        # renumerotation 
        iso[:,1]= dernier+torch.arange(len(iso)) +1                         # on va renumeroter les points 
      
        # Copie des points de l isochrone dans isoglobal
        torch.cuda.synchronize()           # Pour attendre que la synchronisation soit complete et essayer d eviter des erreurs 
        # print ('iso.shape ' ,iso.shape)
        premier= int(iso[0,1].item())
        dernier= int(iso[-1,1].item())
        # print ('numiso {} dt {} premier {} dernier {} shape {} isoglobal.shape {} '.format(numiso ,dt,premier,dernier,iso.shape,self.isoglobal.shape))
       
        
        self.isoglobal[premier:dernier+1, :] = iso[:, 0:15]
        
        return iso, tmini,distmini,nptmini
    


session         = RoutageSession(course, user_id,isMe,ari)
def routageGlobal(course,user_id,isMe,ari,y0,x0,t0,tolerancehvmg,optionroutage,mode):                              # version Vrouteur 1/12/2025
    ''' Calcule le routage '''
    ''' Recupère les données et parcoure les differents waypoints '''

    session         = RoutageSession(course, user_id,isMe,ari) 

    waypoints       = session.waypoints
    tabvmg10to      = session.tabvmg10to
    iso             = session.isodepart            # la on est systematiquement sur ma pposition 
    posStartVR      = session.posStartVR
    posStart        = session.posStart
   #x0=lon_to_360(x0)
    typevoile       = session.typevoile

    print()
    print ('(4784) waypoints ',waypoints)
    print()       

    # for wp in waypoints:
    #     wp[3] = lon_to_360(wp[3])
    # print ('4791 x0',x0)    

   
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

        # print ('paramRoutage ',paramRoutage)
        

        impression_tensor15 (iso,titre='\niso de depart correspondant à PositionVR decalee')
       

        tmini            = 10000
        distmini         = 1000
        rwp              = paramRoutage["rwp"]
        if rwp==0:            # probleme des lignes d arrivee a gerer ulterieurement
            rwp=0.2


        numisoini        = paramRoutage['posStartPartiel']['numisoini']
        # print ('rwp',rwp)
        # print ('numisoini',numisoini)
        print ('rwp',rwp) 
        while distmini > rwp:
            iso, tmini, distmini, nptmini = session.isoplusun(iso, tmini,paramRoutage,mode)
            
            print ('tmini {} distmini {}'.format(tmini,distmini))       
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



def calculeroutage(course,user_id,isMe,ari,tolerancehvmg,y0,x0,t0,mode,optionroutage=0,retardpeno=0):     
    print('******************************************************************')
    print ('Simulation de valeurs Vr issues de boatinfos recuperees par l URL')
    username                                       = findUsername(user_id)  
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
    print ('mode                                    ',mode  ) 
    print('******************************************************************')
    print()
   
    # try:
    waypoints,isoglobal,posStartVR,posStart,nptmini,exclusions,tabvmg10to,dico_isochrones = routageGlobal(course, user_id,isMe,ari,y0,x0,t0,tolerancehvmg,optionroutage,mode)    
   
    chemin          = reconstruire_chemin_rapide(isoglobal, nptmini)    
    routage         = cheminToRoutage(chemin,tabvmg10to)             # routage est du torch  
    arrayroutage    = routage.cpu().tolist()                      # passage de torch a tableau python
    routage_np      = np.array(arrayroutage,dtype=np.float64)
    routagelisse    = lissage(course,routage_np,t0,posStartVR,posStart)  

  
    arrayroutage2=[arr.tolist() for arr in routagelisse]
    
    tabtwa       = routagelisse[:,5]
    # twasmooth=smooth(tabtwa)                        # c'est du smooth numpy 
    # twasmooth2=smooth(twasmooth)  
    # routagelisse[:,5]= twasmooth2                   # c'est juste une substitution de facade, il faudrait recalculer le routage   
    dernier = len(arrayroutage2) - 1
    t0=posStart['t0']
    ETA=(t0 + arrayroutage2[dernier][1]) 
    dureeETA = arrayroutage2[dernier][1]
    
    dico              = {'message':'Routage OK','waypoints':waypoints,'arrayroutage':arrayroutage,'arrayroutage2':arrayroutage2,'isochrones':dico_isochrones,'t0routage':t0,'ETA':ETA,'dureeETA':dureeETA}
    eta = t0 + routagelisse[-1,1]

    
    
    # impression_chemin(chemin,t0,titre='chemin')
    # impression_routage (routage_np,t0,titre='Routage_np ')
    impression_routage (routagelisse,t0,titre='Routage ')
    print()
    print('ETA :',time.strftime(" %d %b %H:%M ",time.localtime(ETA)))
    print('Durée',format_duration(dureeETA))   
    
    # except:
    #     waypoints,arrayroutage,arrayroutage2,dico_isochrones,t0,eta=0,0,0,0,0,0  
    #     dico={'message':'Erreur','waypoints':waypoints,'arrayroutage':arrayroutage,'arrayroutage2':arrayroutage2,'isochrones':dico_isochrones,'t0routage':t0}    

  
    return dico

    print (dico['arrayroutage'])
# print (' t0 de base pour impression        {} '.format( time.strftime(" %d %b %H:%M %S",time.localtime(t0))))
# print (' t0 de base pour impression        {} '.format( time.strftime(" %d %b %H:%M %S",dico['t0routage'])))


donnees                                                                           = rechercheDonneesCourseCache(course)                #retourne une classe donnees avec  leginfos, tabexclusions, tabicelimits, carabateau,polaires_np,vmg_cpu, polaires_gpu, vmg_gpu
boatinfos,posStart,tabexclusions,positionvr                                       = rechercheDonneesCourseUser( user_id,course)   
personalinfos,ari,waypoints,exclusions,barrieres,trajets,tolerancehvmg,retardpeno = rechercheDonneesPersoCourseUser( user_id,course)
# print ('personalinfos\n', personalinfos)

print (waypoints)
print('ari',ari)

# tic=time.time()
tolerancehvmg=0
optionroutage=0
y0 =posStart['y0']
x0 =posStart['x0']
t0 =posStart['t0']
print('ari ',ari)
#mode='mixte'

print('ari',ari)
print (tigGFS)
print (tigECM)

mode='gfs'
dico=calculeroutage(course,user_id,isMe,ari,tolerancehvmg,y0,x0,t0,mode)
print ('temps exe routage y compris chargement des donnees',time.time()-tic)


    

