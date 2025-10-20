import time
import os
import math
from datetime import datetime,timezone
import torch
import numpy as np
import json
import sqlite3 as sql
import zlib
import requests
import folium
from scipy.signal import savgol_filter
from scipy.ndimage import label, generate_binary_structure
from urllib.request import urlretrieve
from collections import defaultdict, deque     # necessaire pour la carte re

import psycopg2                                # Necessaire pour la table postgre 
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

basedirnpy='/home/jp/static/npy/'

# from scipy.interpolate import RegularGridInterpolator,interp2d,interpn
# from shapely.geometry import (LineString, MultiLineString, MultiPoint, Point,Polygon)
# from shapely.ops import unary_union
# from shapely.prepared import prep
# from joblib import Parallel,delayed


basedir = os.path.abspath(os.path.dirname("__file__"))
typevoile=['Jib','Spi','Sta','LJ ','C0 ', 'HG ','LG ']
R = 6371.000    # Rayon moyen de la Terre (en km)

filename='/home/jp/staticLocal/cumsum/sequenceglobale.pt'
sequences2 = torch.load(filename,map_location="cuda:0")                     # sequences2 fait 87 Mo imp
cumsumx = torch.cumsum(sequences2, dim=0)      


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
    basedirnpy          = '/home/jp/static/npy/'
    basedirGribs025     = '/home/jp/gribslocaux/gribs025/'
    basedirGribsVR32    = '/home/jp/gribslocaux/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribslocaux/gribsgfs32/'
    staticbd            = '/home/jp/static/bd/basededonnees.db'
    # staticCommandes     = '/home/jp/staticLocal/bd/commandes.db'







if hostname=='portable' :          # sur ordi portable  (3eme ordi)
    basedirnpy          = '/home/jp/static/npy/'
    basedirGribs025     = '/home/jp/gribslocaux/gribs025/'
    basedirGribsVR32    = '/home/jp/gribslocaux/gribsvr32/'
    basedirGribsGfs32   = '/home/jp/gribslocaux/gribsgfs32/'
    staticbd            = '/home/jp/staticLocal/bd/basededonnees.db'
    staticCommandes     = '/home/jp/staticLocal/bd/commandes.db'




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






def ftwato(HDG,TWD):
    '''Twa orientee pour des float ou torch mais pas np '''
    return ((TWD-HDG+540)%360)-180


def fcapto(TWAO, TWD) :
    """
    Retourne le cap (HDG) en fonction de la TWA orientée et de la direction du vent (TWD).
    Version optimisée 100% PyTorch.
    Entrées et sorties : torch.Tensor (en degrés).
    """
    return torch.remainder(360 + TWD - TWAO, 360)




def calcul_cap(lat1, lon1, lat2, lon2):
    ''' calcul le cap en degres des elements lat1 lon1 vers lat2 lon2'''
    '''tous les elements a l entree et a la sortie sont des tenseurs torch y compris la sortie '''
    # Conversion en radians
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlon = lon2 - lon1
    x = torch.sin(dlon) * torch.cos(lat2)
    y = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    
    cap = torch.atan2(x, y)
    cap_deg = torch.rad2deg(cap)
    return (cap_deg + 360) % 360  # pour avoir un cap entre 0 et 360°


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









def calcul_cap1(lat0, lon0, lat1, lon1):
    """
    Calcule le cap en degrés du point (lat0, lon0) vers le point (lat1, lon1).
    """
    # Conversion en radians
    lat0, lon0, lat1, lon1 = map(math.radians, [lat0, lon0, lat1, lon1])
    
    delta_lon = lon1 - lon0
    
    # Formule du cap initial
    y = math.sin(delta_lon) * math.cos(lat1)
    x = math.cos(lat0) * math.sin(lat1) - math.sin(lat0) * math.cos(lat1) * math.cos(delta_lon)
    cap = math.atan2(y, x)
    
    # Conversion en degrés et normalisation entre 0 et 360
    cap = (math.degrees(cap) + 360) % 360
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



def dist(lat1, lon1, lat2, lon2):
    '''definit la distance entre 2 points en km en corrigeant l effet de la latitude '''
    R = 6371.000   # en km
    x = (lon2 - lon1) * torch.cos((lat1 + lat2) / 2)
    y = lat2 - lat1  
    return R * torch.sqrt(x**2 + y**2)




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








def rechercheTablePolaires(polar_id):
# il nous faut l id des polaires
    staticbd='/home/jp/static/bd/basededonnees.db'
    with sql.connect(staticbd) as conn:
        cursor=conn.cursor()
        cursor.execute("select polaires FROM polaires where _id=? " ,(polar_id,))
        result = cursor.fetchone()
        stringpolaires=result[0]
        polairesjson=eval(stringpolaires)

    return polairesjson


def cherchecarabateau(polar_id):
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
    
    polairesjson   = rechercheTablePolaires( polar_id)                       # nouvelle version
    nbvoiles       = len(polairesjson['sail'])

    typevoiles          = []
    for i in range(nbvoiles) :
        typevoiles.append( polairesjson['sail'][i]['name'])

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

    carabateau={ "polar_id":polar_id,'typevoiles':typevoiles, "label":label,"globalSpeedRatio":globalSpeedRatio,"foilSpeedRatio":foilSpeedRatio,"coeffboat":coeffboat,\
            "iceSpeedRatio":iceSpeedRatio,"autoSailChangeTolerance":autoSailChangeTolerance,"badSailTolerance":badSailTolerance,\
            "maxSpeed":maxSpeed,"twaMin":twaMin,"twaMax":twaMax,"twaMerge":twaMerge,"twsMin":twsMin,"twsMax":twsMax,"twsMerge":twsMerge,\
            "hull":hull,'lws': lws ,'hws':hws, "lwtimer":lwtimer,"hwratio": hwratio ,"hwtimer":hwtimer,"lwratio":lwratio,\
            "lwtimer":lwtimer,"tackprolwtimer":tackprolwtimer,"tackprolwratio":tackprolwratio,"tackprohwtimer":tackprohwtimer,\
            "tackprohwratio":tackprohwratio,"gybeprolwtimer":gybeprolwtimer,"gybeprolwratio":gybeprolwratio,"gybeprohwtimer":gybeprohwtimer,"gybeprohwratio":gybeprohwratio}

    return carabateau

def rechercheTablePolaires(polar_id):
# il nous faut l id des polaires
    staticbd='/home/jp/static/bd/basededonnees.db'
    with sql.connect(staticbd) as conn:
        cursor=conn.cursor()
        cursor.execute("select polaires FROM polaires where _id=? " ,(polar_id,))
        result = cursor.fetchone()
        stringpolaires=result[0]
        polairesjson=eval(stringpolaires)

    return polairesjson




def vitangleto(res): 
    ''' transforme le complexe u + j*V    en vitesse et angle '''
    if torch.is_tensor(res)==False:
       res=torch.tensor([res])                       #permet de traiter le cas de valeurs simples 
        
    vitessesto=torch.abs(res)* 1.94384
    vitessesto[vitessesto>70] = 70 
    vitessesto[vitessesto<1]  = 1
    angles=torch.angle(res)
    angles=torch.rad2deg(angles)
    angles = (270 - angles) % 360  

    # if len(res)==1:
    #     vitessesto=vitessesto.item()
    #     angles=angles.item()
    return vitessesto, angles


def prevision025to(GRto,tp, latto, lonto):
    tig      = GRto[0,0,0,0]*100
    # si latto et lonto sont des float  
    if torch.is_tensor(latto)==False:
       latto=torch.tensor([latto])                       #permet de traiter le cas de valeurs simples 
    if torch.is_tensor(lonto)==False:
       lonto=torch.tensor([lonto])                       #permet de traiter le cas de valeurs simples 
    
    latto    = (90-latto)*4
    lonto    = (lonto%360)*4
    dim      = latto.shape
  
    t=((tp-tig)/3600/3)
    
    #indices entiers 
    latito   = latto.int()
    lonito   = lonto.int()
    iitempto = int(t)
   
    # parties fractionnaires 
    dyto       = latto%1
    dxto       = lonto%1
    ditempto   = t%1 
    
    # calcul des valeurs 
    UV000to=torch.complex(GRto[iitempto,latito,lonito,0]                   , GRto[iitempto,latito,lonito,1])
    UV010to=torch.complex(GRto[iitempto,(latito+1)%720,lonito,0]           , GRto[iitempto,(latito+1)%720,lonito,1])
    UV001to=torch.complex(GRto[iitempto,latito,(lonito+1)%1440,0]          , GRto[iitempto,latito,(lonito+1)%1440,1])
    UV011to=torch.complex(GRto[iitempto,(latito+1)%720,(lonito+1)%1440,0]  , GRto[iitempto,(latito+1)%720,(lonito+1)%1440,1])
    UV100to=torch.complex(GRto[iitempto+1,latito,lonito,0]                 , GRto[iitempto+1,latito,lonito,1])
    UV110to=torch.complex(GRto[iitempto+1,(latito+1)%720,lonito,0]         , GRto[iitempto+1,(latito+1)%720,lonito,1])
    UV101to=torch.complex(GRto[iitempto+1,latito,(lonito+1)%1440,0]        , GRto[iitempto+1,latito,(lonito+1)%1440,1])
    UV111to=torch.complex(GRto[iitempto+1,(latito+1)%720,(lonito+1)%1440,0], GRto[iitempto+1,(latito+1)%720,(lonito+1)%1440,1])

    # Interpolation sur le temps 
    UVX00to=UV000to+ditempto*(UV100to-UV000to)     
    UVX10to=UV010to+ditempto*(UV110to-UV010to)
    UVX01to=UV001to+ditempto*(UV101to-UV001to)
    UVX11to=UV011to+ditempto*(UV111to-UV011to)

    #Interpolation bilineaire 
    res=UVX00to+(UVX01to-UVX00to)*dxto +(UVX10to-UVX00to)*dyto  +(UVX11to+UVX00to-UVX10to-UVX01to)*dxto*dyto 
    
    #extraction du module et de l'angle
    vitesses,angles=vitangleto(res)

    return vitesses,angles 




def prevision025todtig(GRto,dtig, latto, lonto):
    #tig      = GRto[0,0,0,0]*100
    # si latto et lonto sont des float  
    if torch.is_tensor(latto)==False:
       latto=torch.tensor([latto])                       #permet de traiter le cas de valeurs simples 
    if torch.is_tensor(lonto)==False:
       lonto=torch.tensor([lonto])                       #permet de traiter le cas de valeurs simples 
    
    latto    = (90-latto)*4
    lonto    = (lonto%360)*4
    dim      = latto.shape
  
    t= dtig/3600/3
    
    #indices entiers 
    latito   = latto.int()
    lonito   = lonto.int()
    iitempto = int(t)
   
    # parties fractionnaires 
    dyto       = latto%1
    dxto       = lonto%1
    ditempto   = t%1 
    
    # calcul des valeurs 
    UV000to=torch.complex(GRto[iitempto,latito,lonito,0]                   , GRto[iitempto,latito,lonito,1])
    UV010to=torch.complex(GRto[iitempto,(latito+1)%720,lonito,0]           , GRto[iitempto,(latito+1)%720,lonito,1])
    UV001to=torch.complex(GRto[iitempto,latito,(lonito+1)%1440,0]          , GRto[iitempto,latito,(lonito+1)%1440,1])
    UV011to=torch.complex(GRto[iitempto,(latito+1)%720,(lonito+1)%1440,0]  , GRto[iitempto,(latito+1)%720,(lonito+1)%1440,1])
    UV100to=torch.complex(GRto[iitempto+1,latito,lonito,0]                 , GRto[iitempto+1,latito,lonito,1])
    UV110to=torch.complex(GRto[iitempto+1,(latito+1)%720,lonito,0]         , GRto[iitempto+1,(latito+1)%720,lonito,1])
    UV101to=torch.complex(GRto[iitempto+1,latito,(lonito+1)%1440,0]        , GRto[iitempto+1,latito,(lonito+1)%1440,1])
    UV111to=torch.complex(GRto[iitempto+1,(latito+1)%720,(lonito+1)%1440,0], GRto[iitempto+1,(latito+1)%720,(lonito+1)%1440,1])

    # Interpolation sur le temps 
    UVX00to=UV000to+ditempto*(UV100to-UV000to)     
    UVX10to=UV010to+ditempto*(UV110to-UV010to)
    UVX01to=UV001to+ditempto*(UV101to-UV001to)
    UVX11to=UV011to+ditempto*(UV111to-UV011to)

    #Interpolation bilineaire 
    res=UVX00to+(UVX01to-UVX00to)*dxto +(UVX10to-UVX00to)*dyto  +(UVX11to+UVX00to-UVX10to-UVX01to)*dxto*dyto 
    
    #extraction du module et de l'angle
    vitesses,angles=vitangleto(res)

    return vitesses,angles 









def calc_perte_stamina(tws: torch.Tensor, TackGybe: torch.Tensor, Chgt: torch.Tensor,
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
    Perte = ((m * tws + p) * (TackGybe + 2 * Chgt * MF)) * coeffboat
    return Perte


def calc_perte_stamina_np(tws: np.ndarray, TackGybe: np.ndarray, Chgt: np.ndarray,
                       coeffboat: float, MF: float = 0.8) -> np.ndarray:
    """
    Calcule la perte énergétique selon les conditions de vent, les changements de manœuvre et les coefficients.
    
    Args:
        tws (ndarray): True Wind Speed.
        TackGybe (ndarray): Nombre de virement ou empannage.
        Chgt (ndarray): Nombre de changements de voile ou autre.
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



def splineto (x1, x2, y1, y2, x):
    t=torch.clamp((x - x1) / (x2 - x1), min=0, max=1)
    tm1 = 1 - t
    res = tm1**3 * y1 + 3 * tm1**2 * t * y1 + 3 * tm1 * t**2 * y2 + t**3 * y2
    res = torch.where(x <= x1, y1, torch.where(x >= x2, y2, res))

    return res


def spline(x1, x2, y1, y2, x):
    t = np.clip((x - x1) / (x2 - x1), 0, 1)
    tm1 = 1 - t
    res = tm1**3 * y1 + 3 * tm1**2 * t * y1 + 3 * tm1 * t**2 * y2 + t**3 * y2
    res = np.where(x <= x1, y1, np.where(x >= x2, y2, res))
    return res

def rechercheTablePersonalInfos2(username, course):
    with sql.connect(staticbd) as conn:
        cursor = conn.cursor()
        recherche = """
        SELECT infos 
        FROM personalinfos2 
        WHERE username=? AND course=? 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        donnees = (username, course)
        cursor.execute(recherche, donnees)
        row = cursor.fetchone()  # Récupérer une seule ligne

        if row:
            return row[0]  # Retourne directement la chaîne JSON

        return None  # Si aucun résultat, renvoie None (ou "{}" si nécessaire)




def reconstruire_chemin(isoglobal, nptmini):
    chemin = []
    point_actuel = nptmini

    # Mapping rapide entre numéro de point et son index dans isoglobal
    index_map = {int(isoglobal[i, 1].item()): i for i in range(isoglobal.size(0))}

    while True:
        i = index_map.get(int(point_actuel))
        if i is None:
            raise ValueError(f"Point {point_actuel} introuvable dans isoglobal")

        chemin.append(isoglobal[i])
        parent = int(isoglobal[i, 2].item())
        if parent == 0:
            # Ajouter le point de départ aussi
            i0 = index_map.get(0)
            if i0 is not None:
                chemin.append(isoglobal[i0])
            break
        point_actuel = parent

    # Inverser pour avoir ordre départ → arrivée
    chemin.reverse()
    return torch.stack(chemin)


def reconstruire_chemin(isoglobal: torch.Tensor, nptmini: int) -> torch.Tensor:
    """
    Reconstruit le chemin dans isoglobal depuis le point nptmini jusqu'au point d'origine (0).
    suppose que:
    - isoglobal[:, 1] contient le numéro du point
    - isoglobal[:, 2] contient le numéro du point parent
    """
    # Création du dictionnaire point -> parent
    points = isoglobal[:, 1].to(torch.int32).tolist()
    parents = isoglobal[:, 2].to(torch.int32).tolist()
    dico_parents = dict(zip(points, parents))

    # Création du dictionnaire numéro de point -> ligne dans isoglobal (index)
    dico_index = {int(pt): i for i, pt in enumerate(points)}

    # Remonter la chaîne des parents
    chemin_indices = []
    pt = nptmini
    while True:
        i = dico_index.get(pt)
        if i is None:
            raise ValueError(f"Point {pt} introuvable dans isoglobal")
        chemin_indices.append(i)
        pt_parent = dico_parents.get(pt, 0)
        if pt_parent == 0:
            i0 = dico_index.get(0)
            if i0 is not None and i0 not in chemin_indices:
                chemin_indices.append(i0)
            break
        pt = pt_parent

    # Récupération des lignes correspondantes dans l'ordre départ → arrivée
    chemin_indices.reverse()
    chemin = isoglobal[chemin_indices]
    return chemin 












def frecupstaminato(dt,Tws,pouf=0.8):
    ''' Calcul exact vérifié avec ITYC '''
    ''' tws en noeuds, dt en s '''
    ''' Tws peut etre un np.array'''
    TempsPourUnPoint = splineto(0, 30, 300, 900, Tws)*pouf
    ptsRecuperes = dt / TempsPourUnPoint
    return ptsRecuperes


def frecupstamina(dt,Tws,pouf=0.8):
    ''' Calcul exact vérifié avec ITYC '''
    ''' tws en noeuds, dt en s '''
    ''' Tws peut etre un np.array'''
    TempsPourUnPoint = spline(0, 30, 300, 900, Tws)*pouf
    ptsRecuperes = dt / TempsPourUnPoint
    return ptsRecuperes


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



# def prepare_segments(polygons):
#     segments = []
#     for poly in polygons.values():
#         if not torch.all(poly[0] == poly[-1]):
#             poly = torch.cat([poly, poly[0:1]], dim=0)
#         segs = torch.stack([poly[:-1], poly[1:]], dim=1).reshape(-1, 4)  # [x1, y1, x2, y2]
#         segments.append(segs)
#     return torch.cat(segments, dim=0)





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


def tempsmnm1b(timestamp):
    '''Fonction pour pouvoir avoir le temps en float32 plutot que float64'''
    return(timestamp // 60) * 60 -1.7e9 

# def ajouter_croix(m, lat, lon, taille=0.01, couleur='red'):
#     # Petite croix en forme de deux segments
#     m.add_child(folium.PolyLine(locations=[
#         [lat - taille, lon],
#         [lat + taille, lon]
#     ], color=couleur, weight=2))

#     m.add_child(folium.PolyLine(locations=[
#         [lat, lon - taille],
#         [lat, lon + taille]
#     ], color=couleur, weight=2))


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


def segmenter_par_voile_apres_decalage(tenseur):
    '''fonction faite pour chemin''' 
    ''' les coordonnnees sont en 3 et 4 et la voile en 5 '''
    from collections import defaultdict
    segments = defaultdict(list)
    n = tenseur.size(0)

    if n < 2:
        return segments

    courant_voile = int(tenseur[0, 5].item())
    segment_courant = [
        (float(tenseur[0, 3]), float(tenseur[0, 4])),
        (float(tenseur[1, 3]), float(tenseur[1, 4]))
    ]

    for i in range(1, n - 1):
        voile = int(tenseur[i, 5].item())
        point = (float(tenseur[i + 1, 3]), float(tenseur[i + 1, 4]))

        if voile != courant_voile:
            segments[courant_voile].append(segment_courant)
            segment_courant = [segment_courant[-1], point]
            courant_voile = voile
        else:
            segment_courant.append(point)

    segments[courant_voile].append(segment_courant)
    return segments








def decaler_colonnes(tenseur, colonnes=[5, 6, 7, 8]):
    for col in colonnes:
        tenseur[:-1, col] = tenseur[1:, col]  # décalage vers le haut
        tenseur[-1, col] = 0                  # dernière ligne à zéro
    return tenseur


def lissagepoints_torch(tabpoints, tabtwa, window_length=5, polyorder=2):
    '''
    Lissage des points sans lisser aux changements d'amure (TWA change de signe).

    Paramètres :
    - tabpoints : torch tensor de forme (N, 2), points à lisser (ex: positions X/Y)
    - tabtwa : torch tensor de forme (N,), angles TWA associés (positif ou négatif)
    - window_length : taille de la fenêtre pour savgol_filter (doit être impair)
    - polyorder : ordre du polynôme pour savgol_filter
    '''
    tabres = tabpoints.clone()
    b = pos_virt_torch_vectorized(tabtwa)

    for i in range(b.shape[0] - 1):
        start = b[i].item()
        end = b[i + 1].item()
        segment = tabpoints[start:end]

        if segment.shape[0] > window_length + 1:
            # Suffisamment de points pour lisser
            y = segment[:, 0].cpu().numpy()
            x = segment[:, 1].cpu().numpy()
            Y = savgol_filter(y, window_length, polyorder, mode='nearest')
            X = savgol_filter(x, window_length, polyorder, mode='nearest')
            Y = torch.from_numpy(Y).to(tabpoints.device)
            X = torch.from_numpy(X).to(tabpoints.device)
        else:
            # Trop petit : pas de lissage
            Y = segment[:, 0]
            X = segment[:, 1]

        # Mettre à jour sans toucher les extrémités
        if end - start > 2:
            tabres[start+1:end-1, 0] = Y[1:-1]
            tabres[start+1:end-1, 1] = X[1:-1]

    return tabres

def pos_virt_torch_vectorized(tab):
    '''Retourne le nombre de valeurs pour chaque amure (vectorisé pour GPU).'''
    # Ajouter un zéro à la fin comme dans ta version initiale
    tab2 = torch.cat((tab, torch.tensor([0], device=tab.device, dtype=tab.dtype)))
    
    # Calculer où il y a un changement de signe (virement de bord)
    changement = (tab2[:-1] * tab2[1:]) <= 0

    # On force un 'changement' sur le dernier élément pour conserver la logique
    changement[-1] = True

    # Indices où il y a un changement
    indices_changement = torch.nonzero(changement, as_tuple=False).flatten()

    # Maintenant, pour chaque "tack", calculer combien de valeurs il y avait
    longueurs = indices_changement.diff()
    longueurs = torch.cat((indices_changement[:1] + 1, longueurs))  # Ajout du premier segment

    # Faire la somme cumulée
    b = torch.cumsum(longueurs, dim=0)
    
    # Ajouter un 0 au début
    b = torch.cat((torch.tensor([0], device=tab.device, dtype=b.dtype), b))

    return b




def corriger_pic_1d(twa: torch.Tensor,  seuil: float = 1.0) -> torch.Tensor:
    """
    Corrige les 'pics' isolés dans un vecteur 1D (TWA) si la valeur centrale
    diffère des deux valeurs adjacentes (identiques) d'au plus ±seuil.

    Exemple : [50, 52, 50] avec seuil=2 → [50, 50, 50]

    Args:
        twa (torch.Tensor): Vecteur 1D à corriger.
        seuil (float): Tolérance maximale de différence pour correction (par défaut 1.0).

    Returns:
        torch.Tensor: Vecteur corrigé (copie).
    """
    if twa.ndim != 1 or twa.size(0) < 3:
        return twa.clone()

    twa_corrigé = twa.clone()

    for i in range(1, len(twa) - 1):
        prev = twa[i - 1]
        curr = twa[i]
        next_ = twa[i + 1]

        if prev == next_ and torch.abs(curr - prev) <= seuil:
            twa_corrigé[i] = prev

    return twa_corrigé

##################################################################################################################"
##############     Chargement des cartes    ######################################################################
##################################################################################################################




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


def assemble_masques(lat_min, lat_max, lon_min, lon_max):
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




def fcarte(pos,deltalat=3,deltalon=4):

    ''' donne la polyligne correspondant a la carte autour du point '''
    ''' necessite les fonctions assemble_masques contourterre3 et chainesegments'''
    lat=pos[0]
    lon=pos[1]
    latmin=int(lat-deltalat)
    latmax=int(lat+deltalat)    
    lonmin=int(lon-deltalon)
    lonmax=int(lon+deltalon)
    masque_global, lat0, lon0 = assemble_masques(latmin,latmax,lonmin,lonmax)
    resolutionlat=abs(latmax-latmin)
    resolutionlon=abs(lonmax-lonmin) 
    segments = contour_terre3(masque_global, lat0, lon0,resolutionlat,resolutionlon)
    polylignes = chaine_segments(segments)
    return polylignes

def segments_to_coords_folium_numpy(segments, lat0, lon0, pas):
    """
    Convertit segments numpy (x1,y1,x2,y2) en liste pour folium : [[[lat1, lon1], [lat2, lon2]], ...]
    """
    x1 = segments[:, 0]
    y1 = segments[:, 1]
    x2 = segments[:, 2]
    y2 = segments[:, 3]

    lat1 = lat0 - y1 * pas
    lon1 = lon0 + x1 * pas
    lat2 = lat0 - y2 * pas
    lon2 = lon0 + x2 * pas

    coords = [ [[lat1[i], lon1[i]], [lat2[i], lon2[i]]] for i in range(len(segments)) ]
    return coords


def detecter_segments_numpy(masque):
    nrows, ncols = masque.shape

    # Discontinuités verticales (entre lignes) → segments horizontaux
    diff_vert = masque[:-1, :] != masque[1:, :]  # shape (nrows-1, ncols)
    i_vert, j_vert = np.where(diff_vert)
    # (x1,y1,x2,y2) pour segments horizontaux
    seg_h = np.stack([j_vert, i_vert + 1, j_vert + 1, i_vert + 1], axis=1)

    # Discontinuités horizontales (entre colonnes) → segments verticaux
    diff_horz = masque[:, :-1] != masque[:, 1:]  # shape (nrows, ncols-1)
    i_horz, j_horz = np.where(diff_horz)
    # (x1,y1,x2,y2) pour segments verticaux
    seg_v = np.stack([j_horz + 1, i_horz, j_horz + 1, i_horz + 1], axis=1)

    return seg_h, seg_v

def regrouper_segments_alignes_numpy(segments, axe_fixe='y'):
    """
    Fusionne segments colinéaires (x1,y1,x2,y2) qui se suivent.
    axe_fixe = 'y' pour segments horizontaux (y fixe), 'x' pour verticaux (x fixe)
    Retourne tableau fusionné de mêmes segments.
    """

    if len(segments) == 0:
        return segments

    # Trier segments pour fusion (par axe fixe, puis coord variable)
    if axe_fixe == 'y':
        # segments horizontaux : trier par y1, puis x1
        idx_sort = np.lexsort((segments[:, 0], segments[:, 1]))
    else:
        # segments verticaux : trier par x1, puis y1
        idx_sort = np.lexsort((segments[:, 1], segments[:, 0]))

    segments = segments[idx_sort]

    # Liste pour segments fusionnés
    fusion = []
    debut = segments[0].copy()

    for seg in segments[1:]:
        if axe_fixe == 'y':
            # même y et la fin du précédent est la début du suivant en x
            if (seg[1] == debut[1]) and (seg[0] == debut[2]):
                # étendre le segment fusionné
                debut[2] = seg[2]
                debut[3] = seg[3]
            else:
                fusion.append(debut)
                debut = seg.copy()
        else:
            # même x et fin précédent == début suivant en y
            if (seg[0] == debut[0]) and (seg[1] == debut[3]):
                debut[2] = seg[2]
                debut[3] = seg[3]
            else:
                fusion.append(debut)
                debut = seg.copy()
    fusion.append(debut)

    return np.array(fusion)



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

def construire_carte(masque, lat0, lon0, pas):
    seg_h, seg_v = detecter_segments_numpy(masque)

    # Fusion segments horizontaux (axe y fixe)
    seg_h_fus = regrouper_segments_alignes_numpy(seg_h, 'y')
    # Fusion segments verticaux (axe x fixe)
    seg_v_fus = regrouper_segments_alignes_numpy(seg_v, 'x')

    # Conversion en coords geographiques
    coords_h = segments_to_coords_folium_numpy(seg_h_fus, lat0, lon0, pas)
    coords_v = segments_to_coords_folium_numpy(seg_v_fus, lat0, lon0, pas)

    # Fusion finale
    return coords_h + coords_v



def fcarte2(lat,lon):
    ''' construit la carte -5 +5 pour lat et lon '''
    latmin=5 * math.floor(lat/ 5)
    latmax=latmin+10
    lonmin=5 * math.floor(lon / 5)
    lonmax=lonmin+10
    print('latmin',latmin)
    print(lonmin)
    pas=1/730
    masquecomplet, lat0, lon0 = assemble_masques(latmin,latmax,lonmin,lonmax)
    masque=retirer_lacs(masquecomplet)
    return construire_carte(masque, lat0, lon0, pas)


def fcarte3(lat,lon):
    '''Charge la carte des offsets directement  
     transforme en coordonnees et renvoie le tableau de coordonnees'''
    

    lat0=int(10*(lat//10) +10)
    lon0=int(10*(lon//10))
    print (lat0,lon0)
    filename='maps2/carteoffset_'+str(lat0)+'_'+str(lon0)+'.npy'
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







##################################################################################################################
##############     Decoupage des isochrones sur les terres   #####################################################
##################################################################################################################


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

##################################################################################################################
##############     Impressions              ######################################################################
##################################################################################################################







def impressionpointdep(tab) :
    print (' 0:n    1:date       2:dt    3:opt 4:valeur   5:lat   6:lon     7:voile  8:twa  9:cap     10:speed  11:stam  12:soldepe  13:twd   14:tws      15:Vauto  16:boost   ')
    print ('____________________________________________________________________________________________________________________________________________________________')
 
    for i in range  (tab.shape[0]):
        print(   '{:4.0f} {}{:6.0f}   {:4.0f}   {:6.2f}   {:8.4f}  {:6.4f}  {:6.0f}   {:8.2f}  {:6.2f}    {:8.3f} {:8.2f} {:8.2f}   {:8.2f}    {:6.2f}  {:8.1f}   {:8.3f}        '.\
            format(tab[ i,0].item(),time.strftime(" %d %b %H:%M ",time.localtime(tab[ i,1].item())), tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),tab[ i,5].item(),\
         tab[ i,6].item(), tab[ i,7].item(),tab[ i,8].item()\
        ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),tab[ i,13].item(),tab[ i,14].item(),tab[ i,15].item(),tab[ i,16].item() ))


def impression_routage (tab,t0,titre='Routage '):
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print (' 0:niso  1heure     2:lat   3:lon        4:date    5:twa     6:cap    7:vmgmin   8:vmgmax   9:speed   10:voile 11:boost  12:tws 13:twd    14:stamina  15:peno      ')
    print ('____________________________________________________________________________________________________________________________________________________________')
    print ( ) 
    i=0
    print(   'Etat VR: {:4.0f}     {:8.5f}  {:8.5f} {} {:7.2f}   {:8.2f}  {:6.2f}  {:8.2f}   {:8.2f}  {:6.1f}    {:8.3f} {:8.2f} {:8.2f}   {:8.1f}    {:6.2f}     '.\
            format( 0, tab[ 0,2].item(),tab[ 0,3].item(),time.strftime(" %d %b %H:%M ",time.localtime(t0)),tab[ i,5].item(),tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item()\
        ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),0,tab[ i,14].item() ,tab[ i,15].item()))
    print ('____________________________________________________________________________________________________________________________________________________________')
          
    for i in range (len(tab)):
        print('{:4.0f} {:8.0f}     {:8.5f}  {:8.5f} {} {:7.0f}   {:8.0f}  {:6.2f}  {:8.2f}   {:8.2f}  {:6.1f} || {:8.3f} {:8.2f} {:8.2f}    {:8.1f}   {:6.2f}  '.\
          format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),time.strftime(" %d %b %H:%M ",time.localtime(tab[ i,4].item()+t0)),tab[ i,5].item(),tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item()\
        ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),tab[ i,13].item(),tab[ i,14].item() ,tab[ i,15].item()  ))




def impression_tensor9(tab,titre='Points'):
    print (titre)
    print ('  i     lat     lon    voile   twam1   stam   Speno  ptmere   tws   tws10   twd  cap')
    print ('_______________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f} {:8.4f} {:8.4f} {:6.2f} {:6.2f} {:6.2f}  {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.\
          format(i, tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),tab[ i,5].item(), tab[ i,6].item(), tab[ i,7].item(), tab[ i,8].item(), tab[ i,9].item(), tab[ i,10].item()))

def impression_tensor6 (tab,titre='Calculs'):
    print (titre)
    print ('  i     twa10     Vvoileini    Vvoilemax   Bestvoile        Boost      autre')
    print ('_______________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f}    {:8.0f}      {:4.2f}         {:4.2f}      {:6.1f}       {:10.3f}         {:6.2f}'.\
          format(i, tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),0))

def impression_tensor15 (tab,titre='Calculs'):
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print (' 0:niso   1:i   2:PtMe   3:lat    4:lon      5:Voi  6:twam1 7:stam  8:Speno  9:tws  10:twd   11:tws10  12:Cap   13:CapR  14:twa   15:VVini 16:VVmax 17:BestV 18:Boost  19:voiledef    ')
    print ('_______________________________________________________________________________________________________________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f} {:8.0f} {:6.0f}   {:8.5f} {:8.5f}    {}   {:8.2f}  {:6.2f}  {:6.2f} {:6.2f} {:6.1f} {:8.2f} {:8.2f} {:8.3f}    {:8.1f}    {:6.2f}  {:6.2f}  {:6.2f}  {:10.3f}  {:6.2f}'.\
          format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),typevoile[int(tab[ i,5].item())],tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item()\
        ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),tab[ i,13].item(),tab[ i,14].item() ,tab[ i,15].item(),tab[ i,16].item(),tab[ i,17].item(),tab[ i,18].item(),tab[ i,19].item()))

def impression_tensor15bis (tab,titre='Calculs1'):
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print (' 0:niso  1:i    2:PtMe  3:latf   4:lonf  5:Voilf  6:twam1  7:stamf  8:Spenof 9:distar  10:ordoar   11:latR  12:lonR  13:CapR  14:twa    15:VVini  16:VVmax  17:dt   18:TGybe  19:Chgt    ')
    print ('____________________________________________________________________________________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f} {:8.0f} {:6.0f}  {:8.5f} {:8.5f} {}   {:8.2f}  {:6.2f}  {:8.2f}   {:8.2f}  {:6.4f} || {:8.2f} {:8.2f} {:8.3f}    {:8.1f}    {:6.2f}  {:6.2f}  {:6.2f}  {:10.3f}  {:6.2f}'.\
          format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),typevoile[int(tab[ i,5].item())],tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item()\
        ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),tab[ i,13].item(),tab[ i,14].item() ,tab[ i,15].item(),tab[ i,16].item(),tab[ i,17].item(),tab[ i,18].item(),tab[ i,19].item()))

def impression_tensor15ter (tab,titre='Calculs distar ordoar '):
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print (' 0:niso  1:i    2:PtMe  3:latf   4:lonf  5:Voilf  6:twam1  7:stamf  8:Spenof 9:distar  10:ordoar   11:ecart 12:Mer/Terre  13:CapR  14:twa    15:VVini  16:VVmax  17:dt   18:TGybe  19:Chgt    ')
    print ('____________________________________________________________________________________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f} {:8.0f} {:6.0f}  {:8.5f} {:8.5f} {}   {:8.2f}  {:6.2f}  {:8.2f}   {:8.2f}  {:6.1f} || {:8.2f} {:8.0f} {:8.3f}    {:8.1f}    {:6.2f}  {:6.2f}  {:6.2f}  {:10.3f}  {:6.2f}'.\
          format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),typevoile[int(tab[ i,5].item())],tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item()\
        ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),tab[ i,13].item(),tab[ i,14].item() ,tab[ i,15].item(),tab[ i,16].item(),tab[ i,17].item(),tab[ i,18].item(),tab[ i,19].item()))


def impression_tensor15qua (tab,titre='Calculs'):
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print (' 0:niso  1:i    2:latf   3:lonf 4:Nvoile  5:twa   6:stam    7:Speno    8:PtMe     9:tws  10:tws10   11:latrad    12:lonrad   13:CapR  14:twa10 15:VVini 16:VVmax 17:dt 18:TGybe  19:Chgt    ')
    print ('____________________________________________________________________________________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f}  {:4.0f} {:8.4f} {:8.4f} {:6.2f}   {:8.2f}  {:6.2f}  {:8.2f} {:8.2f} || {:8.2f} {:6.1f} {:8.2f} {:8.2f} {:8.3f}    {:8.1f}    {:6.2f}  {:6.2f}  {:6.2f}  {:10.3f}  {:6.2f}'.\
          format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),tab[ i,5].item(),tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item()\
        ,tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item(),tab[ i,12].item(),tab[ i,13].item(),tab[ i,14].item() ,tab[ i,15].item(),tab[ i,16].item(),tab[ i,17].item(),tab[ i,18].item(),tab[ i,19].item()      ))

def impression_tensor9 (tab,titre='Points initiaux'):
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print ('0:niso  1:i    2:PtMe   3:lat      4:lon  5:Nvoile  6:twam1  7:stam  8:Speno    ')
    print ('________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f} {:8.0f} {:6.0f}  {:8.5f} {:8.5f} {:6.1f}   {:8.2f}  {:6.2f}  {:7.2f}'.\
          format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),tab[ i,5].item(),tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item() ))
        
def impression_tensor10 (tab,titre='Chemin'):
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print ('0:niso  1:i    2:PtMe   3:lat      4:lon  5:Nvoile  6:twam1  7:stam  8:Speno   9: distar ')
    print ('________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f} {:8.0f} {:6.0f}  {:8.5f} {:8.5f} {:6.1f}   {:8.2f}  {:6.2f}  {:7.2f} {:7.2f}  '.\
          format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),tab[ i,5].item(),tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item(),tab[ i,9].item() ))

def impression_tensor12 (tab,titre='Points initiaux'):
    print (titre , ' ' , tab.dtype, ' Device ', tab.device , ' ' ,tab.shape)
    print('________________________________________________________________')
    print ('0:niso  1:i    2:PtMe   3:lat      4:lon  5:Nvoile  6:twam1  7:stam  8:Speno  9:Tws 10:Twd  11:Tws10  ')
    print ('________________________________________________________________________________')
    for i in range (len(tab)):
        print('{:4.0f} {:8.0f} {:6.0f}  {:8.5f} {:8.5f} {:6.1f}   {:8.2f}  {:6.2f}  {:7.2f}  {:6.2f}  {:6.2f} {:6.2f}'.\
              format( tab[i,0].item(),tab[ i,1].item(),tab[ i,2].item(),tab[ i,3].item(),tab[ i,4].item(),tab[ i,5].item(),tab[ i,6].item(),tab[ i,7].item(),tab[ i,8].item(),tab[ i,9].item(),tab[ i,10].item(),tab[ i,11].item() ))


# positionvr=torch.tensor([0,t0,dt,option,valeur,y0,x0,voile,twa,cap,speed,stamina,soldepenovr,twd,tws,voileAuto,boost],dtype=torch.float64)

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


def segmenter_par_voile_routage(tenseur):
    '''fonction faite pour chemin''' 
    ''' les coordonnnees sont en 3 et 4 et la voile en 5 '''

    '''pour routage les coordonnnees sont en 2 et 3 et la voile en 10'''
    from collections import defaultdict
    segments = defaultdict(list)
    n = tenseur.size(0)

    if n < 2:
        return segments

    courant_voile = int(tenseur[0, 10].item())
    segment_courant = [
        (float(tenseur[0, 2]), float(tenseur[0, 3])),
        (float(tenseur[1, 2]), float(tenseur[1, 3]))
    ]

    for i in range(1, n - 1):
        voile = int(tenseur[i, 10].item())
        point = (float(tenseur[i + 1, 2]), float(tenseur[i + 1, 3]))

        if voile != courant_voile:
            segments[courant_voile].append(segment_courant)
            segment_courant = [segment_courant[-1], point]
            courant_voile = voile
        else:
            segment_courant.append(point)

    segments[courant_voile].append(segment_courant)
    return segments


# def segmenter_par_voile_routage_np(array):
#     '''Segmenter un trajet en segments continus avec la même voile.
#     Les coordonnées sont en indices 5 (lat) et 6 (lon), la voile est à l'indice 7.
#     '''
#     segments = defaultdict(list)
#     n = array.shape[0]

#     if n < 2:
#         return segments

#     courant_voile = int(array[0, 7])
#     segment_courant = [
#         (float(array[0, 5]), float(array[0, 6])),
#         (float(array[1, 5]), float(array[1, 6]))
#     ]

#     for i in range(1, n - 1):
#         voile = int(array[i, 7])
#         point = (float(array[i + 1, 5]), float(array[i + 1, 6]))

#         if voile != courant_voile:
#             segments[courant_voile].append(segment_courant)
#             segment_courant = [segment_courant[-1], point]
#             courant_voile = voile
#         else:
#             segment_courant.append(point)

#     segments[courant_voile].append(segment_courant)
#     return segments
def segmenter_par_voile_routage_np(array):
    '''Segmenter un trajet en segments continus avec la même voile.
    Les coordonnées sont en indices 5 (lat) et 6 (lon), la voile est à l'indice 7.
    '''
    segments = defaultdict(list)
    n = array.shape[0]

    if n < 2:
        return segments

    courant_voile = int(array[0, 7])
    segment_courant = [(float(array[0, 5]), float(array[0, 6]))]

    for i in range(1, n):
        voile = int(array[i, 7])
        point = (float(array[i, 5]), float(array[i, 6]))

        if voile != courant_voile:
            # Fin du segment précédent
            segment_courant.append((float(array[i - 1, 5]), float(array[i - 1, 6])))
            segments[courant_voile].append(segment_courant)

            # Nouveau segment avec le point précédent + nouveau point
            segment_courant = [(float(array[i - 1, 5]), float(array[i - 1, 6])), point]
            courant_voile = voile
        else:
            segment_courant.append(point)

    # Ajoute le dernier segment
    segments[courant_voile].append(segment_courant)
    return segments



#on transforme la carte en un tableau numpy
def segments_to_array(carte):
    return np.array([[seg[0][0], seg[0][1], seg[1][0], seg[1][1]] for seg in carte], dtype=np.float32)



def segment_bounding_box_filter(carte_np, y1, x1, y2, x2):
    ymin_seg, ymax_seg = sorted([y1, y2])
    xmin_seg, xmax_seg = sorted([x1, x2])

    # Bounding boxes de tous les segments de la carte
    ymins = np.minimum(carte_np[:, 0], carte_np[:, 2])
    ymaxs = np.maximum(carte_np[:, 0], carte_np[:, 2])
    xmins = np.minimum(carte_np[:, 1], carte_np[:, 3])
    xmaxs = np.maximum(carte_np[:, 1], carte_np[:, 3])

    # Test rapide d'intersection des bounding boxes
    mask = (
        (ymaxs >= ymin_seg) & (ymins <= ymax_seg) &
        (xmaxs >= xmin_seg) & (xmins <= xmax_seg)
    )

    return carte_np[mask]  # Segments candidats


def orientation(p, q, r):
    # p et q : (2,)
    # r : (N, 2)
    return (q[1] - p[1]) * (r[:, 0] - q[0]) - (q[0] - p[0]) * (r[:, 1] - q[1])

def orientation_batch(p, q, r):
    # p et q : (2,)
    # r : (N, 2)
    return (q[1] - p[1]) * (r[:, 0] - q[0]) - (q[0] - p[0]) * (r[:, 1] - q[1])

def orientation_pairs(p, q, r):
    # p, q, r : (N, 2)
    return (q[:, 1] - p[:, 1]) * (r[:, 0] - q[:, 0]) - (q[:, 0] - p[:, 0]) * (r[:, 1] - q[:, 1])



def segments_intersect(p1, p2, q1s, q2s):
    # p1, p2: (2,), q1s, q2s: (N, 2)
    N = len(q1s)
    p1s = np.repeat(p1[None, :], N, axis=0)  # (N, 2)
    p2s = np.repeat(p2[None, :], N, axis=0)

    o1 = orientation_batch(p1, p2, q1s)
    o2 = orientation_batch(p1, p2, q2s)
    o3 = orientation_pairs(q1s, q2s, p1s)
    o4 = orientation_pairs(q1s, q2s, p2s)

    return (o1 * o2 < 0) & (o3 * o4 < 0)



def segment_coupe_carte(carte_np, y1, x1, y2, x2):
    candidats = segment_bounding_box_filter(carte_np, y1, x1, y2, x2)
    if len(candidats) == 0:
        return False  # Rien à tester

    p1 = np.array([y1, x1])
    p2 = np.array([y2, x2])
    q1s = candidats[:, 0:2]
    q2s = candidats[:, 2:4]

    return np.any(segments_intersect(p1, p2, q1s, q2s))


