import math
from fonctions2024 import *
import numpy as np
from geopy.distance import great_circle

lat0,lon0=56.6777,-10.8531

lat1,lon1=56.7523,-11.2397

caportho = calcul_cap(lat0,lon0,lat1,lon1)
caploxo  = calcul_cap_loxodromique(lat0,lon0,lat1,lon1)

distance = distance_haversine(lat0,lon0,lat1,lon1)




def haversine(lat1, lon1, lat2, lon2):
    """ Calcule la distance orthodromique entre deux points (en km) """
    R = 6371  # Rayon terrestre moyen en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def mercator_psi(lat):
    """ Fonction Mercator pour transformer la latitude """
    lat = np.radians(lat)
    return np.log(np.tan(np.pi/4 + lat/2))

def rhumb_line_points(lat0, lon0, lat1, lon1, n=100):
    """ Génère des points intermédiaires sur la loxodromie """
    lat0, lon0, lat1, lon1 = map(np.radians, [lat0, lon0, lat1, lon1])
    
    dlon = lon1 - lon0
    dpsi = mercator_psi(lat1) - mercator_psi(lat0)
    cap = np.arctan2(dlon, dpsi)

    lat_points = np.linspace(lat0, lat1, n)
    lon_points = lon0 + (lat_points - lat0) * np.tan(cap)

    return np.degrees(lat_points), np.degrees(lon_points)

def max_corde(lat0, lon0, lat1, lon1, n=100):
    """ Trouve la corde maximale entre orthodromie et loxodromie """
    lat_lox, lon_lox = rhumb_line_points(lat0, lon0, lat1, lon1, n)
    max_dist = 0
    
    for lat, lon in zip(lat_lox, lon_lox):
        dist = great_circle((lat, lon), (lat0, lon0)).km
        max_dist = max(max_dist, dist)
    
    return max_dist




lat0,lon0=56.6777,-10.8531

lat1,lon1=56.7523,-11.2397

corde_max = max_corde(lat0, lon0, lat1, lon1)
print(f"Corde maximale entre l'orthodromie et la loxodromie: {corde_max:.2f} km")







print()
print()
print ('Point A {} {} '.format(lat0,lon0))
print ('Point B {} {} '.format(lat1,lon1))
print ('cap loxodromique ',caploxo)
print ('cap orthodromique ',caportho)
print ('distance   ',distance)
print()