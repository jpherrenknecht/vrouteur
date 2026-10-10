# def get_carte3 (lat, lon):
#     #la cle est un multiple de 10
#     cle = (lat, lon)
#     with cache_lock:
#         if cle in cache_cartes:
#             return cache_cartes[cle]

#     # Calcul ou chargement de la carte (taille fixe de 10x10)
#     carte = fcarte3(lat, lon)       # dans fonctions2025

#     with cache_lock:
#         cache_cartes[cle] = carte

#     return carte

import numpy as np



import numpy as np
import glob
import os

for filename in glob.glob("maps2/carteoffset_*.npy"):
    offsets = np.load(filename).astype(np.uint16)
    filename_bin = os.path.splitext(filename)[0] + ".bin"
    offsets.tofile(filename_bin)
    print(filename, "->", filename_bin)








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
    print (offsets)

    pas = 1/730                                              # un pas = 1/730 °
    # on va transformer le fichier numpy d'offsets en coordonnees 
    coords = np.empty_like(offsets, dtype=np.float32)
    coords[..., 0] = lat0 - offsets[..., 0].astype(np.float32) * pas    # latitude diminue vers le sud
    coords[..., 1] = lon0 + offsets[..., 1].astype(np.float32) * pas    # longitude augmente vers l’est
    return coords 


# lat=45
# lon=-3

# coords= fcarte3(lat,lon)

# carteliste =coords.tolist() 
# # print (coords)
# # print (carteliste)