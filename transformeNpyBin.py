import numpy as np

filename = "/home/jp/BSP_Dash_8.9/maps2/carteoffset_50_10.npy"

# Chargement du fichier numpy
offsets = np.load(filename)

# Sauvegarde en binaire brut
offsets.tofile("/home/jp/BSP_Dash_8.9/maps2/carteoffset_50_10.bin")

print("Shape :", offsets.shape)
print("dtype :", offsets.dtype)
print("Fichier .bin créé")