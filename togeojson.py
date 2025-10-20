import json

# Exemple de ton tableau (ici abrégé, mets ton tableau complet)
zones = [{"name":"FFV - DST Les Casquets","slowDownRate":0.3,"vertices":[{"lat":50.05666667,"lon":-2.956666667},{"lat":50.14416667,"lon":-2.468333333},{"lat":49.85333333,"lon":-2.351666667},{"lat":49.76833333,"lon":-2.838333333}],"bbox":[49.76833333,-2.956666667,50.14416667,-2.351666667]},{"name":"FFV - DST Ouessant","slowDownRate":0.3,"vertices":[{"lat":49.03416667,"lon":-5.611666667},{"lat":48.94,"lon":-5.86},{"lat":48.70833333,"lon":-6.051666667},{"lat":48.58333333,"lon":-5.708333333},{"lat":48.48916667,"lon":-5.3675},{"lat":48.62,"lon":-5.1975},{"lat":48.81,"lon":-5.416666667}],"bbox":[48.48916667,-6.051666667,49.03416667,-5.1975]},{"name":"FFV - TSS Cabo Sao Vicente","slowDownRate":0.3,"vertices":[{"lat":37.04166667,"lon":-9.195},{"lat":36.945,"lon":-9.171666667},{"lat":36.85833333,"lon":-9.071666667},{"lat":36.835,"lon":-8.953333333},{"lat":36.42,"lon":-9.1},{"lat":36.47433333,"lon":-9.36},{"lat":36.73666667,"lon":-9.664166667},{"lat":36.94333333,"lon":-9.721666667}],"bbox":[36.42,-9.721666667,37.04166667,-8.953333333]},{"name":"FFV - TSS Canaries Est","slowDownRate":0.3,"vertices":[{"lat":28.34166667,"lon":-14.95166667},{"lat":28.33,"lon":-14.795},{"lat":27.813,"lon":-15.00583333},{"lat":27.85833333,"lon":-15.1475}],"bbox":[27.813,-15.1475,28.34166667,-14.795]},{"name":"FFV - TSS Canaries Ouest","slowDownRate":0.3,"vertices":[{"lat":28.635,"lon":-15.78},{"lat":28.56333333,"lon":-15.655},{"lat":27.97333333,"lon":-16.21583333},{"lat":28.0575,"lon":-16.3275}],"bbox":[27.97333333,-16.3275,28.635,-15.655]},{"name":"FFV - TSS Cape Roca","slowDownRate":0.3,"vertices":[{"lat":38.86666667,"lon":-10.23},{"lat":38.86666667,"lon":-9.685},{"lat":38.66166667,"lon":-9.666666667},{"lat":38.565,"lon":-10.195},{"lat":38.68166667,"lon":-10.23}],"bbox":[38.565,-10.23,38.86666667,-9.666666667]},{"name":"FFV - TSS Finisterre","slowDownRate":0.3,"vertices":[{"lat":43.52333333,"lon":-10.08666667},{"lat":43.35,"lon":-9.606666667},{"lat":43.175,"lon":-9.733333333},{"lat":42.88,"lon":-9.733333333},{"lat":42.88,"lon":-10.23083333},{"lat":43.31583333,"lon":-10.23083333}],"bbox":[42.88,-10.23083333,43.52333333,-9.606666667]},{"name":"FFV - TSS South Scilly","slowDownRate":0.3,"vertices":[{"lat":49.7675,"lon":-6.275833333},{"lat":49.76716667,"lon":-6.4925},{"lat":49.5925,"lon":-6.568333333},{"lat":49.59233333,"lon":-6.273333333}],"bbox":[49.59233333,-6.568333333,49.7675,-6.273333333]},{"name":"TCO25 - ZI Mauritanie","slowDownRate":0.3,"vertices":[{"lat":21.51666667,"lon":-17.58333333},{"lat":21.51666667,"lon":-16.41666667},{"lat":16,"lon":-16.41666667},{"lat":16,"lon":-17.58333333}],"bbox":[16,-17.58333333,21.51666667,-16.41666667]},{"name":"TCO25 - ZI Nord Amérique du Sud","slowDownRate":0.3,"vertices":[{"lat":11.35833333,"lon":-60.51666667},{"lat":9.519166667,"lon":-60.96333333},{"lat":-5.161666667,"lon":-35.48333333},{"lat":-3.856,"lon":-33.8175}],"bbox":[-5.161666667,-60.96333333,11.35833333,-33.8175]},{"name":"TCO25 - ZI Parc Eolien Calvados","slowDownRate":0.3,"vertices":[{"lat":49.50438333,"lon":-0.6000166667},{"lat":49.49511667,"lon":-0.51835},{"lat":49.48466667,"lon":-0.4989333333},{"lat":49.48015,"lon":-0.4594666667},{"lat":49.45351667,"lon":-0.4252},{"lat":49.4485,"lon":-0.4201833333},{"lat":49.42383333,"lon":-0.4065666667},{"lat":49.4159,"lon":-0.4167833333},{"lat":49.43078333,"lon":-0.54725},{"lat":49.45738333,"lon":-0.5815666667},{"lat":49.46156667,"lon":-0.5820166667},{"lat":49.48593333,"lon":-0.59845},{"lat":49.48666667,"lon":-0.60475}],"bbox":[49.4159,-0.60475,49.50438333,-0.4065666667]},{"name":"TCO25 - ZI Parc Eolien Saint Quay","slowDownRate":0.3,"vertices":[{"lat":48.93,"lon":-2.569166667},{"lat":48.90916667,"lon":-2.616666667},{"lat":48.89666667,"lon":-2.618333333},{"lat":48.82,"lon":-2.575833333},{"lat":48.80583333,"lon":-2.555},{"lat":48.79166667,"lon":-2.509166667},{"lat":48.79666667,"lon":-2.448333333},{"lat":48.8175,"lon":-2.454166667},{"lat":48.885,"lon":-2.5}],"bbox":[48.79166667,-2.618333333,48.93,-2.448333333]},{"name":"TCO25 - Zone Le Havre Plaisanciers","slowDownRate":0.3,"vertices":[{"lat":49.50516667,"lon":0.07391666667},{"lat":49.49408333,"lon":0.09366666667},{"lat":49.49272002,"lon":0.09368530098},{"lat":49.48666667,"lon":0.0905},{"lat":49.49116667,"lon":0.07116666667},{"lat":49.49266667,"lon":0.06283333333}],"bbox":[49.48666667,0.06283333333,49.50516667,0.09368530098]}]




# Conversion en GeoJSON
features = []
for zone in zones:
    coords = [[v["lon"], v["lat"]] for v in zone["vertices"]]
    # fermer le polygone si nécessaire
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    feature = {
        "type": "Feature",
        "properties": {
            "name": zone["name"],
            "slowDownRate": zone["slowDownRate"],
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords]
        },
        "bbox": zone["bbox"]
    }
    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

# Sauvegarde en fichier
with open("zones.geojson", "w", encoding="utf-8") as f:
    json.dump(geojson, f, indent=2, ensure_ascii=False)

print("✅ Conversion terminée : zones.geojson")
