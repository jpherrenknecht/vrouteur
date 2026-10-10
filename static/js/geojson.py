import json

# Charger le JSON d'origine
filename='/home/jp/vrouteur/static/js/restricted.js'
with open(filename, "r") as f:
    data = json.load(f)

# with open("restricted.js", "r", encoding="utf-8") as f:
#     contenu = f.read()



geojson = {
    "type": "FeatureCollection",
    "features": []
}

for zone in data["restrictedZones"]:
    name = zone.get("name")
    rate = zone.get("slowDownRate")
    vertices = zone.get("vertices", [])

    # Construire la liste des coordonnées [lon, lat]
    coordinates = [[(point["lon"], point["lat"]) for point in vertices]]

    # Fermer le polygone si ce n'est pas déjà le cas
    if coordinates[0][0] != coordinates[0][-1]:
        coordinates[0].append(coordinates[0][0])

    feature = {
        "type": "Feature",
        "properties": {
            "name": name,
            "slowDownRate": rate
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        }
    }

    geojson["features"].append(feature)

# Sauvegarder en GeoJSON
filename='/home/jp/vrouteur/static/js/restrictedzones.geojson'
with open(filename, "w") as f:
    json.dump(geojson, f, indent=2)