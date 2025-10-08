import json

# Charger le JSON d'origine
with open("js/restrictedzones.json", "r") as f:
    data = json.load(f)





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
with open("restrictedzones.geojson", "w") as f:
    json.dump(geojson, f, indent=2)