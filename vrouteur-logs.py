
import os
import psycopg2
from datetime import datetime
import argparse





# pg_pool = pool.SimpleConnectionPool(
#                                         1, 10,  # minconn, maxconn
#                                         dbname="vrouteur",
#                                         user="jp",
#                                         password="Licois1000",
#                                         host="localhost",  # ou l'adresse IP/nom de domaine
#                                         port="5432"        # par défaut PostgreSQL
#                                     )



# --- CONFIG BDD : adapte si besoin ---
DB_NAME = os.getenv("VROUTEUR_DB_NAME", "vrouteur")
DB_USER = os.getenv("VROUTEUR_DB_USER", "jp")
DB_PASSWORD = os.getenv("VROUTEUR_DB_PASSWORD", "Licois1000")
DB_HOST = os.getenv("VROUTEUR_DB_HOST", "localhost")
DB_PORT = os.getenv("VROUTEUR_DB_PORT", "5432")


def format_ts(ts):
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).strftime("%d/%m %H:%M")


def fetch_last(conn, limit):
    sql = """
        SELECT id, timestamp, username, course, user_id, status,
               heuredepart, lat, lon, eta
        FROM historoutages
        ORDER BY timestamp DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall()
    # On inverse pour afficher dans l'ordre croissant (du plus vieux au plus récent)
    return list(reversed(rows))


def safe_str(value):
    return "" if value is None else str(value)


def print_results(rows):
    for row in rows:
        id_, ts, user, course, user_id, status, hd, lat, lon, eta = row
        print(
            f"{id_:4} | "
            f"{format_ts(ts):12} | "
            f"{safe_str(user):25} | "
            f"{safe_str(course):8} | "
            f"{safe_str(user_id):8} | "
            f"Status={safe_str(status):8} | "
            f"Depart {format_ts(hd):12} | "
            f"lat : {lat:8.4f} | "
            f"lon : {lon:9.4f} | "
            f"Arrivee={format_ts(eta):12}"
        )




def main():
    parser = argparse.ArgumentParser(
        description="Affiche les derniers enregistrements de historoutages"
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=50,
        help="nombre de lignes à afficher (par défaut 50)"
    )
    args = parser.parse_args()

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )

    try:
        rows = fetch_last(conn, args.limit)
        print_results(rows)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
