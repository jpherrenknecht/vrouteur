from   psycopg2 import pool
from psycopg2 import sql
import datetime
import time
from datetime import datetime,timedelta,UTC
import json


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
cur = conn.cursor()


def list_databases():
    conn = pg_pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
        dbs = [r[0] for r in cur.fetchall()]
        cur.close()
        return dbs
    finally:
        pg_pool.putconn(conn)


def list_tables():
    conn = pg_pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [r[0] for r in cur.fetchall()]
        cur.close()
        return tables
    finally:
        pg_pool.putconn(conn)        





def list_tables_with_counts():
    conn = pg_pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [r[0] for r in cur.fetchall()]

        result = {}
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            result[t] = count

        cur.close()
        return result
    finally:
        pg_pool.putconn(conn)


print(list_tables_with_counts())

def clean_old_records_multi():
    """Nettoie toutes les tables selon leur délai de rétention et type de timestamp."""

    # ⏳ Délais personnalisés (en jours)
    retention = {
        'boatinfos': 7,
        'coursesactives': 30,
        'fleetinfos': 7,
        'leginfos': 90,
        'personalinfos': 180,
        'progsvr': 5,      # 🕓 attention : timestamptz
        'racesinfos': 30,
        'teammembers': 90
    }

    conn = pg_pool.getconn()
    try:
        cur = conn.cursor()

        for table, days in retention.items():
            # Déterminer le type du champ timestamp
            cur.execute("""
                SELECT data_type
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'timestamp';
            """, (table,))
            result = cur.fetchone()
            if not result:
                print(f"⚠️ {table}: pas de colonne 'timestamp'")
                continue

            timestamp_type = result[0]
            cutoff = None

            # 🧮 Calcul de la limite selon le type
            if "timestamp" in timestamp_type:
                # vrai champ timestamp (TIMESTAMP WITH TIME ZONE)
                cutoff = datetime.now(UTC) - timedelta(days=days)
            else:
                # champ numérique (REAL ou DOUBLE PRECISION)
                cutoff = time.time() - days * 86400

            # 🧹 Suppression
            query = sql.SQL("DELETE FROM {} WHERE timestamp < %s").format(sql.Identifier(table))
            try:
                cur.execute(query, (cutoff,))
                conn.commit()
                print(f"🧹 {cur.rowcount} supprimés de {table} (>{days}j, type={timestamp_type})")
            except Exception as e:
                conn.rollback()
                print(f"⚠️ Erreur sur {table}: {e}")

        cur.close()

    finally:
        pg_pool.putconn(conn)

clean_old_records_multi()


