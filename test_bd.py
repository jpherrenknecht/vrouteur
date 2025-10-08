import psycopg2
from   psycopg2 import pool
import datetime
from datetime import datetime
try:

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
    cursor = conn.cursor()
      # cursor = conn.cursor()
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("Connexion réussie ! Version PostgreSQL :", record)


except Exception as e:
    print("Erreur de connexion :", e)


def rechercheTableBoatInfos(user_id, course):
    conn = None
    cursor = None
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
    except Exception as e:
        print("Erreur lors de l'exécution SQL :", e)

    finally:
        if cursor:
            cursor.close()
        if conn:
            pg_pool.putconn(conn)  # très important : rend la connexion au pool    
        





# try:
#     # ⚠️ adapte les paramètres si besoin
#     conn = psycopg2.connect(
#         dbname="vrouteur",   # la base par défaut
#         user="jp",     # l'utilisateur par défaut si tu en as un
#         password="Licois1000", # mot de passe si tu en as défini un
#         host="localhost",
#         port="5432"
#     )

  
   

cursor.execute("""
                        CREATE TABLE IF NOT EXISTS boatinfos
                            (
                                id SERIAL PRIMARY KEY  ,
                                timestamp REAL,
                                username TEXT,
                                user_id TEXT,
                                course TEXT,
                                boatinfos TEXT                          
                            )          
                                """)
conn.commit()
print ('la table boatinfos      a ete creee si elle n existait pas ')



# Exemple d'enregistrement
cursor.execute("""
    INSERT INTO boatinfos (timestamp, username, user_id, course, boatinfos)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id;
""", (
    datetime.now().timestamp(),   # timestamp en secondes
    "takron",                     # username
    "USR123",                     # user_id
    "vendée_globe",               # course
    '{"lat":48.85,"lon":2.35,"speed":12.5}'  # boatinfos en JSON (sous forme texte ici)
))

new_id = cursor.fetchone()[0]
conn.commit()

print(f"✅ Nouvel enregistrement inséré avec id={new_id}")

cursor.close()
conn.close()

user_id = "USR123"
course = "vendée_globe"

result=rechercheTableBoatInfos(user_id, course)

print ('result:', result)