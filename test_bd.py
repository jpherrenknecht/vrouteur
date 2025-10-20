import psycopg2
from   psycopg2 import pool
import datetime
from datetime import datetime
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
cursor = conn.cursor()


# sera a mettre apres creation des bases dans les fonctions de recherche
conn = pg_pool.getconn()
cursor = conn.cursor()
    # cursor = conn.cursor()
cursor.execute("SELECT version();")
record = cursor.fetchone()
print("Connexion réussie ! Version PostgreSQL :", record)


# except Exception as e:
#     print("Erreur de connexion :", e)


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

user_id="59c2706db395b292ed622d84"
course='757.3'
result=rechercheTablePersonalInfos(user_id, course)
print (result)


def modifpersonalinfos2():
    
    conn = pg_pool.getconn()
    cursor = conn.cursor()
    username='Takron-BSP'

    course='784.1'
    personalinfos={"username": "Takron-BSP", "course": "784.1", "ari": ["Arrivee"],\
    "wp": {"Arrivee": [99, "Arrivee", -0.7386612, -48.33982, 0, "yellow"], "WP1": [1, "WP1", 27.132479801022896, -16.495971679687504, 0.5, "#ffff00"]}, "exclusions": {},\
    "trajets": {}, "tolerancehvmg": 0, \
    "barrieres": {}}

   
    # course='757.3'
    # personalinfos= {"username": "Takron-BSP", "course": "757.3", "ari": ["Arrivee"], "wp": {"Arrivee": [99, "Arrivee", -20.92356, 55.32026, 0, "yellow"], "WP1": [1, "WP1", -24.327076540018634, -15.161132812500002, 0.5, "#ffff00"],\
    #  "WP3": [3, "WP3", -39.823831054924455, 12.230186462402346, 11.318683199999999, "#ffff00"], "WP4": [4, "WP4", -38.190704293996504, 28.495788574218754, 50, "#ffff00"]}, \
    #   "trajets": {}, "tolerancehvmg": 0}
    cursor.execute("""UPDATE personalinfos SET personalinfos = %s WHERE username = %s AND course = %s""", (json.dumps(personalinfos), username, course))
    conn.commit()
    print ('La modification s est executee') 
    cursor.close()
    pg_pool.putconn(conn)  
    return None


# modifpersonalinfos2()
user_id="59c2706db395b292ed622d84"
course='784.1'

result=rechercheTablePersonalInfos(user_id, course)
print (result)





# def rechercheTableCoursesActives(username):
#     conn = pg_pool.getconn()
#     try:
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT  user_id,coursesactives
#             FROM coursesactives
#             WHERE username = %s
#             ORDER BY timestamp DESC
#             LIMIT 1
#         """, (username,))
#         resultats = cursor.fetchone()
#         return resultats if resultats else (None, None)
#     finally:
#         cursor.close()
#         pg_pool.putconn(conn)


# # recherche des courses actives pour un user 

# username='Takron-BSP'
# username ='Charlie2010 BSP'

# result=rechercheTableCoursesActives(username)
# print (' result pour charlie',result)


# username='Francois_FRA-1841-BSP'
# user_id='5fa81586a73ee05018f06d99'


# def rechercheDernieresCoursesActives(limit=50):
#     conn = pg_pool.getconn()
#     try:
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT username, user_id, coursesactives, timestamp
#             FROM coursesactives
#             ORDER BY timestamp DESC
#             LIMIT %s
#         """, (limit,))
#         resultats = cursor.fetchall()
#         return resultats
#     finally:
#         cursor.close()
#         pg_pool.putconn(conn)



# res= rechercheDernieresCoursesActives(limit=50)    


# for username, user_id, coursesactives, timestamp in res:
#     print(username, user_id, timestamp,coursesactives)


# username='Francois_FRA-1841-BSP'
# user_id='5fa81586a73ee05018f06d99'
# course='767.1'

# res= rechercheTableBoatInfos(user_id,course)
# print()
# print (' boatinfos pour {}  \n{} '.format( username,res))



















# def modifpersonalinfos():
#     data = request.get_json()
#     if not data:
#         return jsonify({"message": "Aucune donnée reçue"}), 400

#     username   = data.get('username')
#     user_id    = data.get('user_id')
#     course     = data.get('course')
#     typeinfo   = data.get('typeinfo')
#     typeaction = data.get('typeaction')
#     nom        = data.get('nom')
#     valeur     = data.get('valeur')

    
#     print(type(valeur))          # → <class 'list'>

#     # if typeinfo=='wp':
    #     elements = valeur.split(',')

    #     valeur = [
    #         int(elements[0]),
    #         elements[1],
    #         float(elements[2]),
    #         float(elements[3]),
    #         float(elements[4]),
    #         elements[5]
    #     ]
    #     print('valeur',valeur)
    


    # print ('on est dans modifpersonalinfos reception en POST ') 
    # print('user_id {} course  {} typeinfo {} typeaction {} nom  {} valeur  {} '.format( user_id, course, typeinfo,typeaction,nom,valeur))

    # if not all([username,user_id, course, typeinfo, typeaction, nom]):
    #     return jsonify({"error": "Parametres Manquant"}), 400
    


    # row=rechercheTablePersonalInfos(user_id,course)           # recupere directement un fichier texte
    # if not row:
    #     return jsonify({"error": "User or course not found"}), 404
    # infos = json.loads(row) if row else {}
    
    # # print ('Ligne 3353 infos recuperees dans la table pour le user_id' ,infos )
    # # print()

    # if typeinfo not in infos:
    #     infos[typeinfo] = {}
    
    # if typeaction == "insert":
    #     print('1695 on est dans insert valeur ',valeur)
    #     if nom not in infos[typeinfo]:
    #        infos[typeinfo][nom] =valeur if valeur else 0
    #        print ('valeur de infos pour insertion' ,infos )
    #     #    print()
    
    
    # elif typeaction == "delete":
    #       print ('ligne 2934  avant suppression  ',infos )
    #       print ('typeinfo a supprimer ',typeinfo)
    #       print ('nom de l info ',nom)
    #       infos[typeinfo].pop(nom, None)
    #     #   print('valeur de infos apres suppression ', infos )
    #     #   print()
    #     #   del infos[typeinfo][nom] 

    # elif typeaction == "modify":

    #     print('on est dans modify')
    #     if not valeur:
    #         return jsonify({"error": "Missing valeur for modification"}), 400
       
        
    #     if typeinfo == "tolerancehvmg":
    #         try:
    #             valeur = float(valeur)  # ou int(valeur) si tu veux un entier
    #         except ValueError:
    #             return jsonify({"error": "valeur invalide pour tolerancehvmg"}), 400
    #         infos["tolerancehvmg"] = valeur

    #     else:
    #         infos[typeinfo][nom] = json.loads(valeur)

    #     # print ('valeur de infos apres modification' ,infos )
    #     # print()

    # else:
    #     return jsonify({"error": "Invalid typeaction"}), 400
    # # print ('valeur json.dumps(infos) qui va etre enregistree \n',json.dumps(infos))

    # print()
    # print('test ', json.dumps(infos), username, course)
    # print()
    # conn = pg_pool.getconn()
    # cursor = conn.cursor()
    # cursor.execute("""UPDATE personalinfos SET personalinfos = %s WHERE username = %s AND course = %s""", (json.dumps(infos), username, course))
    # conn.commit()
    # print ('La modification s est executee') 
    # cursor.close()
    # pg_pool.putconn(conn)  
    

    # return jsonify({"message":'La modification de personalinfos s est deroulee avec succes ', "success": True,  "updated_infos": infos })

# Extraction des polaires 
    
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

polar_id=3
res= rechercheTablePolaires(polar_id)

print()
print (res)
  
