# Importing Sqlite3 Module
import sqlite3 as sql
import time
import json 

staticbd='/home/jp/static/bd/basededonnees.db'
sqliteConnection3 =sql.connect(staticbd) 
conn =sql.connect(staticbd) 
cursor = conn.cursor() 

#############################################################################################
##############    Obtention de la liste des tables de la base 

def get_table_list(database):
    try:
        # Connexion à la base de données SQLite3
        conn = sql.connect(database)
        cursor = conn.cursor() 
        # Requête pour obtenir les tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        
        # Récupération des résultats
        tables = cursor.fetchall()
        
        # Formatage en liste simple
        table_list = [table[0] for table in tables]
        
        # Affichage ou retour de la liste des tables
        print("Tables dans la base de données:")
        for table in table_list:
            print(f"- {table}")
        
        return table_list
    except sql.Error as e:
        print(f"Erreur lors de l'accès à la base de données : {e}")
        return []
    finally:
        if conn:
            conn.close()


print('\nListe des tables initiale')
tables = get_table_list(staticbd)


#############################################################################################""






##############################################################################################
######   Destruction d une table      ##############################################
##############################################################################################
# 43 destruction d 'une table 
#65 Obtention d la liste des tables

# Utilitaires de gestion des bases 
def delete_table(database, table_name):
    try:        
            with sql.connect(database)as conn :
                conn = sql.connect(database)
                cursor = conn.cursor()
        # Commande pour supprimer la table
            requete = f"DROP TABLE IF EXISTS {table_name};"
            cursor.execute(requete)
        # Valider la suppression
            conn.commit()
            print(f"La table '{table_name}' a été supprimée avec succès.")
    except sql.Error as e:
        print(f"Erreur lors de la suppression de la table : {e}")

# #Exemple d'utilisation pour la suppression de base 
# delete_table(staticbd, "personalinfos2")
# print('\nListe des tables apres suppression')
# tables = get_table_list(staticbd)



##############################################################################################
######    Exemple de creation d une table       ##############################################
##############################################################################################
# conn = sql.connect(staticbd)
# cursor = conn.cursor()
# cursor.execute("""
#                         CREATE TABLE IF NOT EXISTS personalinfos2
#                             (
#                                 id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE ,
#                                 timestamp REAL,
#                                 username TEXT,
#                                 user_id TEXT,
#                                 course TEXT,
#                                 infos TEXT                          
#                             )          
#                                 """)

# print ('la table personalinfos2 a ete cree ')

# conn.commit()
# conn.close()



##############################################################################################
######    Exemple de creation d une table       ##############################################
##############################################################################################
# conn = sql.connect(staticbd)
# cursor = conn.cursor()
# cursor.execute("""
#                         CREATE TABLE IF NOT EXISTS fleetinfos
#                             (
#                                 id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE ,
#                                 timestamp REAL,
#                                 username TEXT,
#                                 user_id TEXT,
#                                 course TEXT,
#                                 fleetinfos TEXT                          
#                             )          
#                                 """)

# print ('la table fleetinfos a ete creee ')

# conn.commit()
# conn.close()











#############################################################################################""
##############    Vider une table 
#############################################################################################""
def vider_table(conn, table_name):
    cursor = conn.cursor()
    
    # Supprime toutes les lignes de la table
    cursor.execute(f"DELETE FROM {table_name};")
    
    # Réinitialise l'auto-incrémentation si la table a une clé primaire auto-incrémentée
    cursor.execute("DELETE FROM sqlite_sequence WHERE name = ?;", (table_name,))
    
    # Valide les modifications
    conn.commit()


# # Utilisation avec une connexion SQLite existante
maBase = staticbd
conn = sql.connect(maBase)
table ='personalinfos2'

# try:
#     vider_table(conn, table)
#     print("\nTable {} vidée avec succès !".format(table))
# finally:
#     conn.close()

#############################################################################################""
##############    rechercher dans  table 
#############################################################################################""


def rechercheTablePersonalInfos(username,course,typeinfo):
    # retourne les differentes zones sous forme de tableaux 
    with sql.connect(staticbd) as conn :
        conn.row_factory = sql.Row
        cursor=conn.cursor()
        recherche ="SELECT nominfo,valeur FROM personalinfos WHERE username=? and course=? and typeinfo=? " 
        donnees=(username,course,typeinfo)
        cursor.execute(recherche,donnees)
        nominfo=[]
        valeurs= []
        for row in cursor.fetchall() :
             nominfo.append(row[0])    # Ajouter `nominfo` à la liste
             valeur=json.loads(row[1])
             valeurs.append( valeur)   # Ajouter `valeur` à la liste
    return nominfo,valeurs



# fonction amelioree pour personalinfos2 

def rechercheTablePersonalInfos2(username, course):
    with sql.connect(staticbd) as conn:
        cursor = conn.cursor()
        recherche = """
        SELECT infos 
        FROM personalinfos2 
        WHERE username=? AND course=? 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        donnees = (username, course)
        cursor.execute(recherche, donnees)
        row = cursor.fetchone()  # Récupérer une seule ligne
        if row:
            return row[0]  # Retourne directement la chaîne JSON
        return None  # Si aucun résultat, renvoie None (ou "{}" si nécessaire)



username='Takron-BSP'
course='709.1'
infos=rechercheTablePersonalInfos2(username,course)
print()
print(infos)




def rechercheTableFleetinfos(user_id, course):
    with sql.connect(staticbd) as conn:
        cursor = conn.cursor()
        recherche = """
        SELECT fleetinfos 
        FROM fleetinfos 
        WHERE user_id=? AND course=? 
        ORDER BY timestamp DESC 
        LIMIT 1
        """
        donnees = (user_id, course)
        cursor.execute(recherche, donnees)
        row = cursor.fetchone()  # Récupérer une seule ligne
        if row:
            return row[0]  # Retourne directement la chaîne JSON
        return None  # Si aucun résultat, renvoie None (ou "{}" si nécessaire)


user_id= "59c2706db395b292ed622d84"
course='710.1'
fleetinfos=rechercheTableFleetinfos(user_id, course)

print (fleetinfos)




#############################################################################################""
##############    Structure d une table 
#############################################################################################""



def structure(database,table):
     try:
        conn = sql.connect(database)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info("+table+" )")
       
        # Récupérer et afficher les résultats
        columns = cursor.fetchall()
        for column in columns:
            print(column)
     except sql.Error as e:
        print(f"Erreur lors de l affichage de la structure  : {e}")   
     finally:
        if conn:
            conn.close()
   


# print()
# nomtable='personalinfos2'
# print ('structure de la table' ) 
# structure(staticbd,nomtable)


#############################################################################################""
##############    modif dans  table personalinfos2
#############################################################################################""

def modifpersonalinfos():
    # username    = request.args.get('username')   
    # course      = request.args.get('course')
    # typeinfo    = request.args.get('typeinfo')
    # typeaction  = request.args.get('typeaction')
    # nom         = request.args.get('nom')
    # valeur      = request.args.get('valeur')
    # # on decode valeur 
    # valeur = unquote(valeur)


    username    = 'Takron-BSP'  
    course      = '709.1'
    typeinfo    = 'routage'
    typeaction  = 'delete'
    nom         = "Routage 18/02 10:35"
    valeur      = ' '
    # on decode valeur 
    


    
    print('valeur', valeur)
    print ('username',username)
    print('course',course)
    print() 
    print ('on est dans modifpersonalinfos') 
    print()

    # if not all([username, course, typeinfo, typeaction, nom]):
    #     return jsonify({"error": "Parametres Manquant"}), 400
    


    row=rechercheTablePersonalInfos2(username,course)           # recupere directement un fichier texte
   
    # conn = sql.connect(staticbd)
    # conn.row_factory = sql.Row
    # cur = conn.cursor()
    # cur.execute("SELECT infos FROM personalinfos2 WHERE username = ? AND course = ? ORDER BY timestamp DESC LIMIT 1", (username, course))
    # row = cur.fetchone()
    


    print('row',row)   
    print()



    # if not row:
    #     return jsonify({"error": "User or course not found"}), 404

    #infos = json.loads(row["infos"]) if row["infos"] else {}

    infos = json.loads(row) if row else {}
    
    # print ('Ligne 2885 infos recuperees ',infos )
    # print()

    if typeinfo not in infos:
        infos[typeinfo] = {}
    
    if typeaction == "insert":
        print('on est dans innsert')
        if nom not in infos[typeinfo]:
           infos[typeinfo][nom] =json.loads(valeur) if valeur else []
           print ('valeur de infos apres insertion' ,infos )
    
    
    elif typeaction == "delete":
          print ('ligne 2934  avant suppression  ',infos )
          print ('typeinfo a supprimer ',typeinfo)
          print ('nom de l info ',nom)
          infos[typeinfo].pop(nom, None)
          print() 
          print('valeur de infos apres suppression ', infos )
         # del infos[typeinfo][nom]

    elif typeaction == "modify":
        print('on est dans modify')
        # if not valeur:
        #     return jsonify({"error": "Missing valeur for modification"}), 400
        infos[typeinfo][nom] = json.loads(valeur)

        print ('valeur de infos apres modification' ,infos )

    else:
        print('Erreur')
        # return jsonify({"error": "Invalid typeaction"}), 400
    

    # print ('valeur json.dumps(infos) qui va etre enregistree \n',json.dumps(infos))
    conn = sql.connect(staticbd)
    conn.row_factory = sql.Row
    cur = conn.cursor()
    cur.execute("UPDATE personalinfos2 SET infos = ? WHERE username = ? AND course = ?", (json.dumps(infos), username, course))
    conn.commit()
    conn.close()
    
    

    # return jsonify({"success": True, "updated_infos": infos})
    # #return jsonify({"message": 'mise au point en cours '})


# modifpersonalinfos()







#############################################################################################""
##############    Purger les tables 
#############################################################################################""






def delete_old_records():
    # Définir les règles de rétention par table
    staticbd= '/home/jp/static/bd/basededonnees.db'
    conn = sql.connect(staticbd)
    cursor = conn.cursor()
    retention_policies = {
        "boatinfos2": 7,   # 7 jours
        "coursesactives": 10, 
        "leginfos": 7,   
        "personalinfos":7,
        "racesinfos":30,
        "progsvr":2
        }
    
    try:   
        # Parcourir chaque table et appliquer la règle de rétention
        for table, peremption in retention_policies.items():
            cutoff_time = time.time() - (peremption * 24 * 60 * 60)
            
            # Supprimer les enregistrements plus vieux que le cutoff_time
            requete = f"DELETE FROM {table} WHERE timestamp < ?"
            cursor.execute(requete, (cutoff_time,))
            print(f"Nettoyage de {table}: {cursor.rowcount} enregistrements supprimés.")
        
        conn.commit()
    except sql.Error as e:
        print(f"Erreur lors du nettoyage des tables : {e}")
    finally:
        if conn:
            conn.close()




# delete_old_records()



#############################################################################################""
##############   
#############################################################################################""




# recherche boatinfos par username 
# def rechercheTableBoatInfos(username,course):
#     staticbd='/home/jp/static/bd/basededonnees.db'
#     with sql.connect(staticbd) as conn :
#         cursor=conn.cursor()
#         recherche ="SELECT username,user_id,course,Max(t0),boatinfos FROM boatinfos2 WHERE username=? and course=? " 
#         donnees=(username,course)
#         data= cursor.execute(recherche,donnees)
#         for row in data :
#             username            = row[0]
#             user_id             = row[1]
#             course              = row[2]
#             t0                  = row[3]
#             boatinfos           = row[4]
#     return boatinfos

# username='Takron-BSP'
# course='654.1'
# boatinfos=rechercheTableBoatInfos(username,course)
# print(' rechercheboatinfos par username \n ',boatinfos)


# recherche boatinfos par id 
# def rechercheTableBoatInfos2parId(user_id,course):
#     with sql.connect(staticbd) as conn :
#         cursor=conn.cursor()
#         recherche ="SELECT username,user_id,course,Max(t0),boatinfos FROM boatinfos2 WHERE user_id=? and course=? " 
#         donnees=(user_id,course)
#         data= cursor.execute(recherche,donnees)
#         for row in data :
#             username            = row[0]
#             user_id             = row[1]
#             course              = row[2]
#             t0                  = row[3]
#             boatinfoscompletes  = row[4]
#     return boatinfoscompletes


# user_id="59c2706db395b292ed622d84"
# course='654.1'
# boatinfoscompletes=rechercheTableBoatInfos2parId(user_id,course)
# print('\nboatinfoscompletes pour recherche par id  \n',boatinfoscompletes)




# recherche trajectoire

# def rechercheTableProgsvr(user_id,course):
#     with sql.connect(staticbd) as conn :
#         cursor=conn.cursor()
#         recherche ="SELECT user_id,course,Max(t0),polyline,programmations FROM progsvr WHERE user_id=? and course=? " 
#         donnees=(user_id,course)
#         data= cursor.execute(recherche,donnees)
#         for row in data :
#             user_id             = row[0]
#             course              = row[1]
#             t0                  = row[2]
#             polyline            = row[3]
#             programmations      = row[4]
#     return t0,polyline,programmations

# user_id="59c2706db395b292ed622d84"
# course='654.1'
# t0,polyline,programmations =rechercheTableProgsvr(user_id,course)

# print()
# print(' enregistrement du ',time.strftime("%Hh:%Mmn:%Ss", time.localtime(t0)))
# print()
# print('polyline',polyline)
# print('programmations ',programmations)


# recherche zoneexclusions


# nomZone='test1 '
# nominfo,valeur=rechercheTablePersonalInfos(username,course,typeinfo)

# print()
# print(nominfo)
# print()
# print(valeur)
# print()


#             username    TEXT NOT NULL,
#             course      TEXT NOT NULL,  
#             typeinfo    TEXT NOT NULL,  
#             nominfo     TEXT NOT NULL,  
#             valeur      TEXT NOT NULL




# with sql.connect(staticbd)as conn :
#         cursor=conn.cursor()
#         efface="DELETE FROM personalinfos WHERE username=? and course=? and nominfo=?"
#         donnees=(username,course,nomZone)
#         cursor.execute(efface,donnees)
#         conn.commit()
#         print ('zone {} effacée dans la base personalinfos '.format(nomZone ))


# nominfo,valeur=rechercheTablePersonalInfos(username,course,typeinfo)

# print()
# print('Apres effacement')
# print(nominfo)
# print()
# print(valeur)
# print()

###############################################################################################
#########   Liste des tables       ############################################################
###############################################################################################


   
# Exemple d'utilisation

#tables = get_table_list(staticbd)


# Connexion à la base de données SQLite3





# def delete2_records(): 
#     try:   
#         conn = sql.connect(staticbd)
#         cursor = conn.cursor()
#         retention_policies = {
#         "boatinfos2": 7 , # 7 jours
#         # "coursesactives": 30,  # 30 jours
#         # "leginfos": 7,   # 7 jours
#         # "personalinfos":7,
#         # "polaires":30,
#         "racesinfos":30,
#         "progsvr":2
#         }

#         current_time=time.time()
#         for table, retention_days in retention_policies.items():
#             cutoff_time = current_time - (retention_days * 24 * 60 * 60)
                
#                 # Supprimer les enregistrements plus vieux que le cutoff_time
#             requete = f"DELETE FROM {table} WHERE timestamp < ?"
#             cursor.execute(requete, (cutoff_time,))
#             print(f"Nettoyage de {table}: {cursor.rowcount} enregistrements supprimés.")
        
#         conn.commit()

#     except sql.Error as e:
#         print(f"Erreur lors du nettoyage des tables : {e}")
    
#     finally:
#         if conn:
#             conn.close()    
    



    








# delete_old_records()

#delete2_records()


# def rename_column(database_path):
#     # renomme une colonne en recreant une copie de la base 
#     try:
#         conn = sql.connect(database_path)
#         cursor = conn.cursor()

#         # 1. Renommer la table existante
#         cursor.execute("ALTER TABLE progsvr RENAME TO progsvr_old2;")
        
#         # 2. Créer une nouvelle table avec le nouveau nom de colonne
#         cursor.execute("""
#                      CREATE TABLE IF NOT EXISTS progsvr
#                          (
#                                id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE ,
#                                username TEXT,
#                                user_id TEXT,
#                                course TEXT,
#                                timestamp REAL,
#                                polyline TEXT,
#                                programmations TEXT
#                          )          
#                                """)

#         cursor.execute("""
#             INSERT INTO progsvr (id, user_id,course,timestamp, polyline,programmations)
#             SELECT               id, user_id,course,t0       , polyline,programmations FROM progsvr_old2;
#         """)

#         # destruction de la copie a remetttre

#         conn.commit()
#         print("Renommage de la colonne 't0' en 'timestamp' réussi.")
#     except sql.Error as e:
#                 print(f"Erreur lors du renommage de la colonne : {e}")
#     finally:
#         if conn:
#             conn.close()



# delete_table('staticbd', 'progsvr')
# #rename_column(staticbd)


    




    




#delete2_records()

#delete_table(staticbd, 'boatinfos2_old')

# print('\n liste apres destuction ')
# get_table_list(staticbd)






# Je vais ex


#affichage des polaires
# with sql.connect(staticbd) as conn:
#     cursor=conn.cursor()
#     cursor.execute("select * FROM polaires")
#     resultats = cursor.fetchall()
#     for resultat in resultats:
#         print (resultat)

# conn.close()        


# print()

# # recherche polaires 21
# polar_id=21

# conn=sql.connect(staticbd) 
# cursor=conn.cursor()
# cursor.execute("select polaires FROM polaires where _id=? ",(polar_id,) )
# result = cursor.fetchone()
# stringpolaires=result[0]

# print (stringpolaires)


# conn.close()     
# 


 
#                                 username TEXT,
#                                 user_id TEXT,
#                                 course TEXT,
#                                 infos TEXT    







# def rechercheTablePersonalInfos2(username,course):
#     # retourne les differentes zones sous forme de tableaux 
#     with sql.connect(staticbd) as conn :
#         cursor=conn.cursor()
#         recherche ="SELECT username,user_id,course,Max(timestamp),infos FROM personalinfos2 WHERE username=? and course=? "
#         donnees=(username,course)
#         data= cursor.execute(recherche,donnees)
#         for row in data :
#             username            = row[0]
#             user_id             = row[1]
#             course              = row[2]
#             timestamp           = row[3]
#             infos               = row[4]
           
#     return infos


# def rechercheTablePersonalInfos2(username, course):
#     with sql.connect(staticbd) as conn:
#         cursor = conn.cursor()
#         # Correction pour récupérer l'entrée avec le timestamp max
#         recherche = """
#         SELECT username, user_id, course, timestamp, infos 
#         FROM personalinfos2 
#         WHERE username=? AND course=? 
#         ORDER BY timestamp DESC 
#         LIMIT 1
#         """
#         donnees = (username, course)
#         cursor.execute(recherche, donnees)
#         row = cursor.fetchone()  # Récupérer une seule ligne

#         if row:
#             username = row[0]
#             user_id = row[1]
#             course = row[2]
#             timestamp = row[3]
#             infos = json.loads(row[4])  # Désérialiser JSON en dict
#             return {
#                 "username": username,
#                 "user_id": user_id,
#                 "course": course,
#                 "timestamp": timestamp,
#                 "infos": infos,
#             }
#         return None  # Retourner None si aucune donnée n'est trouvée






# timestamp=time.time()
# username='Takron-BSP'
# course='698.1'
# user_id="59c2706db395b292ed622d84"
# wp=[[1,'WP1',-56.25,-65.39,10],[1,'WP2',-60.25,-65.39,10],[2,'Arrivee',10,-65.39,10]]
# exclusions=[{'nom':'malouines','polygone':[[10,10],[10,15],[10,15],[15,10]]},{'nom':'test','polygone':[[20,20],[20,25],[20,25],[25,20]]}]
# infos=json.dumps({"wp":wp,"exclusions":exclusions})

############################################################################################################################
#   jeu de test pour personalinfos2
############################################################################################################################

# timestamp=time.time()
# user_id="59c2706db395b292ed622d84"
# username='Takron-BSP'
# course='698.1'
# ari=['wp1','wp2']
# wp={'wp1':[1,'WP1',-56.25,-65.39,10],'wp2':[2,'WP2',-60.25,-65.39,10],'wp5':[3,'Arrivee',10,-65.39,10] }
# exclusions={'Kavaratti':[[10.5621,72.7618],[5.23,76.18],[5.05,75.22],[10.5,72.64],[10.5621,72.7618] ]   ,'test':[[20,20],[20,25],[20,25],[25,20]] }
# trajet={'trajet1':[[10.5621,72.7618],[5.23,76.18],[5.05,75.22],[10.5,72.64],[10.5621,72.7618] ]}
# infos=json.dumps({"username":username,"course":course,'ari':ari,"wp":wp,"exclusions":exclusions,"trajets":trajet})
# print ('infos',infos)

# with sql.connect(staticbd) as conn:
#         cursor = conn.cursor()
# cursor.execute('INSERT INTO personalinfos2 (timestamp,username,course,infos ) VALUES (?,?,?,?)', (timestamp,username,course,infos))
# conn.commit()
# print ('enregistrement effectue')
# #recherche de l enregistrement 
# infos=rechercheTablePersonalInfos2(username,course)

# # print ('infos\n' ,infos)
# # infos=json.loads(infos)
# # print ('Waypoints   : ',infos['wp'])
# # print ('Waypoints 0  : ',infos['wp'][0])
# # print ('Exclusions 0  : ',infos['exclusions'][0])
# # print ('Polygone Exclusions 0  : ',infos['exclusions'][0]['polygone'])

# # procedure d ajout d'un wp 


# print ('infos' ,infos)
# print()
############################################################################################################################
#   Modification de json  Ajout suppression 
############################################################################################################################


# type_ = 'wp'
# nom = 'wp3'
# valeur = [3, "WP3", -50, 15, 3]

# json_str='{"username": "Takron-BSP", "course": "698.1", "ari": ["wp1", "wp2"], "wp": {"wp1": [1, "WP1", -56.25, -65.39, 10], "wp2": [2, "WP2", -60.25, -65.39, 10]}}'
# data = json.loads(json_str)
# if type_ in data:
#     data[type_][nom] = valeur  # Ajouter à "wp"
# else:
#     data[type_] = {nom: valeur}  # Créer "wp" si absent
#     # Sérialiser en JSON
# json_str_updated = json.dumps(data)
# print(json_str_updated)



# json_str='{"username": "Takron-BSP", "course": "698.1", "ari": ["wp1", "wp2"], "wp": {"wp1": [1, "WP1", -56.25, -65.39, 10], "wp2": [2, "WP2", -60.25, -65.39, 10]}}'
# data = json.loads(json_str)
# type_ = 'wp'
# nom = 'wp2'

# # Supprimer l'élément s'il existe
# if type_ in data and nom in data[type_]:
#     del data[type_][nom]  # Suppression de 'wp3'

# json_str_updated = json.dumps(data)
# print(json_str_updated)

############################################################################################################################
#   
############################################################################################################################

