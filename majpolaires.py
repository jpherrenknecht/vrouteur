#calculpolaires
import numpy as np
# import sqlite3 as sql
import math
import os
import sys
import json
import time
import datetime
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator,interp2d,interpn
import json

from psycopg2 import pool

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



print('test ')

# on va charger les polaires dans la base de donnees 







# on va charger les polaires dans la base de donnees 

tabCoeffboat = [
        {"_id":0      ,"name": "unknow",               "stamina": 1},
        {"_id":1      ,"name": "unknow",               "stamina": 1},
        {"_id":2      ,"name": "Figaro 3",             "stamina": 1},
        {"_id":3      ,"name": "Class 40 2021",        "stamina": 1},
        {"_id":4      ,"name": "Imoca",                "stamina": 1.2},
        {"_id":5      ,"name": "Mini 6.50",            "stamina": 1},
        {"_id":6      ,"name": "Ultim (Solo)",         "stamina": 1.5},
        {"_id":7      ,"name": "Volvo 65",             "stamina": 1.2},
        {"_id":8      ,"name": "unknow",               "stamina": 1},
        {"_id":9      ,"name": "Ultim (Crew)",         "stamina": 1.5},
        {"_id":10     ,"name": "Olympus",              "stamina": 1.5},
        {"_id":11     ,"name": "Ocean 50 (Multi 50)",  "stamina": 1},
        {"_id":12     ,"name": "unknow",               "stamina": 1},
        {"_id":13     ,"name": "Caravelle",            "stamina": 2},
        {"_id":14     ,"name": "Super Maxi 100",       "stamina": 1.5},
        {"_id":15     ,"name": "unknow",               "stamina": 1},
        {"_id":16     ,"name": "Tara",                 "stamina": 2},
        {"_id":17     ,"name": "unknow",               "stamina": 1},
        {"_id":18     ,"name": "OffShore Racer",       "stamina": 1},
        {"_id":19     ,"name": "Mod70",                "stamina": 1.2},
        {"_id":20     ,"name": "Cruiser Racer",        "stamina": 1.2},
        {"_id":21     ,"name": "Ultim BP XI",          "stamina": 1.5},
      ]





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





# def rechercheTablePolaires(polar_id):
# # il nous faut l id des polaires
#     staticbd='/home/jp/static/bd/basededonnees.db'
#     with sql.connect(staticbd) as conn:
#         cursor=conn.cursor()
#         cursor.execute("select polaires FROM polaires where _id=? " ,(polar_id,))
#         result = cursor.fetchone()
#         stringpolaires=result[0]
#         polairesjson=eval(stringpolaires)

#     return polairesjson


# polar_id=4
# timestamp,polar_id,polairesjsonstr= rechercheTablePolaires( polar_id)                       # nouvelle version
# polairesjson=json.loads(polairesjsonstr) 
# polairesjson=json.loads(polairesjson) 




def polaire_vect_twa(polaires,tabtwa, tabtws,TWS,TWAO):
    '''Retourne un tableau de polaires en fonction des polaires bateau  de TWS TWD et HDG'''
    '''TWS true Wind speed, TWD true wind direction , HDG caps'''
    '''Les trois tableaux doivent avoir la meme dimension'''
   
    TWA=np.abs(TWAO)
    TWS2=TWS.reshape((-1, 1))
    donnees=np.concatenate((TWA,TWS2),axis=1)
    valeurs = interpn((tabtwa, tabtws), polaires, donnees, method='linear')


    return valeurs

# def polaire_vect(polaires,tab_twa, tab_tws,TWS,TWD,HDG):
#     '''Retourne un tableau de polaires en fonction des polaires bateau  de TWS TWD et HDG'''
#     '''TWS true Wind speed, TWD true wind direction , HDG caps'''
#     '''Les trois tableaux doivent avoir la meme dimension'''
   
#     TWA=(180 - np.abs(((360 - TWD + HDG) % 360) - 180)).reshape((-1, 1))
#     TWS2=TWS.reshape((-1, 1))
#     donnees=np.concatenate((TWA,TWS2),axis=1)
#     valeurs = interpn((tab_twa, tab_tws), polaires, donnees, method='linear')
#     return valeurs




def foil(twa,tws,speedRatio,twaMin,twaMax,twaMerge,twsMin,twsMax,twsMerge):
    '''calcule le coeff des foils en fonction de la twa et tws'''
    if ((twa>twaMin-twaMerge)and(twa<twaMax+twaMerge)and(tws>twsMin-twsMerge)and(tws<twsMax+twsMerge)):
        if (twa>twaMin-twaMerge) and (twa<(twaMin)):
            coeff1=(twa-twaMin+twaMerge)/twaMerge
        else :
            coeff1=1
        if (twa>(twaMax)) and (twa<(twaMax+twaMerge)):
            coeff2=(twaMax+twaMerge-twa)/twaMerge
            # print(' en 10 coeff2', coeff2)
        else :
            coeff2=1  
            
        if (tws>twsMin-twsMerge) and (tws<(twsMin)):
            coeff3=(tws-twsMin+twsMerge)/twsMerge
        else :
            coeff3=1  
        if (tws>(twsMax)) and (tws<(twsMax+twsMerge)):
            coeff4=(twsMax+twsMerge-tws)/twsMerge
        else :
            coeff4=1  

        # print ('coeffs',coeff1,coeff2,coeff3,coeff4)    
        coeff=1+(speedRatio-1)*coeff1*coeff2*coeff3*coeff4
    else :
        coeff=1    
    #print ('Coeff  foils : ',coeff )
    return coeff





def stringPolairestoNpy(polairesjson):
    # A partir de la chaine chargee depuis le dash 
    # fabrique les fichiers polairesglobales polairesglobales10 tabvmg et tabvmg10 et sauve dans le repeertoire dirpolaires
    dirpolaires='/home/jp/staticLocal/npy/'
    # polairesjson=eval(stringpolaires)
    _id            = polairesjson['_id']
    tabtwsvr       = np.asarray(polairesjson['tws'])                                            
    tabtwavr       = np.asarray(polairesjson['twa'])
    nbtws          = len(tabtwsvr)
    nbtwa          = len(tabtwavr)
    bateau         = polairesjson['label']
    # coeffboat      = polairesjson['coeffboat']
    nbvoiles       = len(polairesjson['sail'])
    typevoile      = []
    print (polairesjson)
    print()
    print (bateau)
    print()
    toutespolaires = np.zeros((nbtwa,nbtws,nbvoiles))




    for i in range(nbvoiles) :
        typevoile.append( polairesjson['sail'][i]['name'])
        toutespolaires[:,:,i] = polairesjson['sail'][i]['speed']
        globalSpeedRatio      = polairesjson['globalSpeedRatio']
        speedRatio            = polairesjson['foil']['speedRatio']
        twaMin                = polairesjson['foil']['twaMin']
        twaMax                = polairesjson['foil']['twaMax']
        twaMerge              = polairesjson['foil']['twaMerge']
        twsMin                = polairesjson['foil']['twsMin']
        twsMax                = polairesjson['foil']['twsMax']
        twsMerge              = polairesjson['foil']['twsMerge']
        hull                  = polairesjson['hull']['speedRatio']
        lws                   = polairesjson['winch']['lws']
        hws                   = polairesjson['winch']['hws']
        lwtimer               = polairesjson['winch']['sailChange']['pro']['lw']['timer']
        hwtimer               = polairesjson['winch']['sailChange']['pro']['hw']['timer']
        lwratio               = polairesjson['winch']['sailChange']['pro']['lw']['ratio']
        hwratio               = polairesjson['winch']['sailChange']['pro']['hw']['ratio']
        tackprolwtimer        = polairesjson['winch']['tack']['pro']['lw']['timer']
        tackprolwratio        = polairesjson['winch']['tack']['pro']['lw']['ratio']
        tackprohwtimer        = polairesjson['winch']['tack']['pro']['hw']['timer']
        tackprohwratio        = polairesjson['winch']['tack']['pro']['hw']['ratio']
        gybeprolwtimer        = polairesjson['winch']['gybe']['pro']['lw']['timer']
        gybeprolwratio        = polairesjson['winch']['gybe']['pro']['lw']['ratio']
        gybeprohwtimer        = polairesjson['winch']['gybe']['pro']['hw']['timer']
        gybeprohwratio        = polairesjson['winch']['gybe']['pro']['hw']['ratio']
        polairesmax           = np.amax(toutespolaires,axis=2) 

    # fabrication du tableau des polaires     
    polairesunit10   = np.float32(np.zeros((701,181)))
    polairesunit10ttv= np.float32(np.zeros((7,701,181)))   
    tabfoils         = np.float32(np.zeros((701,181)))

    for j in range (7):                                 # on calcule les vitesses pour chaque voile
        for i in range (701):                           # creation de polaireunit a laide de polairevecttwa
                polairesunit10[i]=globalSpeedRatio*polaire_vect_twa(toutespolaires[:,:,j],tabtwavr,tabtwsvr,np.ones(181)*i/10,np.arange(0,181).reshape(-1,1))
        polairesunit10ttv[j]=polairesunit10

        for i in range (701):                                    # Utilisation d un tableau pour les coeffs foils 
            for j in range (181):
                tabfoils[i,j]=foil(j,i/10,speedRatio,twaMin,twaMax,twaMerge,twsMin,twsMax,twsMerge)

    # calcul des vitesses  avec foils des vitesses max et des voiles          
    polairesmaxttv    = np.amax(polairesunit10ttv,axis=0) *tabfoils*1.003  
    polairesttv       = polairesunit10ttv*tabfoils*1.003  
    voiles            = np.argmax(polairesunit10ttv,axis=0)  

    #on regroupe les 3 dans un tableau unique 
    polairesglobales=np.float32(np.zeros((9,701,181)))    
    polairesglobales[0:7,:,:] = polairesttv
    polairesglobales[7,:,:]   = polairesmaxttv
    polairesglobales[8,:,:]   = voiles

    # Sauvegarde et chargement dans un fichier 
    filenamelocal1='polairesglobales_'+str(_id)+'.npy'   
    filename1=dirpolaires+filenamelocal1 
    with open(filename1,'wb')as f: 
            np.save (f,polairesglobales)     




    # Maintenant on va calculer les polaires tous les 10eme de twa          

    # fabrication du tableau des polaires     
    polairesunit100   = np.float32(np.zeros((701,1801)))
    polairesunit100ttv= np.float32(np.zeros((7,701,1801)))   
    tabfoils         = np.float32(np.zeros((701,1801)))

    for j in range (7):                                 # on calcule les vitesses pour chaque voile
        for i in range (701):                           # creation de polaireunit a laide de polairevecttwa
                polairesunit100[i]=globalSpeedRatio*polaire_vect_twa(toutespolaires[:,:,j],tabtwavr,tabtwsvr,np.ones(1801)*i/10,(np.arange(0,1801)/10).reshape(-1,1))
        polairesunit100ttv[j]=polairesunit100

        for i in range (701):                                    # Utilisation d un tableau pour les coeffs foils 
            for j in range (1801):
                tabfoils[i,j]=foil(j/10,i/10,speedRatio,twaMin,twaMax,twaMerge,twsMin,twsMax,twsMerge)

    # calcul des vitesses  avec foils des vitesses max et des voiles          
    polairesmaxttv100 = np.amax(polairesunit100ttv,axis=0) *tabfoils*1.003  
    polairesttv100    = polairesunit100ttv*tabfoils*1.003  
    voiles            = np.argmax(polairesunit100ttv,axis=0)  
    polairesglobales10 = np.float32(np.zeros((9,701,1801)))  
    polairesglobales10[0:7,:,:]=polairesttv100
    polairesglobales10[7,:,:]=polairesmaxttv100
    polairesglobales10[8,:,:]= voiles  


    filenamelocal1='polairesglobales10_'+str(_id)+'.npy'
    filename1='/home/jp/staticLocal/npy/'+filenamelocal1
    with open(filename1,'wb')as f: 
            np.save (f,polairesglobales10) 
            
    # print ('sauvegarde de ',filename1)

    # meme si ce n'est que partiellement necessaire pour des raisons de compatibilite on va calculer les tabvmg"

    Twa=np.arange(180)
    cos=np.cos(Twa/180*math.pi).astype(np.float32)
    tabvmg=np.zeros((701,7),dtype=np.float32)            # on va constituer un tableau (tws,vmgmax,twavmgmax,vmgmin,twavmgmin,vmax,twavmax)
    for tws10 in range (701) :                           # on fait varier le vent de 0a 70 Noeuds
        Tws=(np.ones(len(Twa))*tws10).astype (int)       # on constitue une serie de vents identiques pour calculer pour chaque twa
        Vitesses = polairesglobales[7,Tws,Twa]
        Vmg=Vitesses*cos
        tabvmg[tws10,0]=tws10
        tabvmg[tws10,1]=np.max(Vmg)
        tabvmg[tws10,2]=np.argmax(Vmg)
        tabvmg[tws10,3]=np.min(Vmg)
        tabvmg[tws10,4]=np.argmin(Vmg)
        tabvmg[tws10,5]=np.max(Vitesses)
        tabvmg[tws10,6]=np.argmax(Vitesses)         
    filenamelocal2='vmg_'+str(_id)+'.npy'   
    filename2='/home/jp/staticLocal/npy/'+filenamelocal2 
    with open(filename2,'wb')as f: 
            np.save (f,tabvmg)   



# on va maintenant constituer le tableau des vmg10 pour chaque bateau          
    Twa10=np.arange(1800)
   
    cos=np.cos(Twa10/10/180*math.pi).astype(np.float32)
    tabvmg10=np.zeros((701,7),dtype=np.float32)            # on va constituer un tableau (tws,vmgmax,twavmgmax,vmgmin,twavmgmin,vmax,twavmax)
    for tws10 in range (701) :                           # on fait varier le vent de 0a 70 Noeuds
        Tws=(np.ones(len(Twa10))*tws10).astype (int)       # on constitue une serie de vents identiques pour calculer pour chaque twa
        Vitesses = polairesglobales10[7,Tws,Twa10]
        Vmg=Vitesses*cos
        tabvmg10[tws10,0]=tws10/10
        tabvmg10[tws10,1]=np.max(Vmg)
        tabvmg10[tws10,2]=np.argmax(Vmg)/10
        tabvmg10[tws10,3]=np.min(Vmg)
        tabvmg10[tws10,4]=np.argmin(Vmg)/10
        tabvmg10[tws10,5]=np.max(Vitesses)
        tabvmg10[tws10,6]=np.argmax(Vitesses)/10       

    filenamelocal2='vmg10_'+str(_id)+'.npy'   
    filename2='/home/jp/staticLocal/npy/'+ filenamelocal2
    with open(filename2,'wb')as f: 
        np.save (f,tabvmg10) 
    return None        


##################################################################################################""
################    Chargement de polaires json    ###############################################""
##################################################################################################""

for i in range (21):
    try: 
        polar_id=i
        timestamp,polar_id,polairesjsonstr= rechercheTablePolaires( polar_id)                       # nouvelle version
        polairesjson=json.loads(polairesjsonstr) 
        polairesjson=json.loads(polairesjson) 

        #print polairesjson 
        bateau         = polairesjson['label']
        lws                   = polairesjson['winch']['lws']
        hws                   = polairesjson['winch']['hws']
        lwtimer               = polairesjson['winch']['sailChange']['pro']['lw']['timer']
        hwtimer               = polairesjson['winch']['sailChange']['pro']['hw']['timer']
        lwratio               = polairesjson['winch']['sailChange']['pro']['lw']['ratio']
        hwratio               = polairesjson['winch']['sailChange']['pro']['hw']['ratio']
        tackprolwtimer        = polairesjson['winch']['tack']['pro']['lw']['timer']
        tackprolwratio        = polairesjson['winch']['tack']['pro']['lw']['ratio']
        tackprohwtimer        = polairesjson['winch']['tack']['pro']['hw']['timer']
        gybeprolwtimer        = polairesjson['winch']['gybe']['pro']['lw']['timer']
        gybeprolwratio        = polairesjson['winch']['gybe']['pro']['lw']['ratio']
        gybeprohwtimer        = polairesjson['winch']['gybe']['pro']['hw']['timer']


        print ('polar_id {}\t bateau \t{} \t lws {} hws {}  lwtimer {},hwtimer {},lwratio {},hwratio {},tackprolwtimer {},tackprohwtimer {},gybeprolwtimer {},gybeprohwtimer {},gybeprolwratio {}'.format (polar_id,bateau,lws,hws,lwtimer,hwtimer,lwratio,hwratio,tackprolwtimer,tackprohwtimer,gybeprolwtimer,gybeprohwtimer,gybeprolwratio) )

# Zone commentee 
###############################################""

        # print (polairesjson)

        # #############  Transformation en npy #################################################################

        # stringPolairestoNpy(polairesjson)



        # ##################################################################################################""
        # ################    Test des valeurs               ###############################################""
        # ##################################################################################################""e

        # # chargement des valeurs 
        # basedirnpy='/home/jp/staticLocal/npy/'
        # filenamelocal1='polairesglobales10_'+str(polar_id)+'.npy'
        # filename1=basedirnpy+filenamelocal1
        # with open(filename1,'rb')as f:
        #         polairesglobales10 = np.load(f)
            
        # filenamelocal2='vmg10_'+str(polar_id)+'.npy'
        # filename2=basedirnpy+filenamelocal2
        # with open(filename2,'rb')as f:
        #         tabvmg10 = np.load(f)   




        # print ('\n******************************************************************')
        # print ('Test sur polaires ',polairesjson['label'])
        # twa=55
        # tws=12.1
        # typeVoiles = ['jib', 'Spi', 'Staysail', 'LightJib', 'Code0', 'HeavyGnk', 'LightGnk']
        # voile=typeVoiles[int(polairesglobales10[8,int(tws*10),int(twa*10)])]
        # print ('pour twa= {} tws= {} voile {} vitesse = {:6.3f} '.format(twa,tws,voile,polairesglobales10[7,int(tws*10),int(twa*10)]))
        # print()
        # for i in range (len (typeVoiles)):
        #     voile=typeVoiles[i]
        #     print ('pour twa= {} tws= {} voile {:15} vitesse = {:6.3f} '.format(twa,tws,voile,polairesglobales10[i,int(tws*10),int(twa*10)]))




        # print ('valeur attendue pour imoca foils :  13.45,13.044,1.170,9.870,13.450,12.203,0,0')
        # print ('******************************************************************\n')

        # print ('Test sur Vmg ')
        # tws=12.1
        # tabvmg       = tabvmg10[int(tws*10)]
        # print('tabvmg \n',tabvmg)
        # print('valeurs attendues tws 12.1 ,vmb pres  8.825478  angle 45 vmg vent arriere -10.182713 angle : 140          vmax 15.680747 angle 105 ')

# Zone commentee 
###############################################""

    except:
        None


# refbateaux=[3,4,5,6,7,9,10,11,13,14,16,18,19,20,21]
# for i in range (len(refbateaux)):
#     try:
#         print('calcul des polaires pour le bateau ',refbateaux[i])
        
#         res=rechercheTablePolaires(refbateaux[i])



#         print(res)
#         stringpolaires=str(res)
#         stringPolairestoNpy(stringpolaires)
#     except:
#          print('le bateau n est pas repertorié dans la base de donnees ',refbateaux[i])