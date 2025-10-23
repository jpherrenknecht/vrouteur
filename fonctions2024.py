
import time
import os
import math
from datetime import datetime,timezone
import numpy as np
import json
import sqlite3 as sql
import zlib
import requests
import folium
from urllib.request import urlretrieve
from scipy.interpolate import RegularGridInterpolator,interp2d,interpn
from shapely.geometry import (LineString, MultiLineString, MultiPoint, Point,Polygon)
from shapely.ops import unary_union
from shapely.prepared import prep
from joblib import Parallel,delayed




leftlon, rightlon, toplat, bottomlat = 0, 360, 90, -90
basedir = os.path.abspath(os.path.dirname("__file__"))
#34     1) Fonctions de gestion
#110    2) Fonctions de navigation
#552    3) Fonctions de gribs
#897     Fonctions d optimisation
#990     Fonctions de cartographie
# 6) Fonctions d impression


# modification du 7 01 2025  Correction dans le fichier polairesglobales10 "
##########################################################################################################
# 1)Fonctions de gestion
##########################################################################################################

def recherche (twa,tab_twa):
    ''' recherche l'indice d'une twa ou tws dans le tableau '''
    k=0
    while (twa > tab_twa[k]):
        k+=1
    return k


def recherchecourse(course,tabcoursescomplet):
    for i in range (len(tabcoursescomplet)):
        if tabcoursescomplet[i][0]== course:
            break
    course           = tabcoursescomplet[i][0]
    caracteristiques = tabcoursescomplet[i][1] 
    tabmarques       = tabcoursescomplet[i][2] 
    polairesjson     = tabcoursescomplet[i][3]  
    polairesnp       = tabcoursescomplet[i][4]   # polaires numpy fullpack et foils
    polairesjs       = tabcoursescomplet[i][5]
    toutespolaires   = tabcoursescomplet[i][6]
    tabtwsvr         = tabcoursescomplet[i][7] 
    tabtwavr         = tabcoursescomplet[i][8] 
    tabvmg           = tabcoursescomplet[i][9]
    polairesunit     = tabcoursescomplet[i][10]
    exclusions       = tabcoursescomplet[i][11]
    return course,caracteristiques,tabmarques,polairesjson,polairesnp,polairesjs,toutespolaires,tabtwsvr,tabtwavr,tabvmg,polairesunit,exclusions


def retourVR(race):
    ''' donne les valeurs issues de Vr pour la course en utilisant la requete ajax''' 
    response2=requests.get('http://vrouteur.ddns.net/api/ajax2')                             # recuperation des donnees VR par dashmap
    #print('Statut de la réponse :',response2.status_code)
    if response2.status_code==200 :
        resultat=response2.content
        resul=json.loads(resultat.decode("utf-8"))
        #print(resul)
        y0vr=float(resul['lat'])
        x0vr=float(resul['lon'])
        staminavr=float(resul['stamina'])
        sailvr=int(resul['sail'])
        race2=   resul['race']
        capvr=float(resul['heading'])

        if int(race2)==int(float(race)) :
            #print('c est la bonne course', race2)
            try : 
                #Donnéees non définies si avant le depart
                twsvr=float(resul['tws'])
                twdvr=float(resul['twd'])
                twavr=float(resul['twa'])
                
                lastCalcDate=float(resul['lastCalcDate'])
                tp=lastCalcDate/1000
                #  print ('\nLastCalcDate {}\n**************************'.format(time.strftime(" %d %b %H:%M ",time.localtime(lastCalcDate/1000))))
                temps='LastCalcDate'
               # print()
                print("############################################################################################################")      
                print  ('Donnéees VR     tp,y0,x0= {:10.0f}, {:6.4f} , {:6.4f}     A {}     Twsvr {:6.3f} Twdvr {:6.3f} '\
                        .format(tp,y0vr,x0vr,time.strftime("%Hh:%Mmn:%Ss",time.localtime(lastCalcDate/1000)),twsvr,twdvr))
                print("############################################################################################################")   
    #                 #" %d %b %H:%M "   

            except:
                print('La recuperation n a pas marche ')
                tp=time.time()
        else :
            tp=time.time()
    else :
            tp=time.time()
            y0vr=0
            x0vr=0

    return tp,y0vr,x0vr
##########################################################################################################
# 2) Fonctions de navigation
##########################################################################################################


def vit_polaires(polairesunit,twss,twaos):
    '''polairesunit est le tableau des polaires mis a 1 par 1 '''
    ''' TWS le tableau des vitesses de vent '''
    ''' TWA le tableau des TWA eventuellement orientees'''
    twas=np.abs(np.ravel(twaos))
    twasi=twas.astype(int)
    dtwas=twas%1
    twssi=twss.astype(int)
    dtwss=twss%1
    vit00=polairesunit[twssi,twasi]
    vit10=polairesunit[twssi+1,twasi]
    vit01=polairesunit[twssi,(twasi+1)%180]
    vit11=polairesunit[twssi+1,(twasi+1)%180]
    vit=vit00+(vit01-vit00)*dtwas +(vit10-vit00)*dtwss  +(vit11+vit00-vit10-vit01)*dtwas*dtwss   #interpolation bilineaire
    return vit 




def calcul_cap(lat0, lon0, lat1, lon1):
    """
    Calcule le cap en degrés du point (lat0, lon0) vers le point (lat1, lon1).
    """
    # Conversion en radians
    lat0, lon0, lat1, lon1 = map(math.radians, [lat0, lon0, lat1, lon1])
    
    delta_lon = lon1 - lon0
    
    # Formule du cap initial
    y = math.sin(delta_lon) * math.cos(lat1)
    x = math.cos(lat0) * math.sin(lat1) - math.sin(lat0) * math.cos(lat1) * math.cos(delta_lon)
    cap = math.atan2(y, x)
    
    # Conversion en degrés et normalisation entre 0 et 360
    cap = (math.degrees(cap) + 360) % 360
    return cap



def distance_haversine(lat0, lon0, lat1, lon1):
    """
    Calcule la distance entre deux points en utilisant la formule de Haversine.
    """
    R = 6371000  # Rayon de la Terre en mètres
    lat0, lon0, lat1, lon1 = map(math.radians, [lat0, lon0, lat1, lon1])
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    a = math.sin(dlat / 2) ** 2 + math.cos(lat0) * math.cos(lat1) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_m= R * c

    return distance_m / 1852  # Conversion en miles nautiques



def calcul_cap_loxodromique(lat0, lon0, lat1, lon1):
    """
    Calcule le cap constant (loxodromique) entre deux points.
    Les latitudes et longitudes sont en degrés.
    """
    # Conversion en radians
    lat0, lon0, lat1, lon1 = map(math.radians, [lat0, lon0, lat1, lon1])

    delta_lon = lon1 - lon0
    delta_phi = math.log(math.tan(math.pi / 4 + lat1 / 2) / math.tan(math.pi / 4 + lat0 / 2))

    # Gestion du cas où delta_phi est proche de 0 (évite une division par zéro)
    if abs(delta_phi) < 1e-10:
        cap = math.atan2(delta_lon, math.cos(lat0))
    else:
        cap = math.atan2(delta_lon, delta_phi)

    # Conversion en degrés et normalisation entre 0 et 360
    cap = (math.degrees(cap) + 360) % 360
    return cap




def ftwa(cap, dvent):
    '''retourne la valeur absolue de la twa'''
    twa = 180 - abs(((360 - dvent + cap) % 360) - 180)
    return twa     


def ftwaos(cap, dvent):
    '''twa orientee simple avec des valeurs float'''
    twa=(cap-dvent+360)%360
    if twa<180:
        twa=-twa
    else:
        twa=360-twa
    return twa   


def ftwao(HDG,TWD):
    '''Twa orientee pour des np.array'''
    return np.mod((TWD-HDG+540),360)-180


def ftwato(HDG,TWD):
    '''Twa orientee pour des float ou torch mais pas np '''
    return ((TWD-HDG+540)%360)-180


def fcap(twao,twd):
    ''' retourne le cap en fonction de la twaorientée et de la direction du vent '''
    cap=(360+twd-twao)%360
    return cap







def cap_dep_ar(ydep,xdep,yar,xar):
    '''donne le cap en degres du point de depart vers le point d arrivee definis par leur coords'''
    cap=math.atan2(xar-xdep,(yar-ydep)/(math.cos(ydep*math.pi/180)))*180/math.pi
    cap=(cap+360)%360
    return cap

def dist(y0, x0, y1, x1):
    # Calcule la distance entre 2 points en miles nautique 
    R = 3440.065    #rayon de la terre 
    # Convertir les degrés en radians
    lat1 = math.radians(y0)
    lon1 = math.radians(x0)
    lat2 = math.radians(y1)
    lon2 = math.radians(x1)
    # Calculer les différences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Calculer la distance en milles nautiques
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def distnp(y0, x0, y1, x1):
    # Calcule la distance entre 2 points en miles nautiques 
    R = 3440.065  # Rayon de la Terre

    # Convertir les degrés en radians
    lat1 = np.radians(y0)
    lon1 = np.radians(x0)
    lat2 = np.radians(y1)
    lon2 = np.radians(x1)

    # Calculer les différences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculer la distance en milles nautiques
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance

def polaire_simple2(twa,  tws, polairesunit):
    dtwa=twa%1
    dtws=tws%1
    twai=int(twa)
    twsi=int(tws)
   
    vit00=polairesunit[twsi,twai]
    vit10=polairesunit[twsi+1,twai]
    vit01=polairesunit[twsi,(twai+1)%180]
    vit11=polairesunit[twsi+1,(twai+1)%180]
    vit=vit00+(vit01-vit00)*dtwa +(vit10-vit00)*dtws  +(vit11+vit00-vit10-vit01)*dtwa*dtws
    return vit




def polaire_vect_twa(polaires,tabtwa, tabtws,TWS,TWAO):
    '''Retourne un tableau de polaires en fonction des polaires bateau  de TWS TWD et HDG'''
    '''TWS true Wind speed, TWD true wind direction , HDG caps'''
    '''Les trois tableaux doivent avoir la meme dimension'''
   
    TWA=np.abs(TWAO)
    TWS2=TWS.reshape((-1, 1))
    donnees=np.concatenate((TWA,TWS2),axis=1)
    valeurs = interpn((tabtwa, tabtws), polaires, donnees, method='linear')
    return valeurs

def polaire_vect(polaires,tab_twa, tab_tws,TWS,TWD,HDG):
    '''Retourne un tableau de polaires en fonction des polaires bateau  de TWS TWD et HDG'''
    '''TWS true Wind speed, TWD true wind direction , HDG caps'''
    '''Les trois tableaux doivent avoir la meme dimension'''
   
    TWA=(180 - np.abs(((360 - TWD + HDG) % 360) - 180)).reshape((-1, 1))
    TWS2=TWS.reshape((-1, 1))
    donnees=np.concatenate((TWA,TWS2),axis=1)
    valeurs = interpn((tab_twa, tab_tws), polaires, donnees, method='linear')
    return valeurs

def opti3(twa, tws,tab_twa , tab_tws  ,polaires):
    ''' Optimise la twa lorsqu elle est autour des valeurs de vmgmax'''
    ''' en entree une twa et une tws'''
    ''' sortie la twa optimisee si la twa est a moins de 2 ° de l'optimum   ---'''
    # retourne les valeurs caracteristiques pour un vent donné 
    # donne les valeurs caracteristiques pour une  vitesse de vent donné
    twamax,vmgmax,twamin,vmgmin,twaspeedmax,speedmax=vmgmaxspeed(tws,tab_twa , tab_tws  ,polaires)
    signe=twa/(abs(twa)+0.0001)
    if abs(twa)-twamax<2:                  # 
        twar=round((twamax*signe),0)
    elif abs(abs(twa)-twamin)<2:
        twar=round((twamin*signe),0)
   
    elif   abs(abs(twa)-twaspeedmax)<2: 
        twar=round((twaspeedmax*signe),1)
    #bannissement des valeurs hors laylines   
    elif abs(twa)<twamax:
        twar=twamax*signe
    elif abs(twa)>twamin:
        twar=round((twamin*signe),1)
    else:
        twar=twa  
    twar=round(twar,0)    
    return twar    


def opti_twa(tabtws,tabtwa,tab_twa,tab_tws,polaires)  : 
    ''' on fournit en entree   le tableau des tws et des twa  pour calcul  '''
    ''' en sortie on obtient les twa optimisee et arrondies au degre pour les valeus proches des optimums au pres au portant et au vent arriere'''
    twaopti=np.zeros(len (tabtwa))
    for i in range (len (tabtwa)):
        tws=tabtws[i]
        twa=tabtwa[i]
        twaopti[i]=opti3(tabtwa[i],tabtws[i],tab_twa,tab_tws,polaires)
    twaopti=twaopti.reshape(-1,1)
    return twaopti


def vit_polaires10 (polairesunit10,twss,twaos):
    twss  =np.around(twss*10,0).astype(int)
    twaos =np.around(np.abs(twaos*10),0).astype(int)
    return polairesunit10[twss,twaos]


def opti2(twa, tws,tab_twa , tab_tws  ,polaires):
    '''Optimise la twa lorsqu elle est autour des valeurs de vmgmax'''
    ''' entree une twa et une tws'''
    ''' sortie la twa optimisee si pas proche de 3  ---'''
    twamax,vmgmax,twamin,vmgmin,twaspeedmax,speedmax=vmgmaxspeed(tws,tab_twa , tab_tws  ,polaires)
    #print (twamax,vmgmax,twamin,vmgmin,twaspeedmax,speedmax)
    signe=twa/(abs(twa)+0.0001)
    if abs(twa)-twamax<3:
        twar=round((twamax*signe),1)
    elif abs(abs(twa)-twamin)<3:
        twar=round((twamin*signe),1)

    elif   abs(abs(twa)-twaspeedmax)<2: 
        twar=round((twaspeedmax*signe),1)
    #bannissement des valeurs hors laylines    
    elif abs(twa)<twamax:
        twar=twamax*signe
    elif abs(twa)>twamin:
        twar=round((twamin*signe),1)
    else:
        twar=0     
    return twar    


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

def stringPolairestoNpy(stringpolaires):
    # A partir de la chaine chargee depuis le dash 
    # fabrique les fichiers polairesglobales polairesglobales10 tabvmg et tabvmg10 

    polairesjson=eval(stringpolaires)
    _id            = polairesjson['_id']
    tabtwsvr       = np.asarray(polairesjson['tws'])                                            
    tabtwavr       = np.asarray(polairesjson['twa'])
    nbtws          = len(tabtwsvr)
    nbtwa          = len(tabtwavr)
    bateau         = polairesjson['label']
    # coeffboat      = polairesjson['coeffboat']
    nbvoiles       = len(polairesjson['sail'])
    typevoile      = []

    toutespolaires = np.zeros((nbtwa,nbtws,nbvoiles))
    for i in range(nbvoiles) :
        typevoile.append( polairesjson['sail'][i]['name'])
        toutespolaires[:,:,i] = polairesjson['sail'][i]['speed']
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
                polairesunit10[i]=polaire_vect_twa(toutespolaires[:,:,j],tabtwavr,tabtwsvr,np.ones(181)*i/10,np.arange(0,181).reshape(-1,1))
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
    filename1='/home/jp/static/npy/'+filenamelocal1 
    with open(filename1,'wb')as f: 
            np.save (f,polairesglobales)     




    # Maintenant on va calculer les polaires tous les 10eme de twa          

    # fabrication du tableau des polaires     
    polairesunit100   = np.float32(np.zeros((701,1801)))
    polairesunit100ttv= np.float32(np.zeros((7,701,1801)))   
    tabfoils         = np.float32(np.zeros((701,1801)))

    for j in range (7):                                 # on calcule les vitesses pour chaque voile
        for i in range (701):                           # creation de polaireunit a laide de polairevecttwa
                polairesunit100[i]=polaire_vect_twa(toutespolaires[:,:,j],tabtwavr,tabtwsvr,np.ones(1801)*i/10,(np.arange(0,1801)/10).reshape(-1,1))
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
    filename1='/home/jp/static/npy/'+filenamelocal1
    with open(filename1,'wb')as f: 
            np.save (f,polairesglobales10) 
            
    # print ('sauvegarde de ',filename1)

    # meme si ce n'est que partiellement necessaire pour des raisons de compatibilite on va calculer les tabvmg"

    Twa=np.arange(180)
    cosi=np.cos(Twa/180*math.pi).astype(np.float32)
    tabvmg=np.zeros((701,7),dtype=np.float32)            # on va constituer un tableau (tws,vmgmax,twavmgmax,vmgmin,twavmgmin,vmax,twavmax)
    for tws10 in range (701) :                           # on fait varier le vent de 0a 70 Noeuds
        Tws=(np.ones(len(Twa))*tws10).astype (int)       # on constitue une serie de vents identiques pour calculer pour chaque twa
        Vitesses = polairesglobales[7,Tws,Twa]
        Vmg=Vitesses*cosi
        tabvmg[tws10,0]=tws10
        tabvmg[tws10,1]=np.max(Vmg)
        tabvmg[tws10,2]=np.argmax(Vmg)
        tabvmg[tws10,3]=np.min(Vmg)
        tabvmg[tws10,4]=np.argmin(Vmg)
        tabvmg[tws10,5]=np.max(Vitesses)
        tabvmg[tws10,6]=np.argmax(Vitesses)         
    filenamelocal2='vmg_'+str(_id)+'.npy'   
    filename2='/home/jp/static/npy/'+filenamelocal2 
    with open(filename2,'wb')as f: 
            np.save (f,tabvmg)   



# on va maintenant constituer le tableau des vmg10 pour chaque bateau          
    Twa10=np.arange(1800)
   
    cosi=np.cos(Twa10/10/180*math.pi).astype(np.float32)
    tabvmg10=np.zeros((701,7),dtype=np.float32)            # on va constituer un tableau (tws,vmgmax,twavmgmax,vmgmin,twavmgmin,vmax,twavmax)
    for tws10 in range (701) :                           # on fait varier le vent de 0a 70 Noeuds
        Tws=(np.ones(len(Twa10))*tws10).astype (int)       # on constitue une serie de vents identiques pour calculer pour chaque twa
        Vitesses = polairesglobales10[7,Tws,Twa10]
        Vmg=Vitesses*cosi
        tabvmg10[tws10,0]=tws10/10
        tabvmg10[tws10,1]=np.max(Vmg)
        tabvmg10[tws10,2]=np.argmax(Vmg)/10
        tabvmg10[tws10,3]=np.min(Vmg)
        tabvmg10[tws10,4]=np.argmin(Vmg)/10
        tabvmg10[tws10,5]=np.max(Vitesses)
        tabvmg10[tws10,6]=np.argmax(Vitesses)/10       

    filenamelocal2='vmg10_'+str(_id)+'.npy'   
    filename2='/home/jp/static/npy/'+ filenamelocal2
    with open(filename2,'wb')as f: 
        np.save (f,tabvmg10) 
    return None        








def chgtgrib(Tiso):
    liste=[4,10,16,22]                        # heure des changements de grib en UTC
    dateTuple=time.gmtime(Tiso)
    return  ((dateTuple[3]in liste) and (dateTuple[4]<10))
 



##########################################################################################################
# 3)Fonctions de gribs et previsions
##########################################################################################################
def gribFileName(basedir):
    ''' cherche le dernier grib complet disponible au temps en secondes '''
    ''' temps_secondes est par defaut le temps instantané '''
    ''' Cherche egalement le dernier grib chargeable partiellement'''
    ''' Change le nom du fichier a 48 '''

    temps_secondes=time.time()
    date_tuple       = time.gmtime(temps_secondes) 
    date_formatcourt = time.strftime("%Y%m%d", time.gmtime(temps_secondes))
    dateveille_tuple = time.gmtime(temps_secondes-86400) 
    dateveille_formatcourt=time.strftime("%Y%m%d", time.gmtime(temps_secondes-86400))
    mn_jour_utc =date_tuple[3]*60+date_tuple[4]
   
    if (mn_jour_utc <3*60+48):                          #avant 3h 48 UTC le nom de fichier est 18 h de la veille 
        filename=basedir+"gfs_"+dateveille_formatcourt+"-18.npy"
    elif (mn_jour_utc<9*60+48):   
        filename=basedir+"gfs_"+date_formatcourt+"-00.npy"
    elif (mn_jour_utc<15*60+48): 
        filename=basedir+"gfs_"+date_formatcourt+"-06.npy"
    elif (mn_jour_utc<21*60+48):   
        filename=basedir+"gfs_"+date_formatcourt+"-12.npy"
    else:                                              # entre 21h 48UTC  et minuit    
        filename=basedir+"gfs_"+date_formatcourt+"-18.npy" 
    date,heure,tig =dateheure(filename) 
    return filename,tig  





def fileNames025(basedir):
    '''A priori plus necessaire avec nouvelle version'''
    temps_secondes   = time.time()
    date_tuple       = time.gmtime(temps_secondes) 
    date_formatcourt = time.strftime("%Y%m%d", time.gmtime(temps_secondes))
    dateveille_tuple = time.gmtime(temps_secondes-86400) 
    dateveille_formatcourt=time.strftime("%Y%m%d", time.gmtime(temps_secondes-86400))
    mn_jour_utc =date_tuple[3]*60+date_tuple[4]
    ''' cherche le dernier grib  disponible pour chargement au temps en secondes '''
    if (mn_jour_utc <230):                       #correspond à 00 pas encore dispo
        print('Dernier grib disponible : veille 18h')
        #filenameA=basedir+"/gfs_"+dateveille_formatcourt+"-12.npy"
        fileName=basedir+"gfs_"+dateveille_formatcourt+"-18.npy"
    elif  (mn_jour_utc >230)&(mn_jour_utc<590):  # debut de la dispo du 00
        print ('Dernier Grib disponible  jour 0h')
        #filenameA=basedir+"/gfs_"+dateveille_formatcourt+"-18.npy"
        fileName=basedir+"gfs_"+date_formatcourt+"-00.npy"
    elif (mn_jour_utc>590)&(mn_jour_utc<950): 
        print ('Dernier grib disponible  6h')
        #filenameA=basedir+"/gfs_"+date_formatcourt+"-00.npy"
        fileName=basedir+"gfs_"+date_formatcourt+"-06.npy"      
    elif (mn_jour_utc>950)&(mn_jour_utc<1310):  
        print ('Dernier grib disponible 12h') 
        #filenameA=basedir+"/gfs_"+date_formatcourt+"-06.npy"
        fileName=basedir+"gfs_"+date_formatcourt+"-12.npy"       
    else:
        print ('Dernier grib disponible 18h')   
        #filenameA=basedir+"/gfs_"+date_formatcourt+"-12.npy"
        fileName=basedir+"gfs_"+date_formatcourt+"-18.npy"
    return fileName




def dateheure(filename):
    '''retourne la date et heure du fichier grib a partir du nom'''
    ''' necessaire pour charger le NOAA'''
    tic=time.time()
    ticstruct = time.localtime()
    utc = time.gmtime()
    decalage = ticstruct[3] - utc[3]
    x     = filename.split('.')[0]
    x     = x.split('/')[-1]
    heure = x.split('-')[1]
    date  = (x.split('-')[0]).split('_')[1]
    year  = int(date[0:4])
    month = int(date[4:6])
    day   = int(date[6:8])
    tigt=datetime(year,month,day,int(heure),0, 0)
    tig=time.mktime(tigt.timetuple()) +decalage*3600 # en secondes UTC
    return date,heure,tig 




def polaire_vect_twa(polaires,tabtwa, tabtws,TWS,TWAO):
    '''Retourne un tableau de polaires en fonction des polaires bateau  de TWS TWD et HDG'''
    '''TWS true Wind speed, TWD true wind direction , HDG caps'''
    '''Les trois tableaux doivent avoir la meme dimension'''
   
    TWA=np.abs(TWAO)
    TWS2=TWS.reshape((-1, 1))
    donnees=np.concatenate((TWA,TWS2),axis=1)
    valeurs = interpn((tabtwa, tabtws), polaires, donnees, method='linear')
    return valeurs



def fun(deg): return abs(math.sin(deg/180 * math.pi))
def uv2s(u,v): return(u*u+v*v)**.5
def uv2d(u,v): return math.atan2(u,v)*180/math.pi+180


def previsionzezo32(GR,  tp, latitude, longitude):
    ''' Barbare mais  1/20eme du temps de regulargridinterpolator !! '''
    ''' Prevision6 integre le calcul de type zezo'''

    ''' utilise pour test  '''
    tig=GR[0,0,0,0]*100
    itemp = (tp - tig) / 3600 / 3
    ilati = 90-latitude
    ilong = (longitude) % 360
    iitemp=math.floor(itemp)
    iilati=math.floor(ilati)
    iilong=math.floor(ilong)

    #print ('Dans prevision6test32 indices',itemp,ilati,ilong)
    ditemp=itemp%1
    dilati=ilati%1
    dilong=ilong%1
# on va arrondir le temps par fraction de 10mn
    fraction=math.floor(ditemp*18)/18
    #print( 'Dans prevision 6test v000',v000)          
    v000=(GR[iitemp   ,iilati         ,iilong        , 0]      + GR[iitemp   ,  iilati      , iilong         , 1]*1j ) *3.6   
    v010=(GR[iitemp   ,(iilati+1)%180 ,iilong        , 0]      + GR[iitemp   , (iilati+1 )%180, iilong         , 1]*1j) *3.6
    v001=(GR[iitemp   ,iilati         ,(iilong+1)%360, 0]      + GR[iitemp   , iilati         , (iilong+1)%360 , 1]*1j) *3.6
    v011=(GR[iitemp   ,(iilati+1)%180 ,(iilong+1)%360, 0]      + GR[iitemp   , (iilati+1)%180 , (iilong+1)%360 , 1]*1j) *3.6
    v100=(GR[iitemp+1 ,iilati         ,iilong        , 0]      + GR[iitemp+1 , iilati         , iilong         , 1]*1j) *3.6
    v110=(GR[iitemp+1 ,(iilati+1)%180 ,iilong        , 0]      + GR[iitemp+1 , (iilati+1)%180 , iilong         , 1]*1j) *3.6
    v101=(GR[iitemp+1 ,iilati         ,(iilong+1)%360, 0]      + GR[iitemp+1 , iilati         , (iilong+1)%360 , 1]*1j) *3.6
    v111=(GR[iitemp+1 ,(iilati+1)%180 ,(iilong+1)%360, 0]      + GR[iitemp+1 , (iilati+1)%180 , (iilong+1)%360 , 1]*1j) *3.6
   
    #print('interpolation temporelle')
    vx00=v000+fraction*(v100-v000)
    #print('ligne 245 vx00',vx00)
    vx10=v010+fraction*(v110-v010)
    vx01=v001+fraction*(v101-v001)
    vx11=v011+fraction*(v111-v011)
    u1=vx10.real
    u2=vx11.real
    u3=vx00.real
    u4=vx01.real
    
    v1=vx10.imag
    v2=vx11.imag
    v3=vx00.imag
    v4=vx01.imag
   
    # on calcule l angle aux 4 points
    d1 =(270 - np.angle(vx10, deg=True)) % 360
    d2 =(270 - np.angle(vx11, deg=True)) % 360
    d3 =(270 - np.angle(vx00, deg=True)) % 360
    d4 =(270 - np.angle(vx01, deg=True)) % 360

   
    s1=uv2s(u1,v1)
    s2=uv2s(u2,v2)
    s3=uv2s(u3,v3)
    s4=uv2s(u4,v4)
 
    x=dilong
    y=dilati
    
    u=(u2+u3-u1-u4)*x*y +(u4-u3)*x +(u1-u3)*y + u3
    v=(v2+v3-v1-v4)*x*y +(v4-v3)*x +(v1-v3)*y + v3
    speed_s  = (s2+s3-s1-s4)*x*y +(s4-s3)*x +(s1-s3)*y + s3
    # speed_uv = uv2s(u,v)
    # angle_uv=uv2d(u,v)

    speed_uv,angle_uv=vitanglebrut(u+v*1j)

    cs_uv=uv2s((u1+u2+u3+u4)/4 ,(v1+v2+v3+v4)/4)
    cs_avg=(s1+s2+s3+s4)/4
    cs_ratio=cs_uv/cs_avg
    c=[0,0,0,0]
    if x<.5 :                           # on est cote gauche 
        if y<.5:  
            c[0]=abs(math.sin((d1-d2)/180 * math.pi))
            #c[0]=f(d1-d2)
            c[1]=cs_ratio
            c[2]=1
            c[3]= abs(math.sin((d3-d4)/180 * math.pi))   
        else:
            y-=0.5
            c[0]=1
            c[1]=fun(d1-d2)
            c[2]=fun(d1-d3)
            c[3]=cs_ratio
    else: 
        if y<.5: 
            x-=0.5
            c[0]=cs_ratio
            c[1]=fun(d2-d4)
            c[2]=fun(d3-d4)
            c[3]=1
        else:   
            y-=0.5
            c[0]=fun(d1-d2)
            c[1]=1
            c[2]=cs_ratio
            c[3]=fun(d2-d4)  
    x*=2
    y*=2
    c_coeff=(c[1]+c[2]-c[0]-c[3])*x*y +(c[3]-c[2])*x +(c[0]-c[2])*y +c[2]
    s_coeff= (speed_uv/speed_s)**(1-c_coeff**.7)
    speed_s *=s_coeff
    speed=speed_s/1.852
    speed=max(speed,2)                    # si la vitesse est inferieure a 2
    return speed, angle_uv   



def prevision025(GR025,tp, lat0, lon0):
    '''Calcule une prevision a partir du grib composite'''
    '''le tig est le tig du grib le plus recent '''
    tig=GR025[0,0,0,0]*100
    
   

    if not isinstance(lat0,np.ndarray):           #permet de traiter le cas de valeurs simples 
        lat=np.array([lat0])
        lon=np.array([lon0]) 
    else:
        lat=lat0.ravel()
        lon=lon0.ravel()
    if not isinstance(tp,np.ndarray):    
        tp=np.array([tp]) 


  

    # indices decimaux
    itemp  = (tp-tig)/3600/3 
    lat    = (90-lat)*4
    lon    = (lon%360)*4

    #indices entiers 
    lati   = lat.astype(int)
    loni   = lon.astype(int)
    iitemp = itemp.astype(int)       # le -1 a ete supprime 
    iitemp[iitemp>127]=127
    
    # partie fractionnaire des indices 
    ditemp   = itemp%1 
    dx       = lon%1
    dy       = lat%1
   

    #print ('indices ligne 422 ',itemp,lat,lon)   
    #Recuperation des valeurs sous forme complexe
    UV000=GR025[iitemp,lati,loni,0]                    +GR025[iitemp,lati,loni,1]*1j
    UV010=GR025[iitemp,(lati+1)%720,loni,0]            +GR025[iitemp,(lati+1)%720,loni,1]*1j
    UV001=GR025[iitemp,lati,(loni+1)%1440,0]           +GR025[iitemp,lati,(loni+1)%1440,1]*1j
    UV011=GR025[iitemp,(lati+1)%720,(loni+1)%1440,0]   +GR025[iitemp,(lati+1)%720,(loni+1)%1440,1]*1j
    UV100=GR025[iitemp+1,lati,loni,0]                  +GR025[iitemp+1,lati,loni,1]*1j
    UV110=GR025[iitemp+1,(lati+1)%720,loni,0]          +GR025[iitemp+1,(lati+1)%720,loni,1]*1j
    UV101=GR025[iitemp+1,lati,(loni+1)%1440,0]         +GR025[iitemp+1,lati,(loni+1)%1440,1]*1j
    UV111=GR025[iitemp+1,(lati+1)%720,(loni+1)%1440,0] +GR025[iitemp+1,(lati+1)%720,(loni+1)%1440,1]*1j


    #print ('ligne 434 UV000 : ',UV000)

    # Interpolation sur le temps 
    UVX00=UV000+ditemp*(UV100-UV000)     
    UVX10=UV010+ditemp*(UV110-UV010)
    UVX01=UV001+ditemp*(UV101-UV001)
    UVX11=UV011+ditemp*(UV111-UV011)
    
    #Interpolation bilineaire 
    res=UVX00+(UVX01-UVX00)*dx +(UVX10-UVX00)*dy  +(UVX11+UVX00-UVX10-UVX01)*dx*dy   
    
    vitesses,angles=vitangle(res)
    return vitesses,angles



def prevision025dtig(GR025,dtig, lat0, lon0):
    '''Calcule une prevision a partir du grib composite'''
    '''le tig est le tig du grib le plus recent '''
    tig=GR025[0,0,0,0]*100
    
   

    if not isinstance(lat0,np.ndarray):           #permet de traiter le cas de valeurs simples 
        lat=np.array([lat0])
        lon=np.array([lon0]) 
    else:
        lat=lat0.ravel()
        lon=lon0.ravel()
    # if not isinstance(tp,np.ndarray):    
    #     tp=np.array([tp]) 
    if not isinstance(dtig,np.ndarray):    
        dtig=np.array([dtig]) 

  

    # indices decimaux
    itemp  = dtig/3600/3 
    lat    = (90-lat)*4
    lon    = (lon%360)*4

    #indices entiers 
    lati   = lat.astype(int)
    loni   = lon.astype(int)
    iitemp = itemp.astype(int)       # le -1 a ete supprime 
    iitemp[iitemp>127]=127
    
    # partie fractionnaire des indices 
    ditemp   = itemp%1 
    dx       = lon%1
    dy       = lat%1
   

    #print ('indices ligne 422 ',itemp,lat,lon)   
    #Recuperation des valeurs sous forme complexe
    UV000=GR025[iitemp,lati,loni,0]                    +GR025[iitemp,lati,loni,1]*1j
    UV010=GR025[iitemp,(lati+1)%720,loni,0]            +GR025[iitemp,(lati+1)%720,loni,1]*1j
    UV001=GR025[iitemp,lati,(loni+1)%1440,0]           +GR025[iitemp,lati,(loni+1)%1440,1]*1j
    UV011=GR025[iitemp,(lati+1)%720,(loni+1)%1440,0]   +GR025[iitemp,(lati+1)%720,(loni+1)%1440,1]*1j
    UV100=GR025[iitemp+1,lati,loni,0]                  +GR025[iitemp+1,lati,loni,1]*1j
    UV110=GR025[iitemp+1,(lati+1)%720,loni,0]          +GR025[iitemp+1,(lati+1)%720,loni,1]*1j
    UV101=GR025[iitemp+1,lati,(loni+1)%1440,0]         +GR025[iitemp+1,lati,(loni+1)%1440,1]*1j
    UV111=GR025[iitemp+1,(lati+1)%720,(loni+1)%1440,0] +GR025[iitemp+1,(lati+1)%720,(loni+1)%1440,1]*1j


    #print ('ligne 434 UV000 : ',UV000)

    # Interpolation sur le temps 
    UVX00=UV000+ditemp*(UV100-UV000)     
    UVX10=UV010+ditemp*(UV110-UV010)
    UVX01=UV001+ditemp*(UV101-UV001)
    UVX11=UV011+ditemp*(UV111-UV011)
    
    #Interpolation bilineaire 
    res=UVX00+(UVX01-UVX00)*dx +(UVX10-UVX00)*dy  +(UVX11+UVX00-UVX10-UVX01)*dx*dy   
    
    vitesses,angles=vitangle(res)
    return vitesses,angles





def vitangle(res): 
    ''' transforme le complexe u + j*V    en vitesse et angle '''
    if not isinstance(res,np.ndarray):           #permet de traiter le cas de valeurs simples 
        res=np.array([res])
    vitesses=np.abs(res)* 1.94384
    vitesses[vitesses>70] = 70 
    vitesses[vitesses<1]  = 1
    angles = (270 - np.angle(res, deg=True)) % 360

    if len(res)==1 :   
    #    on est dans le cas dune valeur simple
         vitesses=vitesses[0]
         angles=angles[0]  
    return vitesses, angles





def prevision (fineWinds,GR,tp,lat,lon):
    if fineWinds==True: 
        vitesses, angles = prevision025   (GR, tp, lat, lon)
    else :  
        vitesses,angles  = previsionzezo32(GR, tp, lat, lon)
    return vitesses,angles      



# def vitangle(res): 
#     ''' transforme le complexe u + j*V    en vitesse et angle '''
#     if not isinstance(res,np.ndarray):           #permet de traiter le cas de valeurs simples 
#         res=np.array([res])
#     vitesses=np.abs(res)* 1.94384
#     vitesses[vitesses>70] = 70 
#     vitesses[vitesses<1]  = 1
#     angles = (270 - np.angle(res, deg=True)) % 360
#     if len(res)==1 :   
#     #    on est dans le cas dune valeur simple
#          vitesses=vitesses[0]
#          angles=angles[0]  
#     return vitesses, angles


def vitanglebrut(res): 
    ''' transforme le complexe u + j*V    en vitesse et angle '''
    if not isinstance(res,np.ndarray):           #permet de traiter le cas de valeurs simples 
        res=np.array([res])
    vitesses=np.abs(res)
    vitesses[vitesses>70] = 70 
    vitesses[vitesses<1]  = 1
    angles = (270 - np.angle(res, deg=True)) % 360

    if len(res)==1 :   
    #    on est dans le cas dune valeur simple
         vitesses=vitesses[0]
         angles=angles[0]  
    return vitesses, angles










##########################################################################################################
# 4)Fonctions d optimisation
##########################################################################################################
def ftabvmg(tab_tws,tab_twa,polaires):
    '''constitue un tableau des vmg max et min en fonction du vent '''
    ''' tabvmg comprend tws tous les 0.1 twamin et twamax '''
    vmax=max(tab_tws)
    tabvmg=np.zeros((  (vmax-2)*10,3))
    tabvmg[:,0]=np.arange(2,vmax,0.1) 
    for i in range (len(tabvmg)):
        tabvmg[i,1]=vmgmaxspeed(tabvmg[i,0],tab_twa , tab_tws  ,polaires)[0]
        tabvmg[i,2]=vmgmaxspeed(tabvmg[i,0],tab_twa , tab_tws  ,polaires)[2]   # retourne twamax,vmgmax,twamin,vmgmin,twaspeedmax,speedmax
    return tabvmg    

def vmgmaxspeed(tws,tab_twa , tab_tws  ,polaires):
    '''donne les valeurs de vmgmax et speedmax ainsi que les angles pour une force de vent'''
    '''Attention ici tab twa tabtws polaires sont des valeurs globales'''
    ''' calcule un tableau de valeur tous les 0.1 de twa '''
    TWA =np.arange(30,160,0.1).reshape((-1, 1))
    TWS=(np.ones(len(TWA))*tws).reshape((-1, 1))
    donnees=np.concatenate((TWA,TWS),axis=1)
    valeurs = interpn((tab_twa, tab_tws), polaires, donnees, method='linear')
    costwa=np.cos(TWA*math.pi/180)
    VMG=valeurs*costwa.T
    vmgmax=np.max(VMG)
    vmgmin=np.min(VMG)
    twamax=TWA[np.argmax(VMG,axis=1),0][0]
    twamin=TWA[np.argmin(VMG,axis=1),0][0]
    speedmax=np.max(valeurs)
    twaspeedmax=TWA[np.argmax(valeurs,axis=0),0]
    return twamax,vmgmax,twamin,vmgmin,twaspeedmax,speedmax


def fperteStamina (Tws):
    '''Donne la perte de stamina en % a partir d'un tableau de tws'''
    ''' Ce calcul est valable pour gybe et tack '''
    ''' Attention Pour changement de voile il faut multiplier par 2'''
    Perte=1.6666e-4*np.power(Tws,3)+0.183333*Tws+ 10.01 
    Perte=np.where(Perte>20,20,Perte)
    return Perte 


# anciennes fonctions 

# def StaminaRecuperee(Tws,temps):
#     '''Donne le nombre de points de stamina recuperee en fonction du tableau de vent et d'un temps unitaire'''
#     '''Tws tableau des vents , temps en secondes''' 
#     N=temps/60/((np.arctan((Tws-15)/4)+1.504)*3.8 +4.27)
#     return N


# def frecupstamina(Tws):
#     '''temps de recuperation pour 1 point en secondes mn  5mn si tws =0 15 mn si tws >30 '''
#     y=(np.arctan((Tws-15)/4)+1.504)*3.8 +4.27
#     return y

def StaminaRecuperee(Tws, dt):
    ''' Calcul exact vérifié avec ITYC '''
    ''' tws en noeuds, dt en s ''' 
    ''' Tws peut etre un np.array'''
    pouf = 0.8
    Tws = np.asarray(Tws)  # Conversion en tableau si ce n'est pas déjà le cas
    TempsPourUnPoint = spline(0, 30, 240, 720, Tws) / pouf
    ptsRecuperes = dt / TempsPourUnPoint
    return ptsRecuperes





def spline(x1, x2, y1, y2, x):
    ''' Fonction fonctionnant avec x np.array  '''
    x = np.asarray(x)  # Assurez-vous que x est un tableau NumPy
    t = np.clip((x - x1) / (x2 - x1), 0, 1)  # Normalisation entre 0 et 1
    tm1 = 1 - t
    # Calcul du résultat en utilisant t et tm1
    res = tm1**3 * y1 + 3 * tm1**2 * t * y1 + 3 * tm1 * t**2 * y2 + t**3 * y2
    # Gestion des cas aux bornes
    res = np.where(x <= x1, y1, np.where(x >= x2, y2, res))
    return res













##########################################################################################################
# 5)Fonctions de cartographie
##########################################################################################################
def pos_dec_mn_string(lat,lng):
    ''' transforme les degres decimaux en mn sec '''
    '''retourne une chaine'''
    
    abso1= abs(lat)
    degre1=math.floor(abso1)
    min1=math.floor((abso1-degre1)*60)
    sec1=round(((abso1-degre1)*60-min1)*60)
    if lat>0:
        H1='N'
    else:
        H1='S'
    abso2= abs(lng)
    degre2=math.floor(abso2)
    min2=math.floor((abso2-degre2)*60)
    sec2=round(((abso2-degre2)*60-min2)*60,2)
    if lng:
        H2='E'
    else:
        H2='W'
    # print ('Lat {}-{}-{} {}       Lng {}-{}-{} {}\n'.format(degre1,min1,sec1,H1,degre2,min2,sec2,H2))
    latstring= str(degre1)+'-'+str(min1)+'-'+str(sec1)+H1
    lngstring = str(degre2)+'-'+str(min2)+'-'+str(sec2)+H2
    return latstring,lngstring


def pos_dec_mn_1string(lat,lng):
    latstring,lngstring=pos_dec_mn_string(lat,lng)
    return latstring+' '+lngstring


def charge_carto(y0,x0):
    resolution=1
    lat=math.ceil(y0)
    lng=math.floor(x0)
    pas=730
    resolution=1
   
    # on cherche si le fichier existe dans les txt
    file=str(resolution)+'_'+str(lng)+'_'+str(lat)+'.txt'
    # print(basedir)
    filename='/home/jp/static/carto/txt/'+file

    if os.path.exists(filename) == True:                                             # si le fichier existe on le charge
        with open(filename, "r") as fichier:
            cartetexte=fichier.read()
           
            print('Le fichier carte a ete recuperé en local')
        carte=eval(cartetexte)  
        # print(carte)

    else: 
        print('pas de recuperation du fichier en local')                                            # si le fichier n'existe pas on va le chercher chez vr
        lng_f=str(int(lng/10))
        lat_f=str(int(lat/10))
        folder='/'+str(resolution)+'/'+lng_f+'/'+lat_f+'/'
        file=str(resolution)+'_'+str(lng)+'_'+str(lat)+'.deg'
        folderlocal='/home/jp/static/carto/'
        filelocal=folderlocal+file
        #filelocal=file
        
        url='https://static.virtualregatta.com/ressources/maps/dalles/vro2k16'+folder+file
        print('url',url)
        headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" }
        response = requests.get(url, headers=headers)
        print (response.status_code)
        #print (response.content)


        if response.status_code == 200:
            print('fichierlocal ',filelocal)
            with open(filelocal, "wb") as file:
                file.write(response.content)
                print("Fichier téléchargé avec succès.")
           
        else:
            print(f"Erreur {response.status_code} : Impossible de récupérer le fichier chez vr .")

        #2) Ouverture du fichier VR et extraction du masque
        with open(filelocal,'rb') as fid:
            header=fid.read(11)
            gzbuf=fid.read()
            databuf=zlib.decompress(gzbuf,-zlib.MAX_WBITS)
        data=np.frombuffer(databuf,dtype=np.int8)
        data.resize((730,730))
        mask=   ( data > -1) *1

        #3) Conversion du masque en polygone
        carte=extrait_terres(lat,lng,pas,mask)

        #4 on le sauvegarde en txt et en js local
        # ecriture dans fichier
        lng_f=str(int(lng/10))
        lat_f=str(int(lat/10))
        folder=str(resolution)+'/'+lng_f+'/'+lat_f+'/'
        file=str(resolution)+'_'+str(lng)+'_'+str(lat)+'.txt'
        filename='/home/jp/static/carto/txt/'+file
        # print(' sauvegarde du txt ',filename)
        textecarte=str(carte)
        with open(filename, "w+") as fichier:
            fichier.write(textecarte)
   
     
    return  carte   



def charge_carto_local(y0,x0):
    resolution=1
    lat=math.ceil(y0)
    lng=math.floor(x0)
    pas=730
    resolution=1
   
    # on cherche si le fichier existe dans les txt
    file=str(resolution)+'_'+str(lng)+'_'+str(lat)+'.txt'
    # print(basedir)
    filename='/home/jp/staticLocal/carto/txt/'+file

    if os.path.exists(filename) == True:                                             # si le fichier existe on le charge
        with open(filename, "r") as fichier:
            cartetexte=fichier.read()
           
            print('Le fichier carte a ete recuperé en local')
        carte=eval(cartetexte)  
        # print(carte)

    else: 
        print('pas de recuperation du fichier en local')                                            # si le fichier n'existe pas on va le chercher chez vr
        lng_f=str(int(lng/10))
        lat_f=str(int(lat/10))
        folder='/'+str(resolution)+'/'+lng_f+'/'+lat_f+'/'
        file=str(resolution)+'_'+str(lng)+'_'+str(lat)+'.deg'
        folderlocal='/home/jp/staticLocal/carto/'
        filelocal=folderlocal+file
        #filelocal=file
        
        url='https://static.virtualregatta.com/ressources/maps/dalles/vro2k16'+folder+file
        print('url',url)
        headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" }
        response = requests.get(url, headers=headers)
        print (response.status_code)
        #print (response.content)


        if response.status_code == 200:
            print('fichierlocal ',filelocal)
            with open(filelocal, "wb") as file:
                file.write(response.content)
                print("Fichier téléchargé avec succès.")
           
        else:
            print(f"Erreur {response.status_code} : Impossible de récupérer le fichier chez vr .")

        #2) Ouverture du fichier VR et extraction du masque
        with open(filelocal,'rb') as fid:
            header=fid.read(11)
            gzbuf=fid.read()
            databuf=zlib.decompress(gzbuf,-zlib.MAX_WBITS)
        data=np.frombuffer(databuf,dtype=np.int8)
        data.resize((730,730))
        mask=   ( data > -1) *1

        #3) Conversion du masque en polygone
        carte=extrait_terres(lat,lng,pas,mask)

        #4 on le sauvegarde en txt et en js local
        # ecriture dans fichier
        lng_f=str(int(lng/10))
        lat_f=str(int(lat/10))
        folder=str(resolution)+'/'+lng_f+'/'+lat_f+'/'
        file=str(resolution)+'_'+str(lng)+'_'+str(lat)+'.txt'
        filename='/home/jp/staticLocal/carto/txt/'+file
        # print(' sauvegarde du txt ',filename)
        textecarte=str(carte)
        with open(filename, "w+") as fichier:
            fichier.write(textecarte)
   
     
    return  carte   









def chargecarte_points(points):
    '''charge l ensemble des cartes existant sur un itineraire et les concatene'''
    b=np.ceil(points[:,0]).reshape(-1,1)
    c= np.floor(points[:,1]).reshape(-1,1)
    cartes=np.concatenate((b,c),axis=1)
    listecartes=np.unique(cartes, axis=0)
    print(listecartes)

    # cartevr=[]
    # for i in range (len(listecartes)):
    #     try:
    #         carteunique=charge_carto(listecartes[i][0],listecartes[i][1])
    #         print (carteunique)
    #         print() 
    #         cartevr.append(carteunique)
    #     except:
    #         print('echec de chargecarto pour ',listecartes[i][0],listecartes[i][1])
             
    cartevr=[]    
                
    return cartevr   





def extrait_terres(lat,lng,pas,mask):
    # 1) on constitue la liste des carres
    # print(lat)
    terre=[]
    delta=1/pas
    for i in range(pas):
        for j in range(pas):
                if mask[j,i]==1:
                    terre.append([                        
                        [round(lat-j*delta,5),round(lng+(i*delta),5)]\
                       ,[round(lat-j*delta,5),round(lng+(i+1)*delta,5)]\
                        ,[round(lat-(j+1)*delta,5),round(lng+(i+1)*delta,5)]\
                        ,[round(lat-(j+1)*delta,5),round(lng+i*delta,5)]\
                        ,[round(lat-j*delta,5),round(lng+i*delta,5)]])
                    
    # print(len(terre))
    # print (terre[0:2])
    
    polygone_shp=[]
    for i in range (len(terre)):
        polygone_shp.append(Polygon(terre[i]))
    polygonglobal=unary_union(polygone_shp)     
    listeglobale=[]
    try:
     
        nbpoly=len(polygonglobal.geoms)
        for i in range(nbpoly):
            # a=np.array((polygonglobal[i].exterior.coords.xy))
            a=np.array((polygonglobal.geoms[i].exterior.coords.xy))
            Y=(a[0].reshape(-1,1))
            X=(a[1].reshape(-1,1))
            points=np.concatenate((Y,X),axis=1)
            listepoints=[arr.tolist() for arr in (points)]
            listeglobale.append(listepoints)
    except:
        # print ('un seul polygone')  
        #print(polygonglobal) 
        a=np.array(polygonglobal.exterior.coords.xy)
        Y=a[0].reshape(-1,1)
        X= a[1].reshape(-1,1)
        points=np.concatenate((Y,X),axis=1)
        #print(points)

        listepoints=[arr.tolist() for arr in (points)]
        listeglobale.append(listepoints)
    return listeglobale


def point_terre(listept,carte):
    '''listept est une liste de points sous forme de np.array'''
    '''cartevr est une liste de polygone sous forme de liste de liste de points'''
    '''res est un np.array de points'''
    res=np.zeros(len(listept))
    i=0
    for pt in listept:
        point=Point(pt)
        for polygon in carte:
            polygon_sh=Polygon(polygon)
            if point.within(polygon_sh):
                res[i]=1
        i+=1 
    return res     



import numpy as np
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

def point_terre2(listept, carte):
    """Optimise la vérification d'appartenance de points à des polygones."""
    # Création des objets Polygon uniquement une fois
    # polygones = [Polygon(poly) for poly in carte]
    res = np.zeros(len(listept), dtype=np.int8)  # Utiliser un type plus léger que float
    
    polygones_prepares = [prep(Polygon(poly)) for poly in carte if Polygon(poly).is_valid]

    for i, pt in enumerate(listept):
        point = Point(pt)
        if any(poly.contains(point) for poly in polygones_prepares):
            res[i] = 1

    return res 





def point_terre3(listept, carte, n_jobs=-1):
    """Parallélise la vérification d'appartenance de points à des polygones."""
    # Préparer les polygones une seule fois
    polygones_prepares = [prep(Polygon(poly)) for poly in carte if Polygon(poly).is_valid]
    
    def verifier_point(pt):
        point = Point(pt)
        return any(poly.contains(point) for poly in polygones_prepares)
    
    # Utiliser joblib pour paralléliser les vérifications
    res = Parallel(n_jobs=n_jobs)(
        delayed(verifier_point)(pt) for pt in listept
    )
    return np.array(res, dtype=np.int8)





def intersect (polyline,cartevr):
    '''teste lintersection d'une polyline avec la cartevr'''
    polyline_sh=LineString(polyline) 
    k=1
    for polygon in cartevr:
        polygon_sh=Polygon(polygon)
        
        if polyline_sh.intersects(polygon_sh):
            k*=0
    return k  





def fleche3(y,x,cap,l,couleur,m):
    '''Dessine une fleche se terminant au point y,x suivant un cap avec une longueur l dans la carte m '''
    yfin=y-l*math.cos(cap*math.pi/180)
    xfin=x-l*math.sin(cap*math.pi/180)/math.cos(y*math.pi/180)
    yfin2=yfin-l*0.1*math.cos((cap+30)*math.pi/180)
    xfin2=xfin-l*0.1*math.sin((cap+30)*math.pi/180)/math.cos(yfin*math.pi/180)  
    yfin3=yfin-l*0.1*math.cos((cap-30)*math.pi/180)
    xfin3=xfin-l*0.1*math.sin((cap-30)*math.pi/180) /math.cos(yfin*math.pi/180)
    yfine=y-l*1.1*math.cos(cap*math.pi/180)
    xfine=x-l*1.1*math.sin(cap*math.pi/180)/math.cos(y*math.pi/180)
    yfine2=yfine-l*0.1*math.cos((cap+30)*math.pi/180)
    xfine2=xfine-l*0.1*math.sin((cap+30)*math.pi/180)/math.cos(yfine*math.pi/180)
    yfine3=yfine-l*0.1*math.cos((cap-30)*math.pi/180)
    xfine3=xfine-l*0.1*math.sin((cap-30)*math.pi/180) /math.cos(yfine*math.pi/180)
    fleche=[[[y,x],[yfin,xfin],[yfin2,xfin2],[yfin,xfin],[yfin3,xfin3],[yfin,xfin],[yfine,xfine],[yfine2,xfine2],[yfine,xfine],[yfine3,xfine3]]]
    folium.PolyLine (fleche,color=couleur,weight=2 ).add_to(m)
    return None       


##########################################################################################################
# 6)Fonctions de base de donnees
##########################################################################################################


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


















##########################################################################################################
# 7)Fonctions d impression
##########################################################################################################
def impression_routage_titre(titre,tableau)  :  
    '''impression tableau de routage 16 colonnes avec titre '''
    print('\n {}\n***************************************************\
    \n N \t Y \t\t X      \t\tdate     \ttws \t\ttwd \t\tcap \t\ttwa \t\t speed    Voile    Ordre   Valeur  Auto  Cible'.format(titre))
    for i in range (len(tableau)):
        print('{:2.0f}   \t{:6.4f}    \t{:6.4f} \t{}   \t{:6.2f}   \t{:6.2f}   \t{:6.2f}  \t{:6.2f}    \t{:6.2f}  {:6.0f}   {:6.0f}   {:6.2f}   {:4.0f}  {:4.0f}'\
              .format(tableau[i][0],tableau[i][1], tableau[i][2], time.strftime(" %d %b %H:%M ",time.localtime(tableau[i][3])), tableau[i][4], tableau[i][5],\
                      tableau[i][6], tableau[i][7], tableau[i][8], tableau[i][9], tableau[i][10], tableau[i][11], tableau[i][12], tableau[i][13]))
    return None


def impression10isocreme(tableau):
    '''impressionisocreme 10 colonnes'''
    for i in range (len(tableau)):
        print('{:6.4f} \t{:6.4f} \t {:4.0f} \t {:4.0f} \t{:4.0f}  \t {:6.2f} \t {:6.2f}  \t{:6.2f}  \t{:6.2f}  \t{:6.2f}'\
              .format(tableau[i,0],tableau[i,1], tableau[i,2], tableau[i,3], tableau[i,4], tableau[i,5],tableau[i,6],tableau[i,7], tableau[i,8], tableau[i,9])) 



def impression8isoglobal(tableau):
    '''impressiontableau 8 colonnes'''
    ''' pour chemincomplet'''
    print(' X \t\t Y \t\t  nsiso \t nptsm \t\t npoint  \t npt-ds_iso  \t ordoar  \t twaor') 
    for i in range (len(tableau)):
        print('{:6.4f} \t{:6.4f} \t {:6.0f}  \t{:6.0f}  \t{:6.0f} \t\t {:6.2f} \t {:6.2f}  \t{:6.2f}   '\
              .format(tableau[i,0],tableau[i,1], tableau[i,2], tableau[i,3],tableau[i,4], tableau[i,5],tableau[i,6],tableau[i,7]))         
        

       
def impression11cc(tableau):
    '''impression tableau 11 colonnes pour chemin complet avec twaopti et distance'''
    print ('N pt \tY \t\t  X \t\t\tT \ttws   \t  twd \t\t tabcap\t \ttwaos \t\ttwaopti \tvitesses \tdist')
    for i in range (len(tableau)):
        print('{:6.0f} \t{:6.4f} \t {:6.4f}   \t{} \t{:6.2f} \t {:6.2f} \t {:6.2f}  \t{:6.2f}  \t{:6.2f}  \t{:6.2f} \t\t{:6.4f}'\
              .format(tableau[i,0],tableau[i,1], tableau[i,2], time.strftime(" %d %b %H:%M ",time.localtime(tableau[i,3])), tableau[i,4], tableau[i,5],tableau[i,6],tableau[i,7], tableau[i,8],tableau[i,9],tableau[i,10])) 
           


   














# def dateheure(filename):
#     '''retourne la date et heure du fichier grib a partir du nom'''
#     ''' necessaire pour charger le NOAA'''
#     tic=time.time()
#     ticstruct = time.localtime()
#     utc = time.gmtime()
#     decalage = ticstruct[3] - utc[3]
#     x     = filename.split('.')[0]
#     x     = x.split('/')[-1]
#     heure = x.split('-')[1]
#     date  = (x.split('-')[0]).split('_')[1]
#     year  = int(date[0:4])
#     month = int(date[4:6])
#     day   = int(date[6:8])
#     tigt=datetime(year,month,day,int(heure),0, 0)
#     tig=time.mktime(tigt.timetuple()) +decalage*3600 # en secondes UTC
#     return date,heure,tig  


# def gribFileName(basedir):
#     ''' cherche le dernier grib complet disponible au temps en secondes '''
#     ''' temps_secondes est par defaut le temps instantané '''
#     ''' Cherche egalement le dernier grib chargeable partiellement'''
#     temps_secondes=time.time()
#     date_tuple       = time.gmtime(temps_secondes) 
#     date_formatcourt = time.strftime("%Y%m%d", time.gmtime(temps_secondes))
#     dateveille_tuple = time.gmtime(temps_secondes-86400) 
#     dateveille_formatcourt=time.strftime("%Y%m%d", time.gmtime(temps_secondes-86400))
#     mn_jour_utc =date_tuple[3]*60+date_tuple[4]
#     #print ('mn_jour_utc dans gribFileNames ',mn_jour_utc)
#     if (mn_jour_utc <3*60+48):                          #avant 3h 48 UTC le nom de fichier est 18 h de la veille 
#         filename=basedir+"/gfs_"+dateveille_formatcourt+"-18.npy"
#     elif (mn_jour_utc<9*60+48):   
#         filename=basedir+"/gfs_"+date_formatcourt+"-00.npy"
#     elif (mn_jour_utc<15*60+48): 
#         filename=basedir+"/gfs_"+date_formatcourt+"-06.npy"
#     elif (mn_jour_utc<21*60+48):   
#         filename=basedir+"/gfs_"+date_formatcourt+"-12.npy"
#     else:                                              
#         filename=basedir+"/gfs_"+date_formatcourt+"-18.npy"   
#     return filename  




# def vitangle(res): 
#     ''' transforme le complexe u + j*V    en vitesse et angle '''
#     if not isinstance(res,np.ndarray):           #permet de traiter le cas de valeurs simples 
#         res=np.array([res])
#     vitesses=np.abs(res)* 1.94384
#     vitesses[vitesses>70] = 70 
#     vitesses[vitesses<1]  = 1
#     angles = (270 - np.angle(res, deg=True)) % 360

#     if len(res)==1 :   
#     #    on est dans le cas dune valeur simple
#          vitesses=vitesses[0]
#          angles=angles[0]  
#     return vitesses, angles




# def prevision025(GR025,tig,tp, lat0, lon0):
#     '''Calcule une prevision a partir du grib composite'''
#     '''le tig est le tig du grib le plus recent '''
   
#     if not isinstance(lat0,np.ndarray):           #permet de traiter le cas de valeurs simples 
#         lat=np.array([lat0])
#         lon=np.array([lon0]) 
#     else:
#         lat=lat0.ravel()
#         lon=lon0.ravel()
#     if not isinstance(tp,np.ndarray):    
#         tp=np.array([tp]) 
        
#     # indices decimaux
#     itemp  = (tp-tig)/3600/3 
#     lat    = (90-lat)*4
#     lon    = (lon%360)*4

#     print(' Dans prevision 025 indices en ligne 91',itemp ,lat,lon)

#     #indices entiers 
#     lati   = lat.astype(int)
#     loni   = lon.astype(int)
#     iitemp = itemp.astype(int) -1
#     iitemp[iitemp>127]=127

#     print(' Dans prevision 025 indices en ligne 91',iitemp ,lati,loni)
    
#     # partie fractionnaire des indices 
#     ditemp   = itemp%1 
#     dx       = lon%1
#     dy       = lat%1
   
#     #Recuperation des valeurs sous forme complexe
#     UV000=GR025[iitemp,lati,loni,0]                    +GR025[iitemp,lati,loni,1]*1j
#     UV010=GR025[iitemp,(lati+1)%720,loni,0]            +GR025[iitemp,(lati+1)%720,loni,1]*1j
#     UV001=GR025[iitemp,lati,(loni+1)%1440,0]           +GR025[iitemp,lati,(loni+1)%1440,1]*1j
#     UV011=GR025[iitemp,(lati+1)%720,(loni+1)%1440,0]   +GR025[iitemp,(lati+1)%720,(loni+1)%1440,1]*1j
#     UV100=GR025[iitemp+1,lati,loni,0]                  +GR025[iitemp+1,lati,loni,1]*1j
#     UV110=GR025[iitemp+1,(lati+1)%720,loni,0]          +GR025[iitemp+1,(lati+1)%720,loni,1]*1j
#     UV101=GR025[iitemp+1,lati,(loni+1)%1440,0]         +GR025[iitemp+1,lati,(loni+1)%1440,1]*1j
#     UV111=GR025[iitemp+1,(lati+1)%720,(loni+1)%1440,0] +GR025[iitemp+1,(lati+1)%720,(loni+1)%1440,1]*1j


   
#     # print ('ligne 115 UV000 :{}'.format (UV000))
#     # print ('ligne 115 UV000 :{}'.format (UV010))
#     # print ('ligne 115 UV000 :{}'.format (UV001))
#     # print ('ligne 115 UV000 :{}'.format (UV011))
#     # print ('ligne 115 UV000 :{}'.format (UV100))
#     # print ('ligne 115 UV000 :{}'.format (UV110))
#     # print ('ligne 115 UV000 :{}'.format (UV101))
#     # print ('ligne 115 UV000 :{}'.format (UV111))

    

#     # Interpolation sur le temps 
#     UVX00=UV000+ditemp*(UV100-UV000)     
#     UVX10=UV010+ditemp*(UV110-UV010)
#     UVX01=UV001+ditemp*(UV101-UV001)
#     UVX11=UV011+ditemp*(UV111-UV011)
    
#     #Interpolation bilineaire 
#     res=UVX00+(UVX01-UVX00)*dx +(UVX10-UVX00)*dy  +(UVX11+UVX00-UVX10-UVX01)*dx*dy   
    
#     vitesses,angles=vitangle(res)
#     return vitesses,angles







# def prevision100(GR,tig,tp, lat0, lon0):
#     '''Calcule une prevision a partir du grib composite'''
#     '''le tig est le tig du grib le plus recent '''
   
#     if not isinstance(lat0,np.ndarray):           #permet de traiter le cas de valeurs simples 
#         lat=np.array([lat0])
#         lon=np.array([lon0]) 
#     else:
#         lat=lat0.ravel()
#         lon=lon0.ravel()
#     if not isinstance(tp,np.ndarray):    
#         tp=np.array([tp]) 
        
#     # indices decimaux
#     itemp  = (tp-tig)/3600/3 
#     lat    = (90-lat)
#     lon    = (lon%360)
#     print('indices dans prevision 100 : ',itemp,lat,lon)
#     #indices entiers 
#     lati   = lat.astype(int)
#     loni   = lon.astype(int)
#     iitemp = itemp.astype(int) 
#     iitemp[iitemp>127]=127
    
#     # partie fractionnaire des indices 
#     ditemp   = itemp%1 
#     dx       = lon%1
#     dy       = lat%1
   
#     #Recuperation des valeurs sous forme complexe
#     UV000=GR [iitemp,lati,loni]                   #+GR[iitemp,lati,loni,1]*1j
#     UV010=GR [iitemp,(lati+1)%180,loni]           #+GR[iitemp,(lati+1)%720,loni,1]*1j
#     UV001=GR [iitemp,lati,(loni+1)%360]           #+GR[iitemp,lati,(loni+1)%1440,1]*1j
#     UV011=GR [iitemp,(lati+1)%180,(loni+1)%360]   #+GR[iitemp,(lati+1)%720,(loni+1)%1440,1]*1j
#     UV100=GR [iitemp+1,lati,loni]                 #+GR[iitemp+1,lati,loni,1]*1j
#     UV110=GR [iitemp+1,(lati+1)%180,loni]         #+GR[iitemp+1,(lati+1)%720,loni,1]*1j
#     UV101=GR [iitemp+1,lati,(loni+1)%360]         #+GR[iitemp+1,lati,(loni+1)%1440,1]*1j
#     UV111=GR [iitemp+1,(lati+1)%180,(loni+1)%360] #+GR[iitemp+1,(lati+1)%720,(loni+1)%1440,1]*1j

#     print ('dans prevision 100 ligne 186')
#     print( 'UV000 :{}'.format (UV000))
#     print ('UV010 :{}'.format (UV010))
#     print ('UV001 :{}'.format (UV001))
#     print ('UV011 :{}'.format (UV011))
#     print ('UV100 :{}'.format (UV100))
#     print ('UV110 :{}'.format (UV110))
#     print ('UV101 :{}'.format (UV101))
#     print ('UV111 :{}'.format (UV111))





#     # Interpolation sur le temps 
#     UVX00=UV000+ditemp*(UV100-UV000)     
#     UVX10=UV010+ditemp*(UV110-UV010)
#     UVX01=UV001+ditemp*(UV101-UV001)
#     UVX11=UV011+ditemp*(UV111-UV011)
    

#     print()
#     print( 'UVX00 :{}'.format (UVX00))
#     print ('UVX10 :{}'.format (UVX10))
#     print ('UVX01 :{}'.format (UVX01))
#     #Interpolation bilineaire 
#     res=UVX00+(UVX01-UVX00)*dx +(UVX10-UVX00)*dy  +(UVX11+UVX00-UVX10-UVX01)*dx*dy   
    
#     vitesses,angles=vitangle(res)
#     return vitesses,angles


# def prevision100f(GR,tig,tp, lat0, lon0):
#     '''Calcule une prevision a partir du grib100   en float'''
#     '''le tig est le tig du grib le plus recent '''
   
#     if not isinstance(lat0,np.ndarray):           #permet de traiter le cas de valeurs simples 
#         lat=np.array([lat0])
#         lon=np.array([lon0]) 
#     else:
#         lat=lat0.ravel()
#         lon=lon0.ravel()
#     if not isinstance(tp,np.ndarray):    
#         tp=np.array([tp]) 
        
#     # indices decimaux
#     itemp  = (tp-tig)/3600/3 
#     lat    = (90-lat)
#     lon    = (lon%360)
#     print('indices entiers dans prevision 100f: ',itemp,lat,lon)
#     #indices entiers 
#     lati   = lat.astype(int)
#     loni   = lon.astype(int)
#     iitemp = itemp.astype(int) 
#     iitemp[iitemp>127]=127
    
#     # partie fractionnaire des indices 
#     ditemp   = itemp%1 
#     dx       = lon%1
#     dy       = lat%1
   
#     #Recuperation des valeurs sous forme complexe
#     UV000=GR [iitemp,lati,loni,0]                   +GR[iitemp,lati,loni,1]*1j
#     UV010=GR [iitemp,(lati+1)%180,loni,0]           +GR[iitemp,(lati+1)%180,loni,1]*1j
#     UV001=GR [iitemp,lati,(loni+1)%360,0]           +GR[iitemp,lati,(loni+1)%360,1]*1j
#     UV011=GR [iitemp,(lati+1)%180,(loni+1)%360,0]   +GR[iitemp,(lati+1)%180,(loni+1)%360,1]*1j
#     UV100=GR [iitemp+1,lati,loni,0]                 +GR[iitemp+1,lati,loni,1]*1j
#     UV110=GR [iitemp+1,(lati+1)%180,loni,0]         +GR[iitemp+1,(lati+1)%180,loni,1]*1j
#     UV101=GR [iitemp+1,lati,(loni+1)%360,0]         +GR[iitemp+1,lati,(loni+1)%360,1]*1j
#     UV111=GR [iitemp+1,(lati+1)%180,(loni+1)%360,0] +GR[iitemp+1,(lati+1)%180,(loni+1)%360,1]*1j


#     print ('dans prevision 100f ligne 253')
#     print( 'UV000 :{}'.format (UV000))
#     print ('UV010 :{}'.format (UV010))
#     print ('UV001 :{}'.format (UV001))
#     print ('UV011 :{}'.format (UV011))
#     print ('UV100 :{}'.format (UV100))
#     print ('UV110 :{}'.format (UV110))
#     print ('UV101 :{}'.format (UV101))
#     print ('UV111 :{}'.format (UV111))

   

#     # Interpolation sur le temps 
#     UVX00=UV000+ditemp*(UV100-UV000)     
#     UVX10=UV010+ditemp*(UV110-UV010)
#     UVX01=UV001+ditemp*(UV101-UV001)
#     UVX11=UV011+ditemp*(UV111-UV011)
#     print()
#     print( 'UVX00 :{}'.format (UVX00))
#     print ('UVX10 :{}'.format (UVX10))
#     print ('UVX01 :{}'.format (UVX01))
#     #Interpolation bilineaire 
#     res=UVX00+(UVX01-UVX00)*dx +(UVX10-UVX00)*dy  +(UVX11+UVX00-UVX10-UVX01)*dx*dy   
    
#     vitesses,angles=vitangle(res)
#     return vitesses,angles


# def prevision100fvr(GR,tig,tp, lat0, lon0):
#     '''Calcule une prevision a partir du grib100   en float'''
#     '''le tig est le tig du grib le plus recent '''

#     # on recupere le avail ts 
#     ts=GR[0,0,0,0]
   
   
#     if not isinstance(lat0,np.ndarray):           #permet de traiter le cas de valeurs simples 
#         lat=np.array([lat0])
#         lon=np.array([lon0]) 
#     else:
#         lat=lat0.ravel()
#         lon=lon0.ravel()
#     if not isinstance(tp,np.ndarray):    
#         tp=np.array([tp]) 
        
#     # indices decimaux
#     itemp  = (tp-tig)/3600/3 
#     lat    = (90-lat)
#     lon    = (lon%360)
#     print('indices entiers dans prevision 100f: ',itemp,lat,lon)
#     #indices entiers 
#     lati   = lat.astype(int)
#     loni   = lon.astype(int)
#     iitemp = itemp.astype(int) 
#     iitemp[iitemp>127]=127
    
#     # partie fractionnaire des indices 
#     ditemp   = itemp%1 
#     dx       = lon%1
#     dy       = lat%1
   
#     #Recuperation des valeurs sous forme complexe
#     UV000=GR [iitemp,lati,loni,0]                   +GR[iitemp,lati,loni,1]*1j
#     UV010=GR [iitemp,(lati+1)%180,loni,0]           +GR[iitemp,(lati+1)%180,loni,1]*1j
#     UV001=GR [iitemp,lati,(loni+1)%360,0]           +GR[iitemp,lati,(loni+1)%360,1]*1j
#     UV011=GR [iitemp,(lati+1)%180,(loni+1)%360,0]   +GR[iitemp,(lati+1)%180,(loni+1)%360,1]*1j
#     UV100=GR [iitemp+1,lati,loni,0]                 +GR[iitemp+1,lati,loni,1]*1j
#     UV110=GR [iitemp+1,(lati+1)%180,loni,0]         +GR[iitemp+1,(lati+1)%180,loni,1]*1j
#     UV101=GR [iitemp+1,lati,(loni+1)%360,0]         +GR[iitemp+1,lati,(loni+1)%360,1]*1j
#     UV111=GR [iitemp+1,(lati+1)%180,(loni+1)%360,0] +GR[iitemp+1,(lati+1)%180,(loni+1)%360,1]*1j


#     print ('dans prevision 100f ligne 253')
#     print( 'UV000 :{}'.format (UV000))
#     print ('UV010 :{}'.format (UV010))
#     print ('UV001 :{}'.format (UV001))
#     print ('UV011 :{}'.format (UV011))
#     print ('UV100 :{}'.format (UV100))
#     print ('UV110 :{}'.format (UV110))
#     print ('UV101 :{}'.format (UV101))
#     print ('UV111 :{}'.format (UV111))

   

#     # Interpolation sur le temps 
#     UVX00=UV000+ditemp*(UV100-UV000)     
#     UVX10=UV010+ditemp*(UV110-UV010)
#     UVX01=UV001+ditemp*(UV101-UV001)
#     UVX11=UV011+ditemp*(UV111-UV011)
#     print()
#     print( 'UVX00 :{}'.format (UVX00))
#     print ('UVX10 :{}'.format (UVX10))
#     print ('UVX01 :{}'.format (UVX01))
#     #Interpolation bilineaire 
#     res=UVX00+(UVX01-UVX00)*dx +(UVX10-UVX00)*dy  +(UVX11+UVX00-UVX10-UVX01)*dx*dy   
    
#     vitesses,angles=vitangle(res)
#     return vitesses,angles




























# def previsionf(fineWinds,GR,tig,tp, lat0, lon0):
#     '''previsions a partir des fichiers en float 32'''
#     if fineWinds==True :
#         vitesses,angles=prevision025(GR,tig,tp, lat0, lon0)
#     else : 
#         vitesses,angles=prevision100f(GR,tig,tp, lat0, lon0)
#     return vitesses,angles       







# necessaire pour ancienne prevision 
def recupereGrib (filename):
    filenamejson=filename.split('.')[0]+'.json'
    with open(filename, 'rb') as f:
        GR = np.load(f)

    with open(filenamejson, 'r') as fp:
        data = json.load(fp)
    tig=data['tig']
    indices=data['indices']
    avail_ts=data['avail_ts']
    return GR,tig,indices,avail_ts      




def chargement_grib():
    global GR025
    try:
        # on essaye de charger sur serveur 
        basedirGribs025     = '/home/jp/gribslocaux/gribs025/'
        fileName,tig=gribFileName(basedirGribs025)
        heure= datetime.fromtimestamp(tig, tz=timezone.utc).hour
        with open(fileName, 'rb') as f:
                GR025 = np.load(f)           
        print('Le grib 025  {} h+ {:3.0f}h            {}     a été chargé sur le site distant'.format(heure, GR025[0,0,0,1]*3,fileName))
        return GR025,tig

    except:
        basedirgribs='/home/jp/gribslocaux/gribs025/'
        fileName,tig=gribFileName(basedirgribs)
        heure= datetime.fromtimestamp(tig, tz=timezone.utc).hour
        with open(fileName, 'rb') as f:
                GR025 = np.load(f)
        print('Le grib 025 {} h+ {:3.0f}h            {}      a été chargé sur l ordi local  '.format(heure,GR025[0,0,0,1]*3,fileName))
        return GR025,tig                                                                                            
   



def majgrib():
    global GR025,tig
    basedirGribs025     = '/home/jp/gribslocaux/gribs025/'
    filename,derniertig=gribFileName(basedirGribs025)
    #  si pas sur dernier grib ou si moins de  360 h chargées
    print('Dernier indice chargé',GR025[0,0,0,1]*3,'h')
    if os.path.exists(filename)==True:
        if (derniertig!=GR025[0,0,0,0]*100 )   or (int(GR025[0,0,0,1]<120) ):
            print('\n Rechargement du grib necessaire\n******************************\n')
            GR025,tig = chargement_grib()
            print('Nouveau Dernier indice chargé',GR025[0,0,0,1]*3,'h')
            return 

    else:
        return




