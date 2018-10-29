 # -*- coding: utf-8 -*-

import pandas as pd
#import matplotlib.pyplot as plt
import json
import os
from pygments import highlight, lexers, formatters
import pprint


#import imp
from .prediction_classe import Prediction
#from .mes_classes import Aeroport, Vols


####from . import mes_fonctions_final2 as mes_fonctions_final2
#### /!\ Le module mes_fonctions_final2 doit être présent car appelé 
###  dans le modèle des données 


#from .utils import get_prediction

Pred = Prediction()

# Post = pd.read_csv('', index )
file1 = 'prog_tags/data/QueryResults3.csv'
Post = pd.read_csv(file1, index_col=['Id'])

def get_prediction(t_title, t_body):
    return Pred.predict(
            texte_body=t_body,
            texte_title=t_title)


def get_prediction2(val1, val2):
    return Pred.predict(
            texte_body=val1,
            texte_title=val2)


def recupere_post_id(post_id):
    import re
    from bs4 import BeautifulSoup
    
    u = Post[Post.index == post_id][['Tags', 'Body', 'Title']]

    m_tags = re.findall(r'<(\w+|[0-9a-zA-Z-\-\.\+\#]+)>', u['Tags'].values[0])
    html_in_soup = BeautifulSoup(u['Body'].values[0], 'html.parser')
    html_in_soup.get_text()
    
    res = {
        'Title': u['Title'].values[0],
        'Body': html_in_soup.get_text(), #u['Body'].values[0],
        'Tags': m_tags
    }
    return res

#texte_body='python et java sont dans un bateau', 
 #       texte_title='python tombe a l eau')













from retards_avions.models import Aeroport, Vols

basedir = os.path.abspath(os.path.dirname(__file__))




def mon_test():
    valeur = Aeroport.data
    return '{}'.format(valeur)

def mon_test2():
    l1 = Aeroport.get_ville(with_code=True)
    f1 = lambda x: {'value':x['name'],
                    'label':x['name'],
                    'desc': x['code']}
    return [f1(x) for x in l1]


def get_ville(ville=None):
    l1 = Aeroport.get_ville(ville=ville, with_code=True)
    f1 = lambda x: {'value': x['name'], 
                    'label': x['name'], 
                    'desc': x['code']}
    l_res = [f1(x) for x in l1]
    return l_res
    return ({"villes": l_res})



def get_destinations(ville_origine):
    '''
    Renvoie la liste des destinations 
    pour la ville d'origine donnée
    '''
    l1 = Aeroport.get_destinations(ville_origine=ville_origine)
    f1 = lambda x: {'value': x, 
                    'label': x, 
                    'desc': x}
    l_res =  [f1(x) for x in l1]
    return l_res



def get_vols(ville_origine, ville_arrivee ):
    '''Renvoie la liste des vols 
    pour la ville de départ et la ville d'arrivee données
    '''
    l1 = Vols.get_vols(ville_origine = ville_origine,
                       ville_arrivee = ville_arrivee,
                       colums_sortie = ['ORIGIN_AIRPORT_SEQ_ID', 
                                        'DEST_AIRPORT_SEQ_ID', 
                                        'FL_DATE']
                       )
    
    f1 = lambda row: {'dep': row['ORIGIN_AIRPORT_SEQ_ID'], 
                  'arr': row['DEST_AIRPORT_SEQ_ID'], 
                  'dep_date': row['FL_DATE']
                 }

    l_res = l1.apply(lambda row: f1(row), axis=1).tolist()

    return l_res

def get_date_vols(ville_origine, ville_arrivee ):
    '''Renvoie la liste des heures de vols 
    pour la ville de départ et la ville d'arrivee données
    '''
    l1 = Vols.get_vols(ville_origine = ville_origine,
                       ville_arrivee = ville_arrivee,
                       colums_sortie = ['ORIGIN_AIRPORT_SEQ_ID', 
                                        'DEST_AIRPORT_SEQ_ID', 
                                        'FL_DATE']
                       )
    
    f1 = lambda row: row['FL_DATE']

    l_res = l1.apply(lambda row: f1(row), axis=1).tolist()

    return l_res


def get_vols_jour(ville_origine, ville_arrivee , datedep):
    '''Renvoie les infos completes du vols
    '''
    l1 = Vols.get_vols(ville_origine = ville_origine,
                       ville_arrivee = ville_arrivee,
                       data_recherche = {'date_depart': datedep},
                       colums_sortie = ['ORIGIN_AIRPORT_SEQ_ID', 
                                        'DEST_AIRPORT_SEQ_ID', 
                                        'FL_DATE', 
                                        'AIRLINE_ID',
                                        'CRS_DEP_TIME', 
                                        'CRS_ARR_TIME', 
                                       ]
                       )
    
    ajoute_logs('get_vols_jour')
    liste_res = []
    for i in l1.values:
        val = {}
        val['ORIGIN_AIRPORT_SEQ_ID'] = i[0]
        val['DEST_AIRPORT_SEQ_ID'] = i[1]
        val['FL_DATE'] = i[2]
        val['AIRLINE_ID'] = i[3]
        val['CRS_DEP_TIME'] = i[4]
        val['CRS_ARR_TIME'] = i[5]
        liste_res.append(val)

    return liste_res


def get_prediction(ville_origine, ville_arrivee, datedep,
                    airline_id, dep_hour, arr_hour):
    
    try:
        u, v = Vols.get_prediction(ville_origine, ville_arrivee, datedep,
                            airline_id, dep_hour, arr_hour)
    except Exception as e: 
        ajoute_logs('get_prediction ERREUR {}'.format(e))

    return u,v 

    return Vols.get_prediction(ville_origine, ville_arrivee, datedep,
                            airline_id, dep_hour, arr_hour)


    ########################################
    #####   GESTION DU STOCKAGE DES MODELS
    ########################################

def prepare_models():
    Vols.Prepare_Model()
    

def chargement_models():    
    Vols.Charge_Models()
    

    #########################
    #####   GESTION DES LOGS
    #########################

def affiche_logs():
    return Vols.logs.affiche()

def supprime_logs():
    Vols.logs.remove_msg()
    return True

def ajoute_logs(msg):
    Vols.logs(msg, param = 'APPLIWEB : ')




# Pour afficher les graphiques inline
#%matplotlib inline

def charge_df():
    #FILE = os.path.join(basedir, 'datas/movie_metadata.csv')
    FILE = os.path.join(basedir, 'datas/aeroports_nettoyes.csv')
    
    #FILE = 'datas/movie_metadata.csv'
    #####df = pd.read_csv(FILE, index_col='AIRPORT_SEQ_ID')
    
    return None
    return df


MA_BASE = charge_df()


'''
def get_df_value(mon_id):
    #u = MA_BASE.loc[nb,['director_name','movie_title']]
    #return resultat(MA_BASE, 1)


    #colorful_json = highlight(unicode(formatted_json, 'UTF-8'), lexers.JsonLexer(), formatters.TerminalFormatter())
    formatted_json = json.dumps(resultat(MA_BASE, mon_id), indent=4)
    colorful_json = highlight(formatted_json, lexers.JsonLexer(), formatters.TerminalFormatter())
    print(colorful_json)

    #return json.dumps(resultat(MA_BASE, 1), indent=4)
    return pprint.pformat(formatted_json)
'''

'''def get_df_value(mon_id):
    ma_recherche = resultat(MA_BASE, mon_id)
    return ma_recherche
'''


def get_ville_old(ville = None):
    #####M_AEROPORTS = MA_BASE.head(10)
    M_AEROPORTS = MA_BASE
    
    res_ville_possibles = []

    for c_name, c_code in zip(M_AEROPORTS['CITY_NAME'], M_AEROPORTS.index):
        if ville==None:
            res_v = {}
            res_v['code'] = c_code
            res_v['name'] = c_name
            res_ville_possibles.append(res_v)
        elif ville.lower() in c_name.lower():
            res_v = {}
            res_v['code'] = c_code
            res_v['name'] = c_name
            res_ville_possibles.append(res_v)

    #return res_ville_possibles

    
    #return json.dumps(str(res_ville_possibles), indent=4) 
    res_ville_possibles2 = {"villes": res_ville_possibles}
    #return (json.dumps(str(res_ville_possibles), indent=4)) 
    retour = json.dumps(res_ville_possibles2, indent=4)
    #callback = request.args.get('callback')
    return retour




def get_date_active():
    def transforme_date(date_in, fmt = '%Y-%m-%d'):
        from datetime import datetime
        u = datetime.strptime(date_in, fmt)
        return '{m}-{d}-{y}'.format(y = u.year, m = u.month, d = u.day)

    l_date =  [ '2018-07-30', '2018-08-11',
               '2018-08-14','2018-08-15',
               '2018-08-16', '2018-08-19']

    return [ transforme_date(x) for x in l_date ]








## TEST A VERIFIER


    ############### 
    ##### ICI CE SONT DES ANCIENNES INFORMATIONS
    #####
    
def get_df_value(mon_id):
    ma_recherche = resultat(MA_BASE, mon_id)
    return str(ma_recherche)

def get_df_value2(mon_id):
    ma_recherche = resultat(MA_BASE, mon_id)
    
    #colorful_json = highlight(unicode(formatted_json, 'UTF-8'), lexers.JsonLexer(), formatters.TerminalFormatter())
    formatted_json = json.dumps(str(ma_recherche), indent=4)
    colorful_json = highlight(formatted_json, lexers.JsonLexer(), formatters.TerminalFormatter())
    print(colorful_json)

    return pprint.pformat(formatted_json)

def get_movie_liste_by_name(approximativ_movie_name='', m_dataframe=MA_BASE):
    m = m_dataframe['movie_title']
    res_title = []
    l_resultat = []
    
    for i in m:
        if approximativ_movie_name.lower() in i.lower():
            res_title.append(i)
    
    for movie_name in sorted(res_title):
        id = m_dataframe[m_dataframe['movie_title']==movie_name]['id'].values[0]
        l_resultat.append(retourne_films_info_by_id(id))
     
    return l_resultat


 
def retourne_films_info_by_id(id, m_dataframe=MA_BASE):
    film_info = m_dataframe[m_dataframe['id']==id]
    res_film_info = {}
    res_film_info['id'] = film_info['id'].values[0]
    res_film_info['titre'] = film_info['movie_title'].values[0]
    res_film_info['directeur'] = film_info['director_name'].values[0]
    res_film_info['act1'] = film_info['actor_1_name'].values[0]
    res_film_info['act2'] = film_info['actor_2_name'].values[0]
    res_film_info['act3'] = film_info['actor_3_name'].values[0]
    res_film_info['label'] = film_info['label'].values[0]
    res_film_info['imdb_score'] = film_info['imdb_score'].values[0]
    res_film_info['movie_imdb_link'] = film_info['movie_imdb_link'].values[0]
    res_film_info['gross'] = film_info['gross'].values[0]

    return res_film_info


def retourne_liste_films_info_by_cat(no_cat, m_dataframe=MA_BASE):
    film_info = m_dataframe[m_dataframe['label']==no_cat].\
        sort_values('imdb_score',ascending=False)
    
    film_info = m_dataframe[m_dataframe['label']==no_cat].\
        sort_values('movie_title',ascending=True)
    res = []
    
    #for id in film_info['id'].head():
    #    res.append(retourne_films_info_by_id(id, m_dataframe))
    
    for id in film_info['id']:
        res.append(retourne_films_info_by_id(id, m_dataframe))

    return res

def retourne_resultat_fichier(m_dataframe, id):
    film_name = m_dataframe[m_dataframe['id']==id]['movie_title'].values[0]
    return {"id": id, "name":film_name}

def resultat_old(m_dataframe, id):
    film_label = m_dataframe[m_dataframe['id']==id]['label'].values[0]
    film_age = m_dataframe[m_dataframe['id']==id]['age_limite'].values[0]
    film_director = m_dataframe[m_dataframe['id']==id]['director_name'].values[0]

    res = []
    
    # On recherche les données ayant le même label
    m_df_tmp = m_dataframe[m_dataframe['label']==film_label]\
            .sort_values('imdb_score',ascending=False)
    
        
    l_id = []  
    m_df_tmp2 = m_df_tmp[m_df_tmp['age_limite']==film_age]\
            .sort_values('imdb_score',ascending=False).head(6) 
            
       
    l_id.extend(m_df_tmp2['id'].head(6).get_values())
    m_df_tmp2 = m_df_tmp[m_df_tmp['director_name']==film_director]\
            .sort_values('imdb_score',ascending=False).head(6)   
    l_id.extend(m_df_tmp2['id'].head(6).get_values())  
    # On applique ici un filtrage
    
    l_id.remove(id)
    
    for i in l_id[:5]:
        res.append(retourne_resultat_fichier(m_dataframe, i))
    
    return { "_results": res}

def modelisation_finale(mdata, id):
    # Nous récupérons les id de même label
    film_input = mdata[mdata['id']==id]
    label = film_input['label'].values[0]
    
    director = film_input['director_name'].values[0]
    act1 = film_input['actor_1_name'].values[0]
    act2 = film_input['actor_2_name'].values[0]
    act3 = film_input['actor_3_name'].values[0]
    color = film_input['color'].values[0]
    audience = film_input['audience'].values[0]
    age_limite = film_input['age_limite'].values[0]
    
    # Ensemble des résultats 
    df = mdata[mdata['label']==label]
    my_df = pd.DataFrame({'id':df['id'], 'imdb_score':df['imdb_score']})
    my_df['priorite'] = 0
    
    # Filtrage des informations
    # 1- Ajout pour couleur
    if color=='Black and White':
        tmp = my_df['priorite'] + [2 if i=='Black and White' else 0 for i in df['color']]
        my_df['priorite'] = tmp
        
    # 2- Ajout de l'age limite
    if age_limite<14:
        tmp = my_df['priorite'] + [1 if i==age_limite else 0 for i in df['age_limite']]
        my_df['priorite'] = tmp
     
    # 3- Ajout du réalisateur
    tmp = my_df['priorite'] + [2 if i==director else 0 for i in df['director_name']]
    my_df['priorite'] = tmp
    
    # 4- Ajout des acteurs
    tmp = my_df['priorite'] + [1 if i==act1 else 0 for i in df['actor_1_name']]
    tmp+= [1 if i==act1 else 0 for i in df['actor_2_name']]
    tmp+= [1 if i==act1 else 0 for i in df['actor_3_name']]
    my_df['priorite'] = tmp
    
    tmp = my_df['priorite'] + [1 if i==act2 else 0 for i in df['actor_1_name']]
    tmp+= [1 if i==act2 else 0 for i in df['actor_2_name']]
    tmp+= [1 if i==act2 else 0 for i in df['actor_3_name']]
    my_df['priorite'] = tmp
    
    tmp = my_df['priorite'] + [1 if i==act3 else 0 for i in df['actor_1_name']]
    tmp+= [1 if i==act3 else 0 for i in df['actor_2_name']]
    tmp+= [1 if i==act3 else 0 for i in df['actor_3_name']]
    my_df['priorite'] = tmp
    
    # 5- Ajout de l'audience
    tmp = my_df['priorite'] + [2 if i==audience else 0 for i in df['audience']]
    my_df['priorite'] = tmp
    
    l_id_res = list(my_df.sort_values(['priorite','imdb_score'],ascending=False).head(6)['id'].values)
    l_id_res.remove(id)
    return l_id_res[:5]

def resultat(m_dataframe, id):
    l_res_id = modelisation_finale(m_dataframe, id)
    res = []
    
    for i in l_res_id:
        res.append(retourne_resultat_fichier(m_dataframe, i))
    
    return { "_results": res}

if __name__ == "__main__":
    get_df_value(97576)
