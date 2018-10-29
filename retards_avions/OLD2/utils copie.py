 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
#import matplotlib.pyplot as plt
import json
import os
from pygments import highlight, lexers, formatters
import pprint

basedir = os.path.abspath(os.path.dirname(__file__))



#from .models import charge_df

# Pour afficher les graphiques inline
#%matplotlib inline

def charge_df():
    #FILE = os.path.join(basedir, 'datas/movie_metadata.csv')
    FILE = os.path.join(basedir, 'datas/movie_metadata_nettoye_2.csv')
    
    #FILE = 'datas/movie_metadata.csv'
    df = pd.read_csv(FILE)
    
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

def resultat(m_dataframe, id):
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


if __name__ == "__main__":
    get_df_value(97576)