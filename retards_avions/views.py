from flask import Flask, render_template, request, make_response, url_for
#####from flask.ext.cache import Cache

app = Flask(__name__)

# On ajoute les configurations du serveur 
app.config.from_object('config')


#from .models import MA_BASE
from .utils import get_ville, get_destinations, get_vols, get_date_vols,\
     get_vols_jour, get_prediction

# Gestion des logs
from .utils import affiche_logs, supprime_logs, ajoute_logs

# Gestion du stockage des models
from .utils import prepare_models, chargement_models



from .utils import get_df_value
from .utils import retourne_films_info_by_id, retourne_liste_films_info_by_cat 
from .utils import get_movie_liste_by_name






#from .utils import get_date_active, get_ville2
####from .utils import mon_test2

import json

#import pprint


# Check Configuring Flask-Cache section for more details
####cache = Cache(app,config={'CACHE_TYPE': 'simple'})





   #########################
   #####   POINT D'ENTREE DU FICHIER ACTUEL
   #########################
@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
@app.route('/test5/', methods=['POST', 'GET'])
def index():
    #resp = make_response(render_template('formulaire_index.html'))
    j_villes = json.dumps(get_ville())
    return render_template('formulaire_index.html', 
                           villes = j_villes, 
                           spy = request.args)
    #return resp


### FONCTION DE TEST 
@app.route('/liste_vols_du_jour.html', methods=['GET', 'POST'])
def m_final():
    var = '{} - {} - {}'.format(request.args, request.method, request.form)
    var2 = ''
    if request.method == 'POST':
        if 'ville_depart' in request.form:
            ville_depart = request.form['ville_depart']
            ville_arrivee = request.form['ville_arrivee']
            date_dep = request.form['date_dep']
            #var2 = '{} - {} - {}'.format(ville_depart, ville_arrivee, date_dep)
            
            v1 = get_vols_jour(ville_origine=ville_depart, 
                       ville_arrivee=ville_arrivee, 
                       datedep=date_dep)
    

        def transforme_nom(x):
            def dh_2_hh_mm(u):
                return "{0:02d}:{1:02d}".format(int(int(u)/100),int(u)%100)
        
            item = {}
            item['dep_hour'] = x['CRS_DEP_TIME']
            item['arr_hour'] = x['CRS_ARR_TIME']
            item['airline_id'] = x['AIRLINE_ID']
            item['label'] = ' DEPART: {}  - ARRIVEE: {} (AIRLINE: n° {})'.format(
                dh_2_hh_mm(x['CRS_DEP_TIME']),
                dh_2_hh_mm(x['CRS_ARR_TIME']),str(x['AIRLINE_ID']))
            item['value'] = '{dep_h}_{arr_h}_{air}'.format(
                    dep_h = x['CRS_DEP_TIME'],
                    arr_h = x['CRS_ARR_TIME'],
                    air = str(x['AIRLINE_ID']))
            
    
            return item
            
        l_res = [transforme_nom(x) for x in v1]
    #####<option value="{{ vol['value'] }}" dep_hour="{{ vol['dep_hour'] }}" arr_hour="{{ vol['arr_hour'] }}" airline_id="{{ vol['airline_id'] }}">{{label}}</option>

        return render_template('liste_vols_du_jour.html',
                           l_vol = l_res)
            
    return render_template('liste_vols_du_jour.html',
                       valeurs = var,
                       valeurs2 = var2)
    
    
    return 'Super 1) {} -- 2) {}'.format(var, request.method)
        
    return request.method

    if request.method == 'POST':
        return request.form
    else:
        return request.args


#### A N'APPELER QUE SUR LE SERVEUR D'ENTRAINEMENT
### CELA VA GENERER LES FICHIERS CONTENANT LES MODELES
@app.route('/activate/')
def enregistre_models():
    prepare_models()
    
    return 'Nous avons enregistré les models'


@app.route('/load/')
def charge_models():
    #chargement_models()
    
    return 'Nous avons chargé les models'






@app.route('/test/')
def test():
    
    v1 = get_vols_jour(ville_origine='Dallas, TX', 
                       ville_arrivee='Chicago, IL', 
                       datedep='2016-10-13')
    

    def transforme_nom(x):
        def dh_2_hh_mm(u):
            return "{0:02d}:{1:02d}".format(int(int(u)/100),int(u)%100)        
        
        item = {}
        item['dep_hour'] = x['CRS_DEP_TIME']
        item['arr_hour'] = x['CRS_ARR_TIME']
        item['airline_id'] = x['AIRLINE_ID']
        item['label'] = ' DEPART : {}  - ARRIVEE : {} (AIRLINE : n° {})'.format(
                dh_2_hh_mm(x['CRS_DEP_TIME']),
                dh_2_hh_mm(x['CRS_ARR_TIME']),str(x['AIRLINE_ID']))
        item['value'] = '{dep_h}_{arr_h}_{air}'.format(
                dep_h = x['CRS_DEP_TIME'],
                arr_h = x['CRS_ARR_TIME'],
                air = str(x['AIRLINE_ID']))
        

        return item
        
    l_res = [transforme_nom(x) for x in v1]
    #####<option value="{{ vol['value'] }}" dep_hour="{{ vol['dep_hour'] }}" arr_hour="{{ vol['arr_hour'] }}" airline_id="{{ vol['airline_id'] }}">{{label}}</option>

    return render_template('liste_vols_du_jour.html',
                           l_vol = l_res)
    


@app.route('/resultat_prediction.html', methods=['GET', 'POST'])
def affichage_resultat_prediction():
    def dh_2_hh_mm(u):
        return "{0:02d}:{1:02d}".format(int(int(u)/100),int(u)%100)
            
    if request.method == 'POST':
        if 'ville_depart' in request.form:
            ville_depart = request.form['ville_depart']
            ville_arrivee = request.form['ville_arrivee']
            date_dep = request.form['date_dep']
            airline_id = request.form['airline_id']
            dep_hour = request.form['dep_hour']
            arr_hour = request.form['arr_hour']
            
            #var2 = '{} - {} - {}'.format(ville_depart, ville_arrivee, date_dep)
            ajoute_logs('--> ICI0')
            retard, prediction = get_prediction(
                    ville_origine = ville_depart, 
                    ville_arrivee = ville_arrivee, 
                    datedep = date_dep,
                    airline_id = airline_id,
                    dep_hour = dep_hour,
                    arr_hour = arr_hour
                    )
            ajoute_logs('--> ICI1')            
            l_res = {}
            l_res['ville_depart'] = ville_depart
            l_res['ville_arrivee'] = ville_arrivee
            
            l_res['dep_hour'] = dh_2_hh_mm(float(dep_hour))
            l_res['arr_hour'] = dh_2_hh_mm(float(arr_hour))
            #l_res['dep_hour'] = dep_hour
            #l_res['arr_hour'] = arr_hour
            l_res['dep_date'] = date_dep
            l_res['airline_id'] = airline_id

            l_res['value'] = "value"
            l_res['label'] = "label"

            l_res['retard'] = retard
            l_res['prediction'] = prediction
            
            ajoute_logs('--> ICI2')
            return render_template('resultat_prediction.html',
                           vol = l_res)
    
    else:
        ajoute_logs('--> ICI3')
        return 'ATTENTION IL Y A UNE ERREUR'
    


@app.route('/contents/')
#@app.route('/', methods=['GET', 'POST'])
#@app.route('/index.html', methods=['GET', 'POST'])
def index_old():
    
    if request.method == 'POST':
        film_name = request.form.get('film_name')
        
        # On gère les caractères vides
        if film_name=='':
            return render_template('index.html',
                               resultat_films=None,
                               film_selectionne=False)
        else:    
            l_film = get_movie_liste_by_name(film_name)
        
            return render_template('index.html',
                               resultat_films=l_film,
                               film_selectionne=True)
    else:
        return render_template('index.html',
                               resultat_films=None,
                               film_selectionne=False)
        
        return render_template('index.html')
    
    return render_template('index.html')

#@app.route('/aeroports/')
#def get_liste_ville():
#    return json.dumps(get_ville())

@app.route('/test_old1/')
def test_old1():
    j_villes = json.dumps(get_ville())
    return render_template('test1.html', 
                           villes = j_villes)


### Gestion des formulaires
@app.route('/formulaire/villes.html')
def villes():
    j_villes = json.dumps(get_ville())
    return render_template('villes.html', 
                           villes = j_villes)

@app.route('/test2_old/')
def test2_old():
    l_date_activee = get_date_active()
    resp = make_response(render_template('test2.html',
                         date_activee = l_date_activee))
    return resp

    #return render_template('test2.html')


@app.route('/test2/')
def test2():
    #return "SUPER"
    #ville_depart = request.args['ville_depart']
    #ville_arrivee = request.args['ville_arrivee']
    #if ville_depart=='':
    #    return json.dumps({"villes": {}})
    #if ville_arrivee=='':
    #    return json.dumps({"villes": get_destinations(
    #        ville_origine = ville_depart)})
    
    #l_date_vols = get_date_vols(
    #       ville_origine = ville_depart,
    #       ville_arrivee = ville_arrivee)
    
    def transforme_date(date_in, fmt = '%Y-%m-%d'):
        from datetime import datetime
        u = datetime.strptime(date_in, fmt)
        return '{m}-{d}-{y}'.format(y = u.year, m = u.month, d = u.day)


    l_date_vols = get_date_vols(
            ville_origine = 'Dallas, TX',
            ville_arrivee = 'Chicago, IL')
    
    l_date_activee = [transforme_date(x) for x in l_date_vols ]
    resp = make_response(render_template('date_depart.html',
                         date_activee = l_date_activee, 
                         mdate_min = min(l_date_activee),
                         mdate_max = max(l_date_activee)
                         ))
    return resp



## Liste des aeroports : JSON
@app.route('/vol/aeroports/', methods=['POST', 'GET'])
def get_liste_ville():
    #return request.method
    #ville_depart = request.args['ville_depart']
    
    ## Il faudra peut être modifier ici
    return json.dumps(get_ville())

## Identification des destinations possibles pour une ORIGINE
@app.route('/vol/destinations_possibles/', methods=['POST', 'GET'])
def get_liste_ville_arrivee():
    ville_depart = request.args['ville_depart']
    if ville_depart=='':
        return json.dumps({"villes": {}})
    else:
        return json.dumps({"villes": get_destinations(
            ville_origine = ville_depart)})
    
    
## TEST EN ATTENDANT VALIDATION
@app.route('/test3/')
def result3():
    return mon_test2()


## TEST EN ATTENDANT VALIDATION
@app.route('/test4/', methods=['POST', 'GET'])
def result4():
    value = request.args
    return json.dumps(value)
    return '{}'.format(value)


## URL pour identifier les dates possibles
@app.route('/vol/dates_possibles2/', methods=['POST', 'GET'])
def get_liste_ville_arrivee2():
    from datetime import datetime
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    
    if "ville_depart" in request.args:
        spy1 = "ICI1"
        ville_depart = request.args['ville_depart']
    else: 
        spy1 = "ICI2"
        ville_depart = 'Dallas, TX'
        
    if "ville_arrivee" in request.args:
        spy2 = "ICI21"
        ville_arrivee = request.args['ville_arrivee']
    else: 
        spy2 = "ICI22"
        ville_arrivee = 'Chicago, IL'

    if "test" in request.args:
        spy3 = request.args['test']
    else:
        spy3 = ''


    spy4 = ''
    
    ## Test sur les valeurs nulles : TODO
    if ville_depart=='':
        return json.dumps({"villes": {}})

    if ville_arrivee=='':
        return json.dumps({"villes": get_destinations(
            ville_origine = ville_depart)})
    
    def transforme_date(date_in, fmt = '%Y-%m-%d'):
        from datetime import datetime
        u = datetime.strptime(date_in, fmt)
        return '{m}-{d}-{y}'.format(y = u.year, m = u.month, d = u.day)
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    l_date_vols = get_date_vols(
            ville_origine = ville_depart,
            ville_arrivee = ville_arrivee)
    
    l_date_activee = [transforme_date(x) for x in l_date_vols ]
    min_date = transforme_date(min(l_date_vols))
    max_date = transforme_date(max(l_date_vols))
    
    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    
    return json.dumps({"spy1": spy1, 
                       "spy2": spy2,
                       "spy3": spy3,
                       "spy4_1": str(tstamp2-tstamp1),
                       "spy4_2": str(tstamp3-tstamp2),
                       
            
                        "date_min": min_date, 
                       "date_max": max_date,
                       "dates_actives": l_date_activee})
    
    return json.dumps({"dates_actives": get_vols(
            ville_origine = ville_depart,
            ville_arrivee = ville_arrivee)})
 

   #########################
   #####   POINT D'ENTREE DU FICHIER ACTUEL
   #########################

@app.route('/test5/', methods=['POST', 'GET'])
def result5():
    #resp = make_response(render_template('formulaire_index.html'))
    j_villes = json.dumps(get_ville())
    return render_template('formulaire_index.html', 
                           villes = j_villes, 
                           spy = request.args)
    #return resp

   #########################
   #####   GESTION DES LOGS
   #########################


## AFFICHAGE DES LOGS
@app.route('/logs/')
def retourne_logs():
    #TODO: REMETTRE AU PROPRES
    #TODO: AJOUTER UN BOUTON VERS LA SUPPRESSION
    #TODO: BOUTON AVEC OPTION DU NOMBRE DE LIGNES
    u = affiche_logs()
    u.replace('\\n', '<br>')
    msg = '{}'.format(u)
    return u
    return msg


## AFFICHAGE DES LOGS
@app.route('/logs_remove/')
def f_supprime_logs():
    supprime_logs()
    #ajoute_logs('JUSTE UN TEST')
    return 'LOGS SUPPRIMES'

#TODO: AJOUTER UNE GESTION D'AJOUT DE LOGS : ajoute_logs(msg)







    
   #############################
   #####   ANCIENNES FONCTIONS
   #############################


#### LE RESTE EST ANCIEN : A SUPPRIMER
@app.route('/recommend/')
def my_default():
    return (get_df_value(449088))



@app.route('/recommend/<int:content_id>/')
def recommend(content_id):
     return (get_df_value(content_id))




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return 'STT : Methode POST'
    else:
        return "STT : Methode GET"



@app.route('/test/')
def test_OLD111111():
    fims_info = retourne_films_info_by_id(449088)
    resp = make_response(render_template('info_one_film.html',
                                         film_info = fims_info))
    resp.set_cookie('username', 'the username')
    return resp


   
@app.route('/groupe/<int:content_id>/')
def renvoie_groupe(content_id):
    '''Affiche les infos d'un groupe de film ayant le même label'''
    liste_film_info = retourne_liste_films_info_by_cat(content_id)
    
    resp = make_response(render_template('info_multi_film.html',
                         liste_film = liste_film_info))
    return resp

    
if __name__ == "__main__":
	app.run()

