import pandas as pd
from os import path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import re

import os
import sys


class Logs():
    _liste_msg = None
    _logs_loaded = False

    @classmethod
    def __init__(self):
        if not(self._logs_loaded):
            self._liste_msg = []
            self._logs_loaded = True
    
    def _get_msg(self):
        return self._liste_msg
    
    def _set_msg(self, msg, param=''):
        tstamp = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
        self._liste_msg.append('{}{} : {}'.format(param, tstamp, msg))
    
    msg = property(_get_msg, _set_msg)

    def __str__(self):
        s_msg = ''
        for i in self._liste_msg:
            s_msg = s_msg + i +'\n'
        return s_msg
    
    def affiche(self, nb_lignes=30):
        s_msg = ''
        for i in self._liste_msg[-nb_lignes:]:
            s_msg = s_msg + i +'\n'
        return s_msg
    
    def add_log(self, msg, param=''):
        self._set_msg(msg, param)
    
    def __call__(self, msg, param=''):
        self._set_msg(msg, param)
        
    def remove_msg(self, nb_lignes_a_conserver = None):
        if nb_lignes_a_conserver == None:
            self._liste_msg = []
        else:
            self._liste_msg = self._liste_msg[-nb_lignes_a_conserver:]

class Aeroport():
    """Classe de Stockage de la liste des Aeroports"""
    _nom_fichier = 'aeroports_nettoyes.csv'
    _file_loaded = False

    # Lien sur les vols
    _vols = None
    
    # Mes Logs
    #_logs = None
    _logs_param = 'AEROPORT : '

    @classmethod
    def __init__(self, nom_fichier = None):
        self._logs = Logs()
        self._logs('__init__ : DEBUT', param = self._logs_param)

        if nom_fichier!=None:
            self._nom_fichier = nom_fichier
            
        if not(self._file_loaded):
            OPERATION_VALIDE = True
            try:
                self._data = pd.read_csv(self._nom_fichier,index_col='AIRPORT_SEQ_ID')
            except:
                OPERATION_VALIDE = False
                msg = '__init__ : Erreur de chargement de {}'.format(self._nom_fichier)
                print(msg)
                self._logs(msg, param = self._logs_param)
            finally:
                if (OPERATION_VALIDE):
                    self._file_loaded = True
                    path_dir = path.dirname(self._nom_fichier)
                    path_basename = path.basename(self._nom_fichier)
                    msg = "__init__ : dirname = '{}'".format(path_dir)
                    print(msg)
                    self._logs(msg, param = self._logs_param)

                    msg = "__init__ : Fichier '{}' chargé --> Taille : {}".format(
                        path_basename, len(self._data))
                    print(msg)
                    self._logs(msg, param = self._logs_param)


                    #print("-- AEROPORT : Fichier '{}' chargé --> Taille : {}".format(
                    #    path_basename, len(self._data)))
                    #self._logs("-- AEROPORT : dirname = '{}'".format(path_dir))
                    #self._logs("-- AEROPORT : Fichier '{}' chargé --> Taille : {}".format(
                    #   path_basename, len(self._data)))
        else:
            msg = '-- AEROPORT : FICHIER DEJA CHARGE'
            print(msg)
            self._logs(msg, param = self._logs_param)
    
    def get_info_by_id(self, id, feature = None):
        if feature==None:
            return self._data.loc[id]
        else:
            return self._data.loc[id, feature]
    
    def get_info_by_feature(self, feature_name, feature_value):
        if (feature_name in self._data.columns):
            return self._data[self._data[feature_name]==feature_value]
        
    def __getitem__(self, index):
        return self._data.loc[index]
    
    def get_ville(self, ville = None, with_code = False):
        def Uniq(input):
            output = []
            for x in input:
                if x not in output:
                    if x!=None:
                        output.append(x)
            return output
        
        test_ville = (lambda x: x if (ville.lower() in x['name'].lower() ) else None)
            
        aa = self._data.sort_values(by=['CITY_NAME'], ascending=True).reset_index()
        if with_code==True:
            f = lambda x: {'name':x['CITY_NAME'], 'code':x[ 'AIRPORT_SEQ_ID'] }
        else:
            f = lambda x: {'name':x['CITY_NAME']}
                
        if ville==None:
            res = aa.apply(lambda row: f(row), axis=1)
        else:
            res = aa.apply(lambda row:test_ville(f(row)), axis=1)
        
        res = res.tolist()
        return Uniq(res)
    

    def _get_state(self):
        "Renvoie l'etat de chargement du DataSet"
        if self._file_loaded:
            return True
        else:
            return False
    
    def _get_data(self):
        "Renvoie les datas"
        return self._data
    
    #@staticmethod
    def _get_vols(self):
        "Renvoie les vols liés"
        return self._vols
    
    #@staticmethod
    def _set_vols(self, Vols):
        "Enregistre en lien faible les vols lies"
        self._vols = Vols
        
    state = property(_get_state)
    data = property(_get_data)
    vols = property(_get_vols, _set_vols)
    
    def _get_logs(self):
        return self._logs

    logs = property(_get_logs)

    def ajoute_logs(self, msg):
        self._logs(msg, param = self._logs_param)

    def get_destinations(self, ville_origine):
        l_ville_dep = self.get_ville(ville=ville_origine, with_code = True)
        l_res = []
        for ville in l_ville_dep:
            code = ville['code']
            subset = self._vols.data[self._vols.data.ORIGIN_AIRPORT_SEQ_ID==code]['DEST_AIRPORT_SEQ_ID'].unique()

            subset2 = self._data.loc[subset, 'CITY_NAME'].values
            l_res.extend(subset2)
        return sorted(list(set(subset2)))
    
class Vols():
    """Classe de Stockage de la liste des Aeroports"""
    _nom_fichier = 'datas_total_nettoyees-Q4.csv'
    _file_loaded = False
    
    # Lien sur les aeroports
    _aeroports = None

    # dataframe de vols
    _data = None
    # dataframe de vols indéxés par Id de Départ / Id d'Arrivée / Date de Départ
    _data_indexed = None

    # Mes Logs
    _logs_param = 'VOLS : '
    #_logs = None

    @classmethod
    def __init__(self, nom_fichier = None):
        self._logs = Logs()
        self._logs('__init__ : DEBUT', param = self._logs_param)

        if nom_fichier!=None:
            self._nom_fichier = nom_fichier
            
        if not(self._file_loaded):
            OPERATION_VALIDE = True
            try:
                self._data = pd.read_csv(self._nom_fichier)
            except:
                OPERATION_VALIDE = False
                msg = '__init__ : Erreur de chargement de {}'.format(self._nom_fichier)
                print(msg)
                self._logs(msg, param = self._logs_param)
            finally:
                if (OPERATION_VALIDE):
                    self._file_loaded = True
                    path_dir = path.dirname(self._nom_fichier)
                    path_basename = path.basename(self._nom_fichier)

                    msg = "__init__ : dirname = '{}'".format(path_dir)
                    print(msg)
                    self._logs(msg, param = self._logs_param)

                    msg = "__init__ : Fichier '{}' chargé --> Taille : {}".format(
                        path_basename, len(self._data))
                    print(msg)
                    self._logs(msg, param = self._logs_param)
                    
        else:
            msg = '__init__ : FICHIER DEJA CHARGE'
            print(msg)
            self._logs(msg, param = self._logs_param)   
            
        # On créer aussi les données indexées
        self._logs('__init__ : AVANT INDEXATION', param = self._logs_param)
        self.create_indexed_flies(self)
        self._logs('__init__ : APRES INDEXATION', param = self._logs_param)
        
    
    def get_info_by_id(self, id, feature = None):
        if feature==None:
            return self._data.loc[id]
        else:
            return self._data.loc[id, feature]
    
    def get_info_by_feature(self, feature_name, feature_value):
        if (feature_name in self._data.columns):
            return self._data[self._data[feature_name]==feature_value]
    
    ## Methode pour créer un 2e dataframe indexé par Id de Départ / Id d'Arrivée / Date de Départ
    def create_indexed_flies(self):
        self._data_indexed = self._data.set_index(['ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'FL_DATE'])
        self._data_indexed.sort_index(level=['ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'FL_DATE'],
                                      inplace=True)    
    
    # Les items renvoient les données indexées par Id de Départ / Id d'Arrivée / Date de Départ
    def __getitem__(self, index):
        return self._data_indexed.loc[index]
    
    def get_ville(self, ville = None, with_code = False):
        def Uniq(input):
            output = []
            for x in input:
                if x not in output:
                    if x!=None:
                        output.append(x)
            return output
        
        test_ville = (lambda x: x if (ville.lower() in x['name'].lower() ) else None)
            
        aa = self._data.sort_values(by=['CITY_NAME'], ascending=True).reset_index()
        if with_code==True:
            f = lambda x: {'name':x['CITY_NAME'], 'code':x[ 'AIRPORT_SEQ_ID'] }
        else:
            f = lambda x: {'name':x['CITY_NAME']}
                
        if ville==None:
            res = aa.apply(lambda row: f(row), axis=1)
        else:
            res = aa.apply(lambda row:test_ville(f(row)), axis=1)
        
        res = res.tolist()
        return Uniq(res)
    

    def _get_state(self):
        if self._file_loaded:
            return True
        else:
            return False
    
    def _get_data(self):
        return self._data

    def _get_data_indexed(self):
        return self._data_indexed
    
    #@staticmethod
    def _get_aeroports(self):
        return self._aeroports
    
    #@staticmethod
    def _set_aeroports(self, Aeroport):
        self._aeroports = Aeroport
        
    state = property(_get_state)
    data = property(_get_data)
    data_indexed = property(_data_indexed)
    aeroport = property(_get_aeroports, _set_aeroports)

    def _get_logs(self):
        return self._logs

    logs = property(_get_logs)
    
    def ajoute_logs(self, msg):
        self._logs(msg, param = self._logs_param)

    ## Methodes pour la gestion des vols
    def get_vols(self, ville_origine = 'Dallas, TX',
                 ville_arrivee = 'Chicago, IL',
                 colums_sortie = ['ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                                  'FL_DATE', 'AIRLINE_ID', 'ARR_HOUR', 'DEP_HOUR'],
                 data_recherche = {'date_depart': None, 'airline_id': None, 
                                   'airline_id': None, 
                                   'heure_dep': None, 'heure_arr': None}
                ):
        self._logs('get_vols : DEBUT')
        l_vols_possibles = []
        get_code = lambda x: x['code']
        
        l_ville_dep = self._aeroports.get_ville(ville=ville_origine, with_code = True)
        l_ville_arr = self._aeroports.get_ville(ville=ville_arrivee, with_code = True)
        
        l_ville_dep = [get_code(x) for x in l_ville_dep]
        l_ville_arr = [get_code(x) for x in l_ville_arr]
        
        #subset = self[l_ville_dep, l_ville_arr, :]
        if data_recherche['date_depart'] == None:
            subset = self[l_ville_dep, l_ville_arr, :]
        else:
            subset = self[l_ville_dep, l_ville_arr, data_recherche['date_depart']]

        subset = subset.reset_index().groupby(colums_sortie
                                   ).size().reset_index(name='Freq')
        self._logs('get_vols : FIN')
        return subset





################################################
#####   METHODES SUR LA MODELISATION    ########
################################################

### PRE-MODELISATION_1 : SELEMENT AVEC LA DISTANCE COMME FEATURE NUMERIQUE
def pre_modelisation_1(self, l_features_numeriques=None, **kwargs):
    #name = self.__name__
    name = "pre_modelisation_1"
    
    data=kwargs.get('data', self.data)
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))

    from datetime import datetime
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    self.ajoute_logs('')
    log_info('-- DEBUT de Modélisation : MODELISATION1')

    ## 1 - Encoding
    from sklearn.preprocessing import OneHotEncoder
    log_info('\t1 - Encoding')

    ### 1-1 : Categorisation
    log_info('\t\t1-1 : Categorisation')
    categDF = data[['MONTH', 'DAY_OF_MONTH', 'ORIGIN_AIRPORT_SEQ_ID', 
                     'DEST_AIRPORT_SEQ_ID', 'ARR_HOUR', 'DEP_HOUR', 
                     'FL_NUM', 'DAY_OF_WEEK']] # Categorical features

    encoder = OneHotEncoder() # Create encoder object
    categDF_encoded = encoder.fit_transform(categDF) # Can't convert this to dense array: too large!

    ### 1-2 : Valeurs quantitatives
    log_info('\t\t1-2 : Valeurs quantitatives')
    from scipy import sparse # Need this to create a sparse array
    scalingDF = data[['DISTANCE']]
    scalingDF_sparse = sparse.csr_matrix(scalingDF)

    ### 1-3 : Regroupement Quali / Quanti
    log_info('\t\t1-3 : Regroupement Quali / Quanti')
    x_final = sparse.hstack((scalingDF_sparse, categDF_encoded))
    y_final = data['ARR_DELAY'].values

    ## 2 - Séparation des données Entrainement / Test
    log_info('\t2 - Séparation des données Entrainement / Test')
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x_final,y_final,test_size = 0.2,random_state = 0) # Do 80/20 split

    ## 3 - Standardisation des valeurs numériques
    log_info('\t3 - Standardisation des valeurs numériques')
    ### 3-1 : Extraction de la partie numérique
    log_info('\t\t3-1 : Extraction de la partie numérique')
    x_train_numerical = x_train[:, 0:1].toarray() 
        # We only want the first two features which are the numerical ones.
    x_test_numerical = x_test[:, 0:1].toarray()

    ### 3-2 : Standardisation des valeurs numériques
    log_info('\t\t3-2 : Standardisation des valeurs numériques')
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler() # create scaler object
    scaler.fit(x_train_numerical) # fit with the training data ONLY

    ### 3-3 : Matrices de compression
    log_info('\t\t3-3 : Matrices de compression')
    x_train_numerical = sparse.csr_matrix(scaler.transform(x_train_numerical)) # Transform the data and convert to sparse
    x_test_numerical = sparse.csr_matrix(scaler.transform(x_test_numerical))

    ### 3-4 : Injection des valeurs numériques transformées
    log_info('\t\t3-4 : Injection des valeurs numériques transformées')
    x_train[:, 0:1] = x_train_numerical
    x_test[:, 0:1] = x_test_numerical

    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de Modélisation : ', param = tstamp2-tstamp1) 

    trainy_data = {'x_train' : x_train, 
                   'y_train' : y_train, 
                   'x_test' : x_test,
                   'y_test' : y_test}
    models = {'OneHotEncoder': encoder, 'StandardScaler': scaler}
    
    #return x_train, y_train, x_test, y_test, tstamp2-tstamp1
    return trainy_data, models

### Mainenant on l'assigne
Vols.pre_modelisation_1 = pre_modelisation_1



### REGRESSION LINEAIRE : avec les datas en ligne
def training_LinearRegression_v1(self, x_train, y_train, x_test, y_test):
    #name = self.__name__
    name = "training_LinearRegression_v1"
    
    #if (data == None):
    #   data = self.data
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))
        
    def rmse(np1, np2):
        return np.sqrt(np.mean((np1-np2) ** 2))

    from datetime import datetime
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    self.ajoute_logs('')
    log_info('-- DEBUT : training_LinearRegression')
    
    ## 1 - Entrainement
    log_info('\t1 - Entrainement')
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    
    ## 2 - Mesures
    log_info('\t2 - Mesures')
    from sklearn.metrics import mean_absolute_error
    baseline_error_MAE = mean_absolute_error(y_test, lr.predict(x_test))
    baseline_error_RMSE = rmse(y_test, lr.predict(x_test))
    log_info('\t== MAE = {}\n\t== RMSE = {}'.format(baseline_error_MAE, baseline_error_RMSE))
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN : training_LinearRegression : ', param = tstamp2-tstamp1) 
    
    return baseline_error_MAE, baseline_error_RMSE, tstamp2-tstamp1

Vols.training_LinearRegression_v1 = training_LinearRegression_v1
    

### REGRESSION LINEAIRE : avec les données en dictionnaire
def training_LinearRegression(self, trainy_data, models):
    #name = self.__name__
    name = "training_LinearRegression"
    
    x_train = trainy_data['x_train']
    y_train = trainy_data['y_train']
    x_test = trainy_data['x_test']
    y_test = trainy_data['y_test']
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))
        
    def rmse(np1, np2):
        return np.sqrt(np.mean((np1-np2) ** 2))

    from datetime import datetime
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    self.ajoute_logs('')
    log_info('-- DEBUT : training_LinearRegression')
    
    ## 1 - Entrainement
    log_info('\t1 - Entrainement')
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    
    ## 2 - Mesures
    log_info('\t2 - Mesures')
    from sklearn.metrics import mean_absolute_error
    baseline_error_MAE = mean_absolute_error(y_test, lr.predict(x_test))
    baseline_error_RMSE = rmse(y_test, lr.predict(x_test))

    log_info('\t== MAE = {}'.format(baseline_error_MAE))
    log_info('\t== RMSE = {}'.format(baseline_error_RMSE))
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN : training_LinearRegression ->  Tps = {} '.format(tstamp2-tstamp1))

    #models['LinearRegression'] = lr
    #models['baseline_error_MAE'] = baseline_error_MAE
    #models['baseline_error_RMSE'] = baseline_error_RMSE
    
    #models['erreur'] = y_test-lr.predict(x_test)
    #models['erreur2'] = (y_test-lr.predict(x_test))** 2

    models['LinearRegression'] = {}
    models['LinearRegression']['Model'] = lr
    models['LinearRegression']['baseline_error_MAE'] = baseline_error_MAE
    models['LinearRegression']['baseline_error_RMSE'] = baseline_error_RMSE
    
    models['LinearRegression']['erreur'] = y_test-lr.predict(x_test)
    models['LinearRegression']['erreur2'] = (y_test-lr.predict(x_test))** 2
    models['LinearRegression']['tps'] = tstamp2-tstamp1

    #return baseline_error_MAE, baseline_error_RMSE, tstamp2-tstamp1
    return models

Vols.training_LinearRegression = training_LinearRegression


### RIDGE CV
def training_RidgeCV(self, trainy_data, models, cv_nb = 3):
    #name = self.__name__
    name = "training_LinearRegression"
    
    x_train = trainy_data['x_train']
    y_train = trainy_data['y_train']
    x_test = trainy_data['x_test']
    y_test = trainy_data['y_test']
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))
        
    def rmse(np1, np2):
        return np.sqrt(np.mean((np1-np2) ** 2))

    from datetime import datetime
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    self.ajoute_logs('')
    log_info('-- DEBUT : training_LinearRegression')
    
    ## 1 - Entrainement
    log_info('\t1 - Entrainement')
    from sklearn.linear_model import RidgeCV
    ####ridge2 = RidgeCV(fit_intercept=False, alphas=np.logspace(-5, 5, n_alphas), cv=3)
    ridge = RidgeCV(fit_intercept=False, cv = cv_nb)
    #ridge = RidgeCV(fit_intercept=True, cv = cv_nb)
    ridge.fit(x_train, y_train)

    ## 2 - Mesures
    log_info('\t2 - Mesures')
    from sklearn.metrics import mean_absolute_error
    baseline_error_MAE = mean_absolute_error(y_test, ridge.predict(x_test))
    baseline_error_RMSE = rmse(y_test, ridge.predict(x_test))

    log_info('\t== MAE = {}'.format(baseline_error_MAE))
    log_info('\t== RMSE = {}'.format(baseline_error_RMSE))
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN : training_LinearRegression ->  Tps = {} '.format(tstamp2-tstamp1))

    models['RidgeCV'] = {}
    models['RidgeCV']['Model'] = ridge
    models['RidgeCV']['baseline_error_MAE'] = baseline_error_MAE
    models['RidgeCV']['baseline_error_RMSE'] = baseline_error_RMSE
    
    models['RidgeCV']['erreur'] = y_test-ridge.predict(x_test)
    models['RidgeCV']['erreur2'] = (y_test-ridge.predict(x_test))** 2
    models['RidgeCV']['tps'] = tstamp2-tstamp1
    
    #return baseline_error_MAE, baseline_error_RMSE, tstamp2-tstamp1
    return models

Vols.training_RidgeCV = training_RidgeCV


### LASSOCV
def training_LassoCV(self, trainy_data, models, cv_nb = 3):
    #name = self.__name__
    name = "training_LassoCV"
    
    x_train = trainy_data['x_train']
    y_train = trainy_data['y_train']
    x_test = trainy_data['x_test']
    y_test = trainy_data['y_test']
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))
        
    def rmse(np1, np2):
        return np.sqrt(np.mean((np1-np2) ** 2))

    from datetime import datetime
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    self.ajoute_logs('')
    log_info('-- DEBUT : training_LassoCV')
    
    ## 1 - Entrainement
    log_info('\t1 - Entrainement')
    from sklearn.linear_model import LassoCV
    ####ridge2 = RidgeCV(fit_intercept=False, alphas=np.logspace(-5, 5, n_alphas), cv=3)
    lasso = LassoCV(fit_intercept=False, cv=cv_nb)
    #lasso = LassoCV(fit_intercept=True, cv=cv_nb)
    lasso.fit(x_train, y_train)

    ## 2 - Mesures
    log_info('\t2 - Mesures')
    from sklearn.metrics import mean_absolute_error
    baseline_error_MAE = mean_absolute_error(y_test, lasso.predict(x_test))
    baseline_error_RMSE = rmse(y_test, lasso.predict(x_test))

    log_info('\t== MAE = {}'.format(baseline_error_MAE))
    log_info('\t== RMSE = {}'.format(baseline_error_RMSE))
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN : training_LinearRegression ->  Tps = {} '.format(tstamp2-tstamp1))

    models['LassoCV'] = {}
    models['LassoCV']['Model'] = lasso
    models['LassoCV']['baseline_error_MAE'] = baseline_error_MAE
    models['LassoCV']['baseline_error_RMSE'] = baseline_error_RMSE
    
    models['LassoCV']['erreur'] = y_test-lasso.predict(x_test)
    models['LassoCV']['erreur2'] = (y_test-lasso.predict(x_test))** 2
    models['LassoCV']['tps'] = tstamp2-tstamp1
    
    #return baseline_error_MAE, baseline_error_RMSE, tstamp2-tstamp1
    return models

Vols.training_LassoCV = training_LassoCV


##### METHODES NAIVES
### METHODES NAIVES
def training_Naives(self, trainy_data, models):
    #name = self.__name__
    name = "training_Naives"
    
    x_train = trainy_data['x_train']
    y_train = trainy_data['y_train']
    x_test = trainy_data['x_test']
    y_test = trainy_data['y_test']
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))
        
    def rmse(np1, np2):
        return np.sqrt(np.mean((np1-np2) ** 2))

    from datetime import datetime
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    self.ajoute_logs('')
    log_info('-- DEBUT : training_Naives')
    
    ## 1 - Entrainement
    y_train_min = y_train.min()
    y_train_max = y_train.max()
    y_test_count = len(y_test)
    
    y_pred_random = np.random.randint(y_train_min, y_train_max, y_test_count)

    #from sklearn import preprocessing
    #std_scale = preprocessing.StandardScaler().fit(x_train)
    #X_train_std = std_scale.transform(X_train)
    #X_test_std = std_scale.transform(X_test)

    ####### DUMMY1 : MOYENNE
    from sklearn import dummy
    dum1 = dummy.DummyRegressor(strategy='mean')

    # Entraînement
    log_info('\t1 - Entrainement')
    dum1.fit(x_train, y_train)

    # Prédiction sur le jeu de test
    y_pred_dum1 = dum1.predict(x_test)

    ####### DUMMY2 : MEDIANE
    dum2 = dummy.DummyRegressor(strategy='median')

    # Entraînement
    dum2.fit(x_train, y_train)

    # Prédiction sur le jeu de test
    y_pred_dum2 = dum2.predict(x_test)
    
    
    # Evaluate
    #print "RMSE : %.2f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred_dum))

    ## 2 - Mesures
    log_info('\t2 - Mesures')
    from sklearn.metrics import mean_absolute_error
    
    ### RANDOM
    baseline_error_MAE_random = mean_absolute_error(y_test, y_pred_random)
    baseline_error_RMSE_random = rmse(y_test, y_pred_random)

    log_info('\t== RANDOM MAE = {}'.format(baseline_error_MAE_random))
    log_info('\t== RANDOM RMSE = {}'.format(baseline_error_RMSE_random))
    models['NAIVE_RANDOM'] = {}
    models['NAIVE_RANDOM']['baseline_error_MAE'] = baseline_error_MAE_random
    models['NAIVE_RANDOM']['baseline_error_RMSE'] = baseline_error_RMSE_random
    
    models['NAIVE_RANDOM']['erreur'] = y_test-y_pred_random
    models['NAIVE_RANDOM']['erreur2'] = (y_test-y_pred_random)** 2
    
    
    ### MOYENNE
    baseline_error_MAE_mean = mean_absolute_error(y_test, y_pred_dum1)
    baseline_error_RMSE_mean = rmse(y_test, y_pred_dum1)

    log_info('\t== MEAN MAE = {}'.format(baseline_error_MAE_mean))
    log_info('\t== MEAN RMSE = {}'.format(baseline_error_RMSE_mean))
    models['NAIVE_MEAN'] = {}
    models['NAIVE_MEAN']['baseline_error_MAE'] = baseline_error_MAE_mean
    models['NAIVE_MEAN']['baseline_error_RMSE'] = baseline_error_RMSE_mean
    
    models['NAIVE_MEAN']['erreur'] = y_test-y_pred_dum1
    models['NAIVE_MEAN']['erreur2'] = (y_test-y_pred_dum1)** 2
    
    ### MEDIANE
    baseline_error_MAE_median = mean_absolute_error(y_test, y_pred_dum2)
    baseline_error_RMSE_median = rmse(y_test, y_pred_dum2)

    log_info('\t== MEAN MAE = {}'.format(baseline_error_MAE_median))
    log_info('\t== MEAN RMSE = {}'.format(baseline_error_RMSE_median))
    models['NAIVE_MEDIAN'] = {}
    models['NAIVE_MEDIAN']['baseline_error_MAE'] = baseline_error_MAE_median
    models['NAIVE_MEDIAN']['baseline_error_RMSE'] = baseline_error_RMSE_median
    
    models['NAIVE_MEDIAN']['erreur'] = y_test-y_pred_dum2
    models['NAIVE_MEDIAN']['erreur2'] = (y_test-y_pred_dum2)** 2
    
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN : training_Naives ->  Tps = {} '.format(tstamp2-tstamp1))
    
    models['NAIVE_RANDOM']['tps'] = tstamp2-tstamp1
    models['NAIVE_MEAN']['tps'] = tstamp2-tstamp1
    models['NAIVE_MEDIAN']['tps'] = tstamp2-tstamp1
    
    return models

Vols.training_Naives = training_Naives


### METHODE D'EXECUTION 
def execution1(self):
    name = 'execution1'

    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))

    from datetime import datetime
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    self.ajoute_logs('')
    log_info('DEBUT PRE-MODELISATION_1')
    trainy_data, models = self.pre_modelisation_1()

    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN PRE-MODELISATION_1 -> Tps = {} '.format(tstamp2-tstamp1))
    
    self.ajoute_logs('')
    log_info('DEBUT Entrainement LINEAIRE')
    models = self.training_LinearRegression(trainy_data=trainy_data, models=models)

    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN Entrainement LINEAIRE -> Tps = {} '.format(tstamp3-tstamp2))

    self._models = models

Vols.execution1 = execution1




#TODO : IL FAUT AJOUTER HDAYS : KeyError: "['HDAYS'] not in index"
### METHODE D'OPTIMISATION
def pre_modelisation_2(self, l_features_numeriques  = ['DISTANCE', 'HDAYS'], **kwargs):
    ### ON AJOUTE le nom de la compagnie 'AIRLINE_ID' dans les données catégorielles
    name = "pre_modelisation_2"
    
    data=kwargs.get('data', self.data)
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))
        
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                '%Y-%m-%d %H:%M:%S')

    self.ajoute_logs('')
    log_info('-- DEBUT de Modélisation : MODELISATION2')
    
    ## 1 - Encoding
    from sklearn.preprocessing import OneHotEncoder
    log_info('\t1 - Encoding')
    
    ### 1-1 : Categorisation
    log_info('\t\t1-1 : Categorisation')
    categDF = data[['MONTH', 'DAY_OF_MONTH', 'ORIGIN_AIRPORT_SEQ_ID', 
                     'DEST_AIRPORT_SEQ_ID', 'ARR_HOUR', 'DEP_HOUR', 
                     'FL_NUM', 'DAY_OF_WEEK', 'AIRLINE_ID']] # Categorical features
    
    encoder = OneHotEncoder() # Create encoder object
    # Can't convert this to dense array: too large!
    categDF_encoded = encoder.fit_transform(categDF) 
    
    ### 1-2 : Valeurs quantitatives
    log_info('\t\t1-2 : Valeurs quantitatives')
    from scipy import sparse # Need this to create a sparse array
    #l_features_numeriques  = ['DISTANCE', 'HDAYS']
    #l_features_numeriques  = ['DISTANCE']
    
    scalingDF = data[l_features_numeriques]
    scalingDF_sparse = sparse.csr_matrix(scalingDF)
    
    ### 1-3 : Regroupement Quali / Quanti
    log_info('\t\t1-3 : Regroupement Quali / Quanti')
    x_final = sparse.hstack((scalingDF_sparse, categDF_encoded))
    y_final = data['ARR_DELAY'].values
    
    ## 2 - Séparation des données Entrainement / Test
    log_info('\t2 - Séparation des données Entrainement / Test')
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x_final,y_final,test_size = 0.2,random_state = 0) # Do 80/20 split
    
    ## 3 - Standardisation des valeurs numériques
    log_info('\t3 - Standardisation des valeurs numériques')
    ### 3-1 : Extraction de la partie numérique
    log_info('\t\t3-1 : Extraction de la partie numérique')
    x_train_numerical = x_train[:, 0:len(l_features_numeriques)].toarray() 
        # We only want the first two features which are the numerical ones.
    x_test_numerical = x_test[:, 0:len(l_features_numeriques)].toarray()
    
    ### 3-2 : Standardisation des valeurs numériques
    log_info('\t\t3-2 : Standardisation des valeurs numériques')
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler() # create scaler object
    scaler.fit(x_train_numerical) # fit with the training data ONLY
    
    ### 3-3 : Matrices de compression
    log_info('\t\t3-3 : Matrices de compression')
    # Transform the data and convert to sparse
    x_train_numerical = sparse.csr_matrix(scaler.transform(x_train_numerical)) 
    x_test_numerical = sparse.csr_matrix(scaler.transform(x_test_numerical))

    ### 3-4 : Injection des valeurs numériques transformées
    log_info('\t\t3-4 : Injection des valeurs numériques transformées')
    x_train[:, 0:len(l_features_numeriques)] = x_train_numerical
    x_test[:, 0:len(l_features_numeriques)] = x_test_numerical
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de Modélisation ->  Tps = {} '.format(tstamp2-tstamp1))
     
    #return x_train, y_train, x_test, y_test, tstamp2-tstamp1
    trainy_data = {'x_train' : x_train, 
                   'y_train' : y_train, 
                   'x_test' : x_test,
                   'y_test' : y_test}
    models = {'OneHotEncoder': encoder, 'StandardScaler': scaler}
    
    #return x_train, y_train, x_test, y_test, tstamp2-tstamp1
    return trainy_data, models

### Mainenant on l'assigne
Vols.pre_modelisation_2 = pre_modelisation_2



### 2e execution : en prenant l'optimisation dans les données du modele
### METHODE D'EXECUTION 
def execution2(self):
    name = 'execution2'

    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))

    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    self.ajoute_logs('')
    log_info('DEBUT PRE-MODELISATION_2')
    trainy_data, models = self.pre_modelisation_2()

    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN PRE-MODELISATION_2 -> Tps = {} '.format(tstamp2-tstamp1))
    
    self.ajoute_logs('')
    log_info('DEBUT Entrainement LINEAIRE')
    models = self.training_LinearRegression(trainy_data=trainy_data, models=models)

    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN Entrainement LINEAIRE -> Tps = {} '.format(tstamp3-tstamp2))

    self._models = models

Vols.execution2 = execution2


######################################
######      PREDICTION
######################################
def prediction1(self, d_input_data, d_model, **kwargs):
    '''Lancement de la prédiction à partir des données d'entree
    INPUT : 
        - d_input_data : Dictionnaire des données nécessaires
        - d_model : Dictionnaire comprend
                - la liste des features numeriques
                - la liste des features categorielles
                - le modèle entrainé de Standardisation
                - le modèle entrainé OneHotEncoder
                - le modèle entrainé de Regression Linéaire
    OUTPUT: 
        - la prédiction de retard en minutes
        '''
    name = 'prediction1'

    data=kwargs.get('data', self.data)

    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))

    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    self.ajoute_logs('')
    log_info('DEBUT PREDICTION')
    
    l_numerical = d_model['l_numerical']
    l_categoriel = d_model['l_categoriel']
    
    ## Traitement des datas numeriques
        #numerical_values = np.c_[distance, hdays]
        # ex: np.c_[130, 14]
        ###TODO : A TESTER
        #
        #a = np.array(l1).reshape(1,len(l1))
        #a
    l_num_values = [d_input_data[idx] for idx in l_numerical ]
    
    try:
        #numerical_values = np.c_[l_num_values]
        numerical_values = np.array(l_num_values).reshape(1,len(l_num_values))
        numerical_values_scaled = d_model['StandardScaler'].transform(numerical_values)
    
    except:
        log_info('!!!!!! Exception sur les données en INPUT1')
        log_info(sys.exc_info()[0])
        return None
    
    
    ## Traitement des datas categorielles
    categorial_values = np.zeros(len(l_categoriel))
    
    for i, idx in enumerate(l_categoriel):
        try:
            categorial_values[i] = d_input_data[idx]
        except:
            log_info('!!!!!! Exception sur les données en INPUT2_1 pour idx = {}'.format(idx))
            log_info(sys.exc_info()[0])
            return None
        
    try:
        categorical_values_encoded = d_model['OneHotEncoder'].transform([categorial_values]).toarray()
    except:
        log_info('!!!!!! Exception sur les données en INPUT2_2')
        log_info(sys.exc_info()[0])
        return None
        
    
    ## Prediction : 
    ### Preparation des données d'entree
    try:
        final_test_example = np.c_[numerical_values_scaled, categorical_values_encoded]
    except:
        log_info('!!!!!! Exception sur la Préparation de Prédiction')
        log_info(sys.exc_info()[0])

    ### Lancement de la prediction
    try:
        pred_delay = d_model['LinearRegression'].predict(final_test_example)
    except:
        log_info('!!!!!! Exception sur la PREDICTION')
        log_info(sys.exc_info()[0])
    
    log_info('Le retard prédit est de {} minutes'.format(pred_delay[0]))
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN PREDICTION -> Tps = {}'.format(tstamp2-tstamp1))
    
    return pred_delay
    
Vols.prediction1 = prediction1

### Prépartion de la prédiction 
def preparation_data_pour_prediction(self, d_model, idx_data):
    '''
    Methode qui renvoie un dictionnaire avec les donnees nécessaires
    pour lancer la prediction : à partir des données du DATAFRAME
    INPUT : 
        d_model : Dictionnaire du modele qui contient
                    la liste des variables numeriques
                    la liste des variables categorielles
        idx_data: Index de la ligne dans le DATAFRAME
    OUTPUT : 
        Renvoie le dictionnaire des données pour lancer la prédiction
    '''
    
    # Il faut récupérer les datas du modèle
    l_numerical = d_model['l_numerical'].copy()
    l_categoriel = d_model['l_categoriel'].copy()
    
    ### LE RESULTAT EST LA
    d_input_data = {}
    
    l_numerical.extend(l_categoriel)
    
    for idx in l_numerical:
        d_input_data[idx] = self._data.loc[idx_data, idx]
    
    return d_input_data

Vols.preparation_data_pour_prediction = preparation_data_pour_prediction




#### PRE-MODELISATION_3 
def pre_modelisation_3(self, 
                       l_features_numeriques  = ['DISTANCE', 'HDAYS'], 
                       l_features_categoriel  = ['MONTH', 'DAY_OF_MONTH', 'ORIGIN_AIRPORT_SEQ_ID', 
                                                 'DEST_AIRPORT_SEQ_ID', 'ARR_HOUR', 'DEP_HOUR', 
                                                 'FL_NUM', 'DAY_OF_WEEK', 'AIRLINE_ID'], 
                       **kwargs):

    ### ON AJOUTE le nom de la compagnie 'AIRLINE_ID' dans les données catégorielles
    name = "pre_modelisation_3"
    
    data=kwargs.get('data', self.data)
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))
        
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                '%Y-%m-%d %H:%M:%S')

    self.ajoute_logs('')
    log_info('-- DEBUT de Modélisation : MODELISATION2')
    
    ## 1 - Encoding
    from sklearn.preprocessing import OneHotEncoder
    log_info('\t1 - Encoding')
    
    ### 1-1 : Categorisation
    log_info('\t\t1-1 : Categorisation')
    categDF = data[l_features_categoriel] # Categorical features
    
    encoder = OneHotEncoder() # Create encoder object
    # Can't convert this to dense array: too large!
    categDF_encoded = encoder.fit_transform(categDF) 
    
    ### 1-2 : Valeurs quantitatives
    log_info('\t\t1-2 : Valeurs quantitatives')
    from scipy import sparse # Need this to create a sparse array
    import warnings
    from scipy.sparse import SparseEfficiencyWarning
    warnings.simplefilter('ignore',SparseEfficiencyWarning)
    #l_features_numeriques  = ['DISTANCE', 'HDAYS']
    #l_features_numeriques  = ['DISTANCE']
    
    scalingDF = data[l_features_numeriques]
    scalingDF_sparse = sparse.csr_matrix(scalingDF)
    
    ### 1-3 : Regroupement Quali / Quanti
    log_info('\t\t1-3 : Regroupement Quali / Quanti')
    x_final = sparse.hstack((scalingDF_sparse, categDF_encoded))
    y_final = data['ARR_DELAY'].values
    
    ## 2 - Séparation des données Entrainement / Test
    log_info('\t2 - Séparation des données Entrainement / Test')
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x_final,y_final,test_size = 0.2,random_state = 0) # Do 80/20 split
    
    ## 3 - Standardisation des valeurs numériques
    log_info('\t3 - Standardisation des valeurs numériques')
    ### 3-1 : Extraction de la partie numérique
    log_info('\t\t3-1 : Extraction de la partie numérique')
    x_train_numerical = x_train[:, 0:len(l_features_numeriques)].toarray() 
        # We only want the first two features which are the numerical ones.
    x_test_numerical = x_test[:, 0:len(l_features_numeriques)].toarray()
    
    ### 3-2 : Standardisation des valeurs numériques
    log_info('\t\t3-2 : Standardisation des valeurs numériques')
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler() # create scaler object
    scaler.fit(x_train_numerical) # fit with the training data ONLY
    
    ### 3-3 : Matrices de compression
    log_info('\t\t3-3 : Matrices de compression')
    # Transform the data and convert to sparse
    x_train_numerical = sparse.csr_matrix(scaler.transform(x_train_numerical)) 
    x_test_numerical = sparse.csr_matrix(scaler.transform(x_test_numerical))

    ### 3-4 : Injection des valeurs numériques transformées
    log_info('\t\t3-4 : Injection des valeurs numériques transformées')
    x_train[:, 0:len(l_features_numeriques)] = x_train_numerical
    x_test[:, 0:len(l_features_numeriques)] = x_test_numerical
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de Modélisation ->  Tps = {} '.format(tstamp2-tstamp1))
     
    #return x_train, y_train, x_test, y_test, tstamp2-tstamp1
    trainy_data = {'x_train' : x_train, 
                   'y_train' : y_train, 
                   'x_test' : x_test,
                   'y_test' : y_test}
    
    models = {'OneHotEncoder': encoder, 
              'StandardScaler': scaler, 
              'l_numerical': l_features_numeriques, 
              'l_categoriel': l_features_categoriel
             }
    
    #return x_train, y_train, x_test, y_test, tstamp2-tstamp1
    return trainy_data, models

### Mainenant on l'assigne
Vols.pre_modelisation_3 = pre_modelisation_3




###########################################################################
####            EXECUTION 3 : Lancement + Prédiction 
###########################################################################

## On ajoute plus de parametrage
def execution3(self, **kwargs):
    name = 'execution3'

    data=kwargs.get('data', self.data)
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))

    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    
    ## PRE MODELISATION
    self.ajoute_logs('')
    log_info('DEBUT PRE-MODELISATION_3')
    
    # Liste des features 
    l_features_numeriques=['DISTANCE', 'HDAYS']
    l_features_categoriel=['MONTH', 'DAY_OF_MONTH', 
                           'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                           'ARR_HOUR', 'DEP_HOUR', 'FL_NUM', 
                           'DAY_OF_WEEK', 'AIRLINE_ID']
    
    trainy_data, models = self.pre_modelisation_3(
        l_features_numeriques= l_features_numeriques, 
        l_features_categoriel=l_features_categoriel, data=data
    )
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN PRE-MODELISATION_3 -> Tps = {} '.format(tstamp2-tstamp1))
    
    ## MODELISATION PAR REGRESSION LINEAIRE
    self.ajoute_logs('')
    log_info('DEBUT Entrainement LINEAIRE')
    models = self.training_LinearRegression(trainy_data=trainy_data, models=models)

    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN Entrainement LINEAIRE -> Tps = {} '.format(tstamp3-tstamp2))
    
    self._models = models
    
    ## On va tester la fonction de prediction 
    idx_data = 1
    d_input_data = self.preparation_data_pour_prediction(d_model=models , idx_data=idx_data)
    
    # On lance la prediction
    pred_delay = self.prediction1(d_input_data= d_input_data, d_model=models)
    

Vols.execution3 = execution3



##### SEPARATION EN 2 PARTIES

## On ajoute plus de parametrage
def execution3_1_bis(self, 
                     l_features_numeriques=['DISTANCE'],
                     l_features_categoriel=['MONTH', 'DAY_OF_MONTH', 
                                            'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                                            'ARR_HOUR', 'DEP_HOUR', 'FL_NUM', 
                                            'DAY_OF_WEEK', 'AIRLINE_ID'], 
                    **kwargs):
    name = 'execution3_1_bis'

    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))

    data=kwargs.get('data', self.data)
    ridgeCV_cv_nb=kwargs.get('ridgeCV_cv_nb', 3)
    lassoCV_cv_nb=kwargs.get('lassoCV_cv_nb', 3)

    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    
    ## PRE MODELISATION
    self.ajoute_logs('')
    log_info('DEBUT PRE-MODELISATION_3')
    
    trainy_data, models = self.pre_modelisation_3(
        l_features_numeriques= l_features_numeriques, 
        l_features_categoriel=l_features_categoriel, 
        data = data
    )
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN PRE-MODELISATION_3 -> Tps = {} '.format(tstamp2-tstamp1))
    
    ## MODELISATION PAR REGRESSION LINEAIRE
    self.ajoute_logs('')
    log_info('DEBUT Entrainement LINEAIRE')
    models = self.training_LinearRegression(trainy_data=trainy_data, models=models)
    models = self.training_RidgeCV(trainy_data=trainy_data, models=models, cv_nb = ridgeCV_cv_nb)
    models = self.training_LassoCV(trainy_data=trainy_data, models=models, cv_nb = lassoCV_cv_nb)
    
    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN Entrainement LINEAIRE -> Tps = {} '.format(tstamp3-tstamp2))
    
    return models
    

Vols.execution3_1_bis = execution3_1_bis


## On ajoute plus de parametrage
def execution3_1_bis(self, 
                     l_features_numeriques=['DISTANCE'],
                     l_features_categoriel=['MONTH', 'DAY_OF_MONTH', 
                                            'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                                            'ARR_HOUR', 'DEP_HOUR', 'FL_NUM', 
                                            'DAY_OF_WEEK', 'AIRLINE_ID'], 
                    **kwargs):
    ''' execution3_1_bis:
    Information:
    ------------
        Génére le modèle et le renvoie en sortie
        
    Options:
    --------
        - l_features_numeriques : liste des features numerique à poser
        - l_features_categoriel : listes des features categorielles
        - data : par défaut ce sont les datas de la classe qui sont utilisées
        - ridgeCV_cv_nb : nb de mainfolds pour le RidgeCV (3)
        - lassoCV_cv_nb : nb de mainfolds pour le LASSO (3)
        - is_linear : pour savoir si on passe par une regression linéaire
        - is_ridge :
        - is_lasso :
        - is_naive : 
    
    Output:
    -------
        - Modele : avec toutes les informations souhaitées (selon les choix de modélisation)
    
    '''
    name = 'execution3_1_bis'

    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))

    l_options = ['data', 'ridgeCV_cv_nb', 'lassoCV_cv_nb', 'is_linear', 'is_ridge', 'is_lasso', 'is_naive']

    data=kwargs.get('data', self.data)
    ridgeCV_cv_nb=kwargs.get('ridgeCV_cv_nb', 3)
    lassoCV_cv_nb=kwargs.get('lassoCV_cv_nb', 3)
    
    is_linear=kwargs.get('is_linear', True)
    is_ridge=kwargs.get('is_ridge', True)
    is_lasso=kwargs.get('is_lasso', True)
    is_naive=kwargs.get('is_naive', True)
    
    for k in kwargs.keys():
        if k not in l_options:
            print("** WARNING *** Paramètre '{}' non reconnue".format(k))
            print('\t-> Les Paramètres reconnus sont ', l_options)

    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    
    ## PRE MODELISATION
    self.ajoute_logs('')
    log_info('DEBUT PRE-MODELISATION_3')
    
    trainy_data, models = self.pre_modelisation_3(
        l_features_numeriques= l_features_numeriques, 
        l_features_categoriel=l_features_categoriel, 
        data = data
    )
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN PRE-MODELISATION_3 -> Tps = {} '.format(tstamp2-tstamp1))
    
    ## MODELISATION PAR REGRESSION LINEAIRE
    self.ajoute_logs('')
    log_info('DEBUT Entrainement LINEAIRE')
    
    if is_linear:
        models = self.training_LinearRegression(trainy_data=trainy_data, models=models)
    if is_ridge:
        models = self.training_RidgeCV(trainy_data=trainy_data, models=models, cv_nb = ridgeCV_cv_nb)
    if is_lasso:
        models = self.training_LassoCV(trainy_data=trainy_data, models=models, cv_nb = lassoCV_cv_nb)
    if is_naive:
        models = self.training_Naives(trainy_data=trainy_data, models=models)
    
    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN Entrainement LINEAIRE -> Tps = {} '.format(tstamp3-tstamp2))
    
    ### TESTS
    models['M_DATA'] = trainy_data
    
    return models
    

Vols.execution3_1_bis = execution3_1_bis

### PARTIE DE PREDICTION
def execution3_2(self, models, idx_data = 1, **kwargs ):
    name = 'execution3_2'

    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))

    data=kwargs.get('data', self.data)

    ## On va tester la fonction de prediction 
    d_input_data = self.preparation_data_pour_prediction(d_model=models , idx_data=idx_data)
    
    # On lance la prediction
    pred_delay = self.prediction1(d_input_data= d_input_data, d_model=models, data = data)
    return pred_delay

Vols.execution3_2 = execution3_2





############## CONSERVATION : MAIS NORMALEMENT RIEN D'INTERESSANT
## On ajoute plus de parametrage
def execution3_1_OLD(self, **kwargs):
    name = 'execution3_1_OLD'

    data=kwargs.get('data', self.data)
    
    def log_info(msg, param=''):
        self.ajoute_logs('{} : {} {}'.format(name, msg, param))
        
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    
    ## PRE MODELISATION
    self.ajoute_logs('')
    log_info('DEBUT PRE-MODELISATION_3')
    
    # Liste des features 
    #l_features_numeriques=['DISTANCE', 'HDAYS']
    l_features_numeriques=['DISTANCE']
    l_features_categoriel=['MONTH', 'DAY_OF_MONTH', 
                           'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                           'ARR_HOUR', 'DEP_HOUR', 'FL_NUM', 
                           'DAY_OF_WEEK', 'AIRLINE_ID']
    
    trainy_data, models = self.pre_modelisation_3(
        l_features_numeriques= l_features_numeriques, 
        l_features_categoriel=l_features_categoriel,
        data = data
    )
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN PRE-MODELISATION_3 -> Tps = {} '.format(tstamp2-tstamp1))
    
    ## MODELISATION PAR REGRESSION LINEAIRE
    self.ajoute_logs('')
    log_info('DEBUT Entrainement LINEAIRE')
    models = self.training_LinearRegression(trainy_data=trainy_data, models=models)

    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    log_info('FIN Entrainement LINEAIRE -> Tps = {} '.format(tstamp3-tstamp2))
    
    return models
    

Vols.execution3_1_OLD = execution3_1_OLD

def testX():
        l_features_numeriques=[
            #'DIVERTED'
            'DISTANCE', 
            #'DIFF_CRS'
            #'DIFF_CRS_DEP_PRECEDENT'
            #'HDAYS',
            #'TOTAL_ADD_GTIME_bis', 
            #'LONGEST_ADD_GTIME_bis', 
            #'FIRST_DEP_TIME_bis'
        ]
        l_features_categoriel=[
            'QUARTER',
            'MONTH', 'DAY_OF_MONTH',
            'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
            'ARR_HOUR', 'DEP_HOUR', 
            'DAY_OF_WEEK', 

            #'AIRLINE_ID'
        ]

        return l_features_numeriques, l_features_categoriel

##### TESTS PONCTUELS
def Mon_execution_OLD(self, m_data, fonction_feature=testX, **kwargs):
    from datetime import datetime
    
    
    
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    print('\tDEBUT BLOC -- ', tstamp1)

    #l_features_numeriques, l_features_categoriel= testX()
    l_features_numeriques, l_features_categoriel= fonction_feature()

    m_model = self.execution3_1_bis(l_features_numeriques= l_features_numeriques,
                                     l_features_categoriel = l_features_categoriel, 
                                     data=m_data, **kwargs)

    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    try:
        print('\t\t-- LinearRegression : ', m_model['LinearRegression']['baseline_error_RMSE'])
    except:
        pass
    
    try:
        print('\t\t-- RidgeCV          : ', m_model['RidgeCV']['baseline_error_RMSE'])
    except:
        pass
    
    try:
        print('\t\t-- LassoCV          : ', m_model['LassoCV']['baseline_error_RMSE'])
    except:
        pass

    try:
        print('\t\t-- Naive Random     : ', m_model['NAIVE_RANDOM']['baseline_error_RMSE'])
        print('\t\t-- Naive Moyenne    : ', m_model['NAIVE_MEAN']['baseline_error_RMSE'])
        print('\t\t-- Naive Mediane    : ', m_model['NAIVE_MEDIAN']['baseline_error_RMSE'])
    except:
        pass
    
    print('\tFIN BLOC -- ', tstamp2, ' --- ', tstamp2-tstamp1)
    
    return m_model

def Mon_execution(self, m_data, fonction_feature=testX, **kwargs):
    from datetime import datetime
    
    
    
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    #print('\tDEBUT BLOC -- ', tstamp1)

    #l_features_numeriques, l_features_categoriel= testX()
    l_features_numeriques, l_features_categoriel= fonction_feature()

    m_model = self.execution3_1_bis(l_features_numeriques= l_features_numeriques,
                                     l_features_categoriel = l_features_categoriel, 
                                     data=m_data, **kwargs)

    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    
    #print('\tFIN BLOC -- ', tstamp2, ' --- ', tstamp2-tstamp1)
    
    return m_model

Vols.Mon_execution = Mon_execution


##### FONCTION
def test_fonction(my_data):
    from datetime import datetime
    
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    #u3 = Vols2.data.ARR_DELAY<150
    #u4 = Vols2.data.ARR_DELAY>-30
    #m_data = Vols2.data[u1 & u2].copy()
    #m_data = Vols2.data[].copy()
    u1 = my_data.data.AIRLINE_ID==20304
    u2 = my_data.data.AIRLINE_ID==19393
    u3 = my_data.data.ARR_DELAY>0
    #m_data = Vols2.data[(u1 | u2) & u3 & u4].copy()
    m_data = my_data.data[(u1 | u2) & u3].copy()


    ## MODIFICATION DES FEATURES
    #m_data['DEP_DELAY_GROUP_bis'] = m_data.DEP_DELAY_GROUP+2.0
    #m_data['ARR_DELAY_GROUP_bis'] = m_data.ARR_DELAY_GROUP+2.0
    m_data['FIRST_DEP_TIME_bis'] = m_data.FIRST_DEP_TIME
    m_data['FIRST_DEP_TIME_bis'].fillna(value=0, inplace=True)

    m_data['LONGEST_ADD_GTIME_bis'] = m_data.LONGEST_ADD_GTIME
    m_data['LONGEST_ADD_GTIME_bis'].fillna(value=0, inplace=True)

    m_data['TOTAL_ADD_GTIME_bis'] = m_data.TOTAL_ADD_GTIME
    m_data['TOTAL_ADD_GTIME_bis'].fillna(value=0, inplace=True)

    m_data['CARRIER_DELAY'].fillna(value=0, inplace=True)
    m_data['WEATHER_DELAY'].fillna(value=0, inplace=True)
    m_data['NAS_DELAY'].fillna(value=0, inplace=True)
    m_data['SECURITY_DELAY'].fillna(value=0, inplace=True)
    m_data['LATE_AIRCRAFT_DELAY'].fillna(value=0, inplace=True)

    m_data['TEST'] = - m_data['DEP_DELAY'] + m_data['CARRIER_DELAY'] +  m_data['WEATHER_DELAY'] + \
        m_data['NAS_DELAY']+ m_data['SECURITY_DELAY']+ m_data['LATE_AIRCRAFT_DELAY']
    #####m_data = m_data[m_data.TEST<150].copy()

    ###m_data['DIFF_CRS_DEP_PRECEDENT'].fillna(value=1720, inplace=True)

    m_data['DIFF_CRS'].fillna(value=1720, inplace=True)

    l_id_compagnies = m_data.AIRLINE_ID.unique()
    print('DEBUT GENERAL -- ', tstamp1)

    l_model2_2 = []

    for id in l_id_compagnies:
        print('\n--> ID = ', id)
        m_data2 = m_data[m_data.AIRLINE_ID==id]
        model = my_data.Mon_execution(m_data2)
        model['AIRLINE_ID'] = id
        l_model2_2.append(model)

    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
    print('\nFIN GENERAL -- ', tstamp2, ' --- ', tstamp2-tstamp1)


### AJOUT DE FEATURES
def ajout_Features_CRS_PREC_SUIV(self):
    m_data = self.data
    df2 = m_data[['ORIGIN_AIRPORT_SEQ_ID','FL_DATE',  
                  'CRS_DEP_TIME', 'DEP_HOUR', 'DEP_TIME','AIRLINE_ID']]\
        .sort_values(by=['ORIGIN_AIRPORT_SEQ_ID', 'FL_DATE', 'CRS_DEP_TIME'], 
                     ascending=True)
    
    ## Gestion des Précédents
    ### PRECEDENT
    dh2second = lambda x: None if np.isnan(x) else int(x / 100)*60 + (x%100)

    # IDX
    r = [None]
    r.extend(df2.index.tolist())
    r = r[:-1]
    df2['IDX_PRED'] = r


    # ORIGIN_ID
    r = [None]
    r.extend(df2.ORIGIN_AIRPORT_SEQ_ID.tolist())
    r = r[:-1]
    df2['ORIGIN_AIRPORT_SEQ_ID_PRED'] = r

    # FL_DATE
    r = [None]
    r.extend(df2.FL_DATE.tolist())
    r = r[:-1]
    df2['FL_DATE_PRED'] = r


    # CRS_DEP_TIME
    r = [0]
    r.extend(df2.CRS_DEP_TIME.tolist())
    r = r[:-1]
    r = [dh2second(x) for x in r]
    df2['CRS_DEP_TIME_PRED'] = r


    df2['CRS_DEP_TIME_2'] = df2.CRS_DEP_TIME.apply(lambda x: dh2second(x))
    df2['DIFF_CRS_DEP_PRECEDENT'] = np.where(
        ((df2.ORIGIN_AIRPORT_SEQ_ID==df2.ORIGIN_AIRPORT_SEQ_ID_PRED) & 
        (df2.FL_DATE==df2.FL_DATE_PRED)), 
        df2['CRS_DEP_TIME_2']-df2['CRS_DEP_TIME_PRED'], np.NAN)


    ## SUPPRESSION DES COLONNES INTERMEDIAIRES
    df2.drop(columns=['ORIGIN_AIRPORT_SEQ_ID_PRED', 'FL_DATE_PRED', 'CRS_DEP_TIME_PRED',
                      'CRS_DEP_TIME_2','IDX_PRED', 
                     ], inplace=True)
    
    ## Gestion des Suivants
    ### SUIVANT

    # IDX
    r = df2.index.tolist()[1:]
    r.append(None)
    df2['IDX_SUIV'] = r

    # ORIGIN_ID
    r = df2.ORIGIN_AIRPORT_SEQ_ID.tolist()[1:]
    r.append(None)
    df2['ORIGIN_AIRPORT_SEQ_ID_SUIV'] = r

    # FL_DATE
    r = df2.FL_DATE.tolist()[1:]
    r.append(None)
    df2['FL_DATE_SUIV'] = r

    # CRS_DEP_TIME
    r = df2.CRS_DEP_TIME.tolist()[1:]
    r.append(0)
    df2['CRS_DEP_TIME_SUIV'] = [dh2second(x) for x in r]

    df2['CRS_DEP_TIME_2'] = df2.CRS_DEP_TIME.apply(lambda x: dh2second(x))

    df2['DIFF_CRS_DEP_SUIV'] = np.where(
        ((df2.ORIGIN_AIRPORT_SEQ_ID==df2.ORIGIN_AIRPORT_SEQ_ID_SUIV) & 
        (df2.FL_DATE==df2.FL_DATE_SUIV)), 
        df2['CRS_DEP_TIME_SUIV']-df2['CRS_DEP_TIME_2'], np.NAN)

    df2.drop(columns=['ORIGIN_AIRPORT_SEQ_ID_SUIV', 
                      'FL_DATE_SUIV', 'CRS_DEP_TIME_SUIV',
                      'CRS_DEP_TIME_2', 
                     'IDX_SUIV'], inplace=True)
    
    ## AJOUTE LA COLONNE DE MIN 
    def my_min(a, b):
        if np.isnan(a):
            return b
        elif np.isnan(b):
            return a
        elif a>b:
            return b
        else:
            return a

    df2['MIN_DIFF_CRS'] = df2.apply(lambda row: my_min(row['DIFF_CRS_DEP_PRECEDENT'], 
                                                       row['DIFF_CRS_DEP_SUIV']), axis=1)
    
    #### Sauvegarde des résultats
    m_data['DIFF_CRS_DEP_SUIV'] = df2.sort_index().DIFF_CRS_DEP_SUIV.values
    m_data['DIFF_CRS_DEP_PRECEDENT'] = df2.sort_index().DIFF_CRS_DEP_PRECEDENT.values
    m_data['MIN_DIFF_CRS'] = df2.sort_index().MIN_DIFF_CRS.values
    self._data = m_data

Vols.ajout_Features_CRS_PREC_SUIV = ajout_Features_CRS_PREC_SUIV
