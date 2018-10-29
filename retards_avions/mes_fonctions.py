# -*- coding: utf-8 -*-

########################################################
########        INFOS GENERIQUES
########################################################

# Les librairies générales
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import re

#import os
from datetime import datetime

# Les librairies de modélisation
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy import sparse

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn import dummy

DEBUG = False


## Fonction de suivi
if DEBUG:
    def ajoute_logs(msg):
        print(msg)
else:
    def ajoute_logs(msg):
        pass
    
name = 'GENERAL'

def log_info(msg, param=''):
    ajoute_logs('{} : {} {}'.format(name, msg, param))


########################################################
########        FONCTIONS 
########################################################
    
###########################
####    FABRICATION MODELE
###########################
    
### Fonctions à définir
#   fabrication_modele_general : 
#   INPUT : 
#       - data
#       - dict(l_num, l_categ)
#       - isLasso 
#       - isRidge
#
#   INTERMEDIAIRE : 
#       - Fonction de Transfo : DataSet Pandas vers SPARSE
#
#   OUTPUT : 
#       - N_DATA : dict(X_train, Y_train, X_test, Y_test) [ ce sont des DataSet Pandas]
#       - N_Model_Optimisation dict(Encoder, Standard)
#       - N_Model : dict(Lineaire , Lasso, Ridge + avec le tps dans les models)


def fabrication_model_general(data, d_features, 
                              isRidge=False, isLasso=False):
    name = 'fabrication_model_general'
    
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    # Les variables utiles
    l_numerical       = d_features['l_numerical']
    l_categoriel    = d_features['l_categoriel'] 
    
    # Debut
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'Preparation', tstamp1))
    
    
    ## 1 - Entrainement de l'Encodage categoriel
    encoder = OneHotEncoder(sparse=True)
    encoder.fit(data[l_categoriel])
    
    ## JE VEUX CONSERVER TOUTES LES DATAS DANS LE RESTE
    #X_data = data[(l_numerical + l_categoriel)]
    X_data = data
    Y_data = data['ARR_DELAY']
    
    
    ## 2 - Séparation des données 
    X_train, X_test, Y_train, Y_test = train_test_split( 
            X_data, Y_data, test_size = 0.2, random_state = 0)
    
    ## ICI MODIF
    X_train_bis = X_train[(l_numerical + l_categoriel)]
    
    
    ## 3 - Préparation Modélisation Générale
    ## 3_1 - Standardisation des données numériques
    scaler = StandardScaler()

    #### Entrainement
    #scaler.fit(X_train[l_numerical])
    scaler.fit(X_train_bis[l_numerical])
    #### Transformation
    #X_train_numerical = sparse.csr_matrix(scaler.transform(X_train[l_numerical]))
    X_train_numerical = sparse.csr_matrix(scaler.transform(X_train_bis[l_numerical]))

    ## 3_2 - Encodage des données categorielles
    #X_train_categoriel = encoder.transform(X_train[l_categoriel])
    X_train_categoriel = encoder.transform(X_train_bis[l_categoriel])

    ## 3_3 - Fabrication des données optimisées
    #print('X_train_numerical.shape = ', X_train_numerical.shape)
    #print('X_train_categoriel.shape = ', X_train_categoriel.shape)
    
    Opt_X_train = sparse.hstack((X_train_numerical, X_train_categoriel))
    #Opt_X_test  = sparse.hstack(X_test_numerical, X_test_categoriel)
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de {} : {} - {} -> {}\n'.format(name, 'Preparation', tstamp2, tstamp2-tstamp1))
    
    ## 4 - Modélisation
    ## 4_1 - Modélisation Linéaire
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'Modélisation', tstamp2))
    
    ## REGRESSION LINEAIRE
    tstamp_lr1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    lr = LinearRegression()
    lr.fit(Opt_X_train, Y_train)
    tstamp_lr2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    ## VERSION NAIVE
    ## Regression Naive
    tstamp_dum1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    dum = dummy.DummyRegressor(strategy='mean')
    dum.fit(Opt_X_train, Y_train)
    tstamp_dum2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de {} : {} - {} -> {}\n'.format(name, 'Modélisation', tstamp3, tstamp3-tstamp2))

    ## 5 - Sauvegarde des données 
    N_Data = {'X_train': X_train, 'Y_train': Y_train, 
              'X_test': X_test, 'Y_test': Y_test}
    N_Model_Optimisation = {'OneHotEncoder': encoder, 
                            'StandardScaler': scaler}
    N_Model = {'LinearRegression': {'Model':lr, 'Temps':tstamp_lr2-tstamp_lr1}}
    N_Model['Naive'] = {'Model':dum, 'Temps':tstamp_dum2-tstamp_dum1}
    
    if isRidge:
        ridge = RidgeCV(fit_intercept=False, cv = 3)
        tstamp_ridge1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        ridge.fit(Opt_X_train, Y_train)
        tstamp_ridge2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        N_Model['RidgeCV'] = {'Model':ridge, 'Temps':tstamp_ridge2-tstamp_ridge1}
    
    if isLasso:
        tstamp_lasso1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        lasso = LassoCV(fit_intercept=False, cv = 3)
        lasso.fit(Opt_X_train, Y_train)
        tstamp_lasso2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        N_Model['LassoCV'] = {'Model':lasso, 'Temps':tstamp_lasso2-tstamp_lasso1}
    
    return N_Data, N_Model_Optimisation, N_Model


###########################
####    FABRICATION MODELE : Intermediaire
###########################
 

### fabrication_modele_feature_delay
#   INPUT : 
#       - X_train
#       - dict(l_num, l_categ)
#       - nom_feature_cible
#       - isLasso
#       - isRidge
#
#   ATTENTION :
#       - Il faudra verifier avant d'envoyer que les features sont ok
#
#   OUTPUT : 
#       - F_Model_Optimisation : dict(Encoder, Standard)
#       - F_Model : dict(Lineaire, Lasso, Ridge + avec le temps dans les models)

def fabrication_modele_feature_delay(input_X_Train, d_features, n_feature_cible, 
                              isRidge=False, isLasso=False, input_X_test=None):
    name = 'fabrication_modele_feature_delay'
    
    data = input_X_Train
    
    # ON s'assure que la feature est bien presente
    if n_feature_cible not in data.columns:
        log_info('!!!! ERREUR dans {} : feature {} non presente'.format(name, n_feature_cible))
        return None, None
    
    
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    # Les variables utiles
    l_numerical       = d_features['l_numerical']
    l_categoriel    = d_features['l_categoriel'] 
    
    # Debut
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'Preparation', tstamp1))
 
    ## 1 - Entrainement de l'Encodage categoriel
    encoder = OneHotEncoder(sparse=True)
    #encoder.fit(data[l_categoriel])
    #### HACK : on ajoute input_X_test pour éviter que OneHotEncoder 
    ## ne tombe sur des données non encore rencontrées !!!!
    
    ####JE PENSE QUE CELA RAJOUTE BEAUCOUP DE TEMPS
    ###tmp_data = input_X_Train.copy()
    tmp_data = input_X_Train
    tmp_data = tmp_data.append(input_X_test)
    encoder.fit(tmp_data[l_categoriel])
    
    X_data = data[(l_numerical + l_categoriel)]
    Y_data = data[n_feature_cible]

    ## 3 - Préparation Modélisation Générale
    ## 3_1 - Standardisation des données numériques
    scaler = StandardScaler()

    #### Entrainement
    scaler.fit(X_data[l_numerical])
    #### Transformation
    X_data_numerical = sparse.csr_matrix(scaler.transform(X_data[l_numerical]))

    ## 3_2 - Encodage des données categorielles
    X_data_categoriel = encoder.transform(X_data[l_categoriel])

    ## 3_3 - Fabrication des données optimisées
    #print('X_train_numerical.shape = ', X_train_numerical.shape)
    #print('X_train_categoriel.shape = ', X_train_categoriel.shape)
    
    Opt_X_data = sparse.hstack((X_data_numerical, X_data_categoriel))
    #Opt_X_test  = sparse.hstack(X_test_numerical, X_test_categoriel)
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de {} : {} - {} -> {}\n'.format(name, 'Preparation', tstamp2, tstamp2-tstamp1))
 
    ## 4 - Modélisation
    ## 4_1 - Modélisation Linéaire
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'Modélisation', tstamp2))
    
    tstamp_lr1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    lr = LinearRegression()
    lr.fit(Opt_X_data, Y_data)
    tstamp_lr2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
  

    
    
    
    ## Regression Naive
    dum = dummy.DummyRegressor(strategy='mean')
    tstamp_dum1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    dum.fit(Opt_X_data, Y_data)
    tstamp_dum2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')


    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de {} : {} - {} -> {}\n'.format(name, 'Modélisation', tstamp3, tstamp3-tstamp2))

    ## 5 - Sauvegarde des données 
    #N_Data = {'X_train': X_train, 'Y_train': Y_train, 
    #          'X_test': X_test, 'Y_test': Y_test}
    F_Model_Optimisation = {'OneHotEncoder': encoder, 
                            'StandardScaler': scaler}
    F_Model = {'LinearRegression': {'Model':lr, 'Temps':tstamp_lr2-tstamp_lr1}}
    F_Model['Naive'] = {'Model':dum, 'Temps':tstamp_dum2-tstamp_dum1}
    
    if isRidge:
        ridge = RidgeCV(fit_intercept=False, cv = 3)
        tstamp_ridge1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        ridge.fit(Opt_X_data, Y_data)
        tstamp_ridge2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        F_Model['RidgeCV'] = {'Model':ridge, 'Temps':tstamp_ridge2-tstamp_ridge1}
    
    if isLasso:
        lasso = LassoCV(fit_intercept=False, cv = 3)
        tstamp_lasso1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        lasso.fit(Opt_X_data, Y_data)
        tstamp_lasso2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
        F_Model['LassoCV'] = {'Model':lasso, 'Temps':tstamp_lasso2-tstamp_lasso1}
    
    return F_Model_Optimisation, F_Model



###########################
####    Prediction MODELE : Intermediaire
###########################


### prediction_modele_feature_delay
#   INPUT: 
#       - X_test
#       - dict(l_num, l_categ)
#       - F_Model_Optimisation
#       - F_Model 
#       - Type_Model : qui est le type de Regression
#
#   OUTPUT:
#       - Vecteur prédit X_0_x

def prediction_modele_feature_delay(input_X_test, d_features, F_Model_Optimisation, F_Model, Type_Model):
    name = 'prediction_modele_feature_delay'
    
    data = input_X_test
     
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    # Les variables utiles
    l_numerical       = d_features['l_numerical']
    l_categoriel    = d_features['l_categoriel'] 
    
    # Debut
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'Preparation', tstamp1))
    
    ## 1 - Recupération de l'Encodage categoriel
    encoder = F_Model_Optimisation['OneHotEncoder']
    
    X_data = data[(l_numerical + l_categoriel)]

    ## 3 - Préparation Modélisation Générale
    ## 3_1 - Standardisation des données numériques : Récupération
    scaler = F_Model_Optimisation['StandardScaler']

    #         Transformation
    X_data_numerical = sparse.csr_matrix(scaler.transform(X_data[l_numerical]))

    ## 3_2 - Encodage des données categorielles
    X_data_categoriel = encoder.transform(X_data[l_categoriel])

    ## 3_3 - Fabrication des données optimisées
    Opt_X_data = sparse.hstack((X_data_numerical, X_data_categoriel))
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de {} : {} - {} -> {}\n'.format(name, 'Preparation', tstamp2, tstamp2-tstamp1))

    ## 4 - Prédiction
    ## 4_1 - Modélisation Linéaire
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'Prédiction', tstamp2))
    
    ## ON UTILISE LE MODELE DONNE EN PARAMETRE Type_Model
    lr = F_Model[Type_Model]['Model']
    Y_predict = lr.predict(Opt_X_data)

    tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de {} : {} - {} -> {}\n'.format(name, 'Prédiction', tstamp3, tstamp3-tstamp2))

    return Y_predict


###########################
####    Prediction MODELE : GENERAL
###########################


### prediction_modele_feature_delay
#   INPUT: 
#       - X_test
#       - dict(l_num, l_categ)
#       - F_Model_Optimisation
#       - F_Model 
#       - Type_Model : qui est le type de Regression
#
#   OUTPUT:
#       - Vecteur prédit X_0_x
def prediction_modele_general(input_X_test, d_features, F_Model_Optimisation, F_Model, Type_Model):
    #name = 'prediction_modele_general'
    return prediction_modele_feature_delay(input_X_test, d_features, F_Model_Optimisation, F_Model, Type_Model)


    
    

###########################
####    Remplacement de X_Test avec les données prédites
###########################

### modification_X_test:
#   INPUT:
#       A DEFINIR
#       - X_test
#       - liste : des vecteurs prédits
#       - dict (l_num, l_categ)
#       - Type_Model: modele choisi

#   CONDITION : 
#       - si dans le dictionnaire des features on utilise des donnees 
#       intermediaire alors on les remplace dans X_test
#   
#   OUTPUT :
#       - X_test_bis

def modification_X_test(
        input_X_train, input_X_test, 
        l_feature_a_modifier, d_features, 
        Type_Model,
        isRidge=False, isLasso=False
        ):
    name = 'modification_X_test'
    
    data_modifiees = input_X_test.copy()
    d_mesure = {}
     
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    # Debut
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'MODIFICATION DE X_test', tstamp1))
 
    for feature in l_feature_a_modifier:
        tstamp3 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.f'), '%Y-%m-%d %H:%M:%S.f')
        log_info('\t\tDEBUT : Transformation de {}'.format(feature))
        
        ## On entraine le modele
        o_F_Model_Optimisation, o_F_Model = \
            fabrication_modele_feature_delay(input_X_train, 
                                             d_features, feature,
                                             isRidge, isLasso, input_X_test)
        
        if feature in input_X_train.columns:
            d_mesure[feature] = {}
            ## On Predit les donnees et on remplace si la feature est bien dans les features d'entrainement
            m_data_feature_modifiee = prediction_modele_feature_delay(
                    input_X_test,  
                    d_features, 
                    o_F_Model_Optimisation, 
                    o_F_Model,
                    Type_Model
                    )
            data_modifiees[feature] = m_data_feature_modifiee
            
            
            ### TEST DE MESURE
            liste_model = ['Naive', 'LinearRegression', 'RidgeCV', 'LassoCV']
            for mod in liste_model:
                if mod in o_F_Model.keys():
                    log_info('JE SUIS DANS LE BOUCLE POUR MOD = {}'.format(mod))
                    
                    # Lineaire
                    tmp = prediction_modele_feature_delay(
                            input_X_test,  
                            d_features, 
                            o_F_Model_Optimisation, 
                            o_F_Model,
                            mod
                            )
                    d_mesure[feature][mod] = mesure_model(tmp, input_X_test[feature])
                    
            
        tstamp4 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S.f'), '%Y-%m-%d %H:%M:%S.f')
        log_info('\t\tFIN : Transformation de {}  [{}]'.format(feature, tstamp4-tstamp3))

    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    log_info('-- FIN de {} : {} - {} -> {}\n'.format(name, 'MODIFICATION DE X_test', tstamp2, tstamp2-tstamp1))
    
    return data_modifiees, d_mesure





###########################
####    MESURE DE LE PREDICTION
###########################


### mesure_model_feature_intermediaire
#   INPUT:
#       - Y_test
#       - Y_pred
#
#   OUTPUT:
#       - dict(RMSE, MAE, diff_erreur)
def mesure_model(Y_test, Y_pred):
    def rmse(np1, np2):
        return np.sqrt(np.mean((np1-np2) ** 2))   

    res = {}
    res['RMSE'] = rmse(Y_test, Y_pred)
    res['Diff_Erreur'] = Y_test - Y_pred

    return res




###########################
####    PROGRAMME PRINCIPAL : On deporte les variables à l'extérieur
###########################
 

def Execution_general_intermediaire(m_data = None, 
                       isRidge_general=False, isLasso_general=False, 
                       isRidge_interne=False, isLasso_interne=False,
                       Type_Model_general='LinearRegression', 
                       Type_Model_interne='LinearRegression',
                       d_features_general = {}, 
                       d_features_interne = {},
                       l_feature_a_modifier = []
                       ):
    name = 'Execution_general1'
      
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    # Debut de Fabrication du General
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'Fabrication du Model General', tstamp1))
    
    
    # On fabrique le modele general
    N_Data, N_Model_Optimisation, N_Model = fabrication_model_general(
            m_data, 
            d_features_general,
            isRidge_general, 
            isLasso_general)
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    log_info('-- FIN de {} : {} -- {}  --> {}\n'.format(name, 
             'Fabrication du Model General', tstamp2, tstamp2-tstamp1))
    
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    # Debut de Modification
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'Modification des Features', tstamp1))
    
    #modification_X_test(input_X_train, input_X_test, l_feature_a_modifier, d_features, Type_Model)
    
    new_X_test, d_mesure = modification_X_test(N_Data['X_train'], N_Data['X_test'], 
                                     l_feature_a_modifier, d_features_interne, 
                                     Type_Model_interne, isRidge_interne, 
                                     isLasso_interne)
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    log_info('-- FIN de {} : {} - {} --> {}\n'.format(name, 
             'Modification des Features', tstamp2, tstamp2-tstamp1))
    
    tstamp1 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    # Debut de Prédiction
    log_info('-- DEBUT de {} : {} - {}'.format(name, 'Prediction', tstamp1))
    
    
    ##### MESURES DES DIFF MODELES
    ### MODIF POUR IDENTIFIER LES MESURES
    #Y_pred = prediction_modele_general(
    #        new_X_test, d_features_general, 
    #        N_Model_Optimisation, N_Model, 
    #        Type_Model_general)

    #mesure = mesure_model(N_Data['Y_test'], Y_pred)
    
    d_mesure['GENERAL'] = {}
    
    liste_model = ['Naive', 'LinearRegression', 'RidgeCV', 'LassoCV']
    for mod in liste_model:
        if mod in N_Model.keys():
            log_info('GENERAL ---- JE SUIS DANS LE BOUCLE POUR MOD = {}'.format(mod))
            tmp = prediction_modele_general(
                    new_X_test, d_features_general, 
                    N_Model_Optimisation, N_Model, 
                    mod)
            d_mesure['GENERAL'][mod] = mesure_model(tmp, N_Data['Y_test'])
            

    #mesure = mesure_model(N_Data['Y_test'], Y_pred)
    
    tstamp2 = datetime.strptime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    
    # Fin de Prédiction
    log_info('-- FIN de {} : {} - {} --> {}\n'.format(name, 
             'Prediction', tstamp2, tstamp2-tstamp1))
    
    
    return d_mesure, N_Data['X_test'], new_X_test, N_Data, N_Model_Optimisation, N_Model
    return d_mesure, N_Data['X_test'], new_X_test



###########################
####    PROGRAMME PRINCIPAL : Lancement via des Paramètres
###########################
 
    
### Lancement de l'Execution via des paramétres
def Execution_general(m_data, 
                       isRidge_general=False, isLasso_general=False, 
                       isRidge_interne=False, isLasso_interne=False,
                       Type_Model_general='LinearRegression', 
                       Type_Model_interne='LinearRegression',
                       d_features_general = None, 
                       d_features_interne = None,
                       l_feature_a_modifier = None
                       ):
 
    if d_features_general==None:
        d_features_general = {
                'l_numerical': ['DISTANCE', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
                                'SECURITY_DELAY','LATE_AIRCRAFT_DELAY', 'DEP_DELAY' ],
                                
                'l_categoriel': ['MONTH', 'DAY_OF_MONTH', 
                              'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                              'ARR_HOUR', 'DEP_HOUR', 
                              'DAY_OF_WEEK', 'AIRLINE_ID', 'HDAYS', 'ARR_DEL15']
                }

    if d_features_interne==None:
        d_features_interne = {
                'l_numerical': ['DISTANCE'],
                                
                'l_categoriel': ['MONTH', 'DAY_OF_MONTH', 
                              'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                              'ARR_HOUR', 'DEP_HOUR', 
                              'DAY_OF_WEEK', 'AIRLINE_ID', 'HDAYS']
                }
    
    if l_feature_a_modifier==None:
        l_feature_a_modifier = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
                                'SECURITY_DELAY','LATE_AIRCRAFT_DELAY', 'DEP_DELAY']
    
    
    mesure, data_test_X, data_test_X_modifiee , N_Data, N_Model_Optimisation, N_Model= \
        Execution_general_intermediaire(m_data, 
                       isRidge_general, isLasso_general, 
                       isRidge_interne, isLasso_interne,
                       Type_Model_general, 
                       Type_Model_interne,
                       d_features_general, 
                       d_features_interne,
                       l_feature_a_modifier
                       )
    return mesure, data_test_X, data_test_X_modifiee, N_Data, N_Model_Optimisation, N_Model



###########################
####    PAR COMPAGNIE : Lancement via des Paramètres
###########################
 ### Lancement de l'Execution via des paramétres
def Execution_general_Par_Compagnie(
        m_data, 
        isRidge_general=False, 
        isLasso_general=False, 
        isRidge_interne=False, 
        isLasso_interne=False,
        Type_Model_general='LinearRegression', 
        Type_Model_interne='LinearRegression',
        d_features_general = None, 
        d_features_interne = None,
        l_feature_a_modifier = None
        ):
    '''TEST'''
    if d_features_general==None:
        d_features_general = {
                'l_numerical': ['DISTANCE', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
                                'SECURITY_DELAY','LATE_AIRCRAFT_DELAY', 'DEP_DELAY' ],
                                
                'l_categoriel': ['MONTH', 'DAY_OF_MONTH', 
                              'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                              'ARR_HOUR', 'DEP_HOUR', 
                              'DAY_OF_WEEK', 'AIRLINE_ID', 'HDAYS', 'ARR_DEL15']
                }

    if d_features_interne==None:
        d_features_interne = {
                'l_numerical': ['DISTANCE'],
                                
                'l_categoriel': ['MONTH', 'DAY_OF_MONTH', 
                              'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                              'ARR_HOUR', 'DEP_HOUR', 
                              'DAY_OF_WEEK', 'AIRLINE_ID', 'HDAYS']
                }
    
    if l_feature_a_modifier==None:
        l_feature_a_modifier = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
                                'SECURITY_DELAY','LATE_AIRCRAFT_DELAY', 'DEP_DELAY']

    l_mesure_total = []
    l_model = []
    l_model_optimisation = []
    
    for m_id in m_data['AIRLINE_ID'].unique().tolist():
        print("Traitement de l'ID", m_id)
        data = m_data[m_data['AIRLINE_ID']==m_id]
        
        mesure, data_test_X, data_test_X_modifiee , N_Data, N_Model_Optimisation, N_Model = Execution_general(
                m_data=data,  
                isRidge_general = isRidge_general, 
                isRidge_interne = isRidge_interne, 
                isLasso_general=isLasso_general, 
                isLasso_interne=isLasso_interne, 
                Type_Model_general=Type_Model_general, 
                Type_Model_interne=Type_Model_interne, 
                d_features_general=d_features_general, 
                d_features_interne=d_features_interne, 
                l_feature_a_modifier = l_feature_a_modifier
                )
        
        mesure['AIRLINE_ID'] = m_id
        N_Model['AIRLINE_ID'] = m_id
        N_Model_Optimisation['AIRLINE_ID'] = m_id
        
        l_mesure_total.append(mesure)
        l_model.append(N_Model)
        l_model_optimisation.append(N_Model_Optimisation)
               
    return l_mesure_total, data_test_X, data_test_X_modifiee, N_Data, l_model_optimisation, l_model
    return l_mesure_total, N_Model
#    return mesure, data_test_X, data_test_X_modifiee


###########################
####    AFFICHAGE DES RESULTATS
###########################
    
# =============================================================================
# def get_info(model):
#     if 'GENERAL' in model.keys():
#         ## Nous sommes directement sur un niveau d'infos
#         ### Sinon nous fonctionnons par AIRLINE_ID
#         
#         l_model_label = ['Naive', 'LinearRegression', 'RidgeCV', 'LassoCV'] 
#         df1 = pd.DataFrame(columns=l_model_label+['FEATURE'])
#     
#         for m in model.keys():
#             l_res = []
#             if m!='FEATURE':
#                 for i in l_model_label:
#                     try:
#                         l_res.append(model[m][i]['RMSE'])
#                     except:
#                         l_res.append(None)
#                 l_res.append(m)
#             df1 = df1.append({},ignore_index=True)
#             df1.iloc[df1.index.max()] = l_res
#     return df1.set_index(keys='FEATURE').T
# 
# =============================================================================


def get_info(model):
    if type(model)==dict:
     ## Nous sommes directement sur un niveau d'infos
        ### Sinon nous fonctionnons par AIRLINE_ID
        
        l_model_label = ['Naive', 'LinearRegression', 'RidgeCV', 'LassoCV'] 
        df1 = pd.DataFrame(columns=l_model_label+['FEATURE'])
    
        for m in model.keys():
            l_res = []
            if m!='FEATURE':
                for i in l_model_label:
                    try:
                        l_res.append(model[m][i]['RMSE'])
                    except:
                        l_res.append(None)
                l_res.append(m)
            df1 = df1.append({},ignore_index=True)
            df1.iloc[df1.index.max()] = l_res
        return df1.set_index(keys='FEATURE').T
    else:
        res = []
        l_id = []
        for x in model:
            id = x['AIRLINE_ID']
            tmp = get_info(x)[['GENERAL']]
            res.append(tmp)
            l_id.append(id)
            
        val = res[0]
        for i,x in zip(l_id, res):
            val.loc[:,i] = x['GENERAL'].values
        del val['GENERAL']
        return val


# =============================================================================
# 
# 
# 
# Stockage des Models
# - Une fois les models entraines, nous allons les stocker dans des fichiers
# - Nous allons aussi gérer le chargement de ces derniers
# 
# 
# 
# =============================================================================


 
# Sauvegarde d'un model dans un fichier via un nom créé à partir des paramètres
def save_model(path_directory, id_compagnie, type_modelisation, model):
    from sklearn.externals import joblib
    
    nom_fichier = '{}model_{}_{}.pkl'.format(
            path_directory,
            id_compagnie, 
            type_modelisation 
            )
    joblib.dump(model, nom_fichier) 
    print('sauvegarde du fichier {}'.format(nom_fichier))
    
    
# Chargement d'un model à partir d'un fichier :
##  dont le nom est créé à partir des paramètres
    
def load_model(path_directory, id_compagnie, type_modelisation):
    from sklearn.externals import joblib
    from os.path import isfile
    
    nom_fichier = '{}model_{}_{}.pkl'.format(
            path_directory,
            id_compagnie, 
            type_modelisation 
            )
    
    ## On ajoute un contrôle d'existence du fichier
    if (isfile(nom_fichier)):    
        ## On charge le fichier
        model = joblib.load(nom_fichier) 
        print('chargement du fichier {}'.format(nom_fichier))
    else:
        print("Attention le fichier {} n'existe pas".format(nom_fichier))
    
    ## On retourne le fichier
    return model


## Sauvegarde des modèles principaux pour toutes les compagnies
## A partir d'une liste de modèles        
def save_my_models(models, path_directory=''):
    l_modelisation = ['LinearRegression', 'RidgeCV', 'LassoCV']
    
    for m_model in models:
        AIRLINE_ID = m_model['AIRLINE_ID']
        for type_model in l_modelisation:
            if type_model in m_model.keys():
                save_model(path_directory = path_directory, 
                           id_compagnie = AIRLINE_ID,
                           type_modelisation = type_model,
                           model = m_model[type_model]['Model'],
                           )

## Sauvegarde des modèles principaux pour toutes les compagnies
## A partir d'une liste d'optimisation
def save_my_optimisation_models(models, path_directory=''):
    l_modelisation = ['OneHotEncoder', 'StandardScaler']
    
    for m_model in models:
        AIRLINE_ID = m_model['AIRLINE_ID']
        for type_model in l_modelisation:
            if type_model in m_model.keys():
                save_model(path_directory = path_directory,
                           id_compagnie = AIRLINE_ID,
                           type_modelisation = type_model,
                           model = m_model[type_model],
                           )

## Chargement des modèles principaux pour toutes les compagnies
## A partir d'une liste de modèles 
def load_my_models(l_AIRLINE_ID, path_directory=''):
    l_modelisation = ['LinearRegression', 'RidgeCV', 'LassoCV']
    l_model = []
    
    for id in l_AIRLINE_ID:
        d_model = {'AIRLINE_ID': id}
        for m_type_modelisation in l_modelisation:
            d_model[m_type_modelisation] = {}
            
            loaded_model = load_model(
                    path_directory, 
                    id_compagnie = id, 
                    type_modelisation = m_type_modelisation
                    )
            
            d_model[m_type_modelisation]['Model'] = loaded_model
        
        ## On ajoute ce modele à la liste
        l_model.append(d_model)

    # On renvoie la liste créée
    return l_model
    
## Chargement des modèles d'optimisation pour toutes les compagnies
## A partir d'une liste d'optimisation
def load_my_optimisation_models(l_AIRLINE_ID, path_directory=''):
    l_modelisation = ['OneHotEncoder', 'StandardScaler']
    l_model = []
    
    for id in l_AIRLINE_ID:
        d_model = {'AIRLINE_ID': id}
        for m_type_modelisation in l_modelisation:
            
            loaded_model = load_model(
                    path_directory, 
                    id_compagnie = id, 
                    type_modelisation = m_type_modelisation
                    )
            d_model[m_type_modelisation] = loaded_model
        
        ## On ajoute ce modele à la liste
        l_model.append(d_model)
        
    # On renvoie la liste créée
    return l_model





