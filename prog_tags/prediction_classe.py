#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:58:11 2018

@author: seb
"""

import pandas as pd
from os import path
from datetime import datetime

import numpy as np
import re

import os
import sys
import json

from sklearn.externals import joblib
from os.path import isfile


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
        tstamp = datetime.strptime(datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')
        self._liste_msg.append('{}{} : {}'.format(param, tstamp, msg))

    msg = property(_get_msg, _set_msg)

    def __str__(self):
        s_msg = ''
        for i in self._liste_msg:
            s_msg = s_msg + i + '\n'
        return s_msg

    def affiche(self, nb_lignes=200):
        s_msg = ''
        for i in self._liste_msg[-nb_lignes:]:
            s_msg = s_msg + i + '\n'
        return s_msg

    def add_log(self, msg, param=''):
        self._set_msg(msg, param)

    def __call__(self, msg, param=''):
        self._set_msg(msg, param)

    def remove_msg(self, nb_lignes_a_conserver=None):
        if nb_lignes_a_conserver is None:
            self._liste_msg = []
        else:
            self._liste_msg = self._liste_msg[-nb_lignes_a_conserver:]


class Prediction():
    """Classe de Prediction : Stockage des models"""
    _file_vocabulary_loaded = False
    _file_models_loaded = False

    # Vocabulaire et mots utiles
    _voc_body = None
    _voc_title = None
    _voc_tags = None
    _stopwords = None

    # Models
    _model_body_vectCount = None
    _model_title_vectCount = None
    _model_body_tfidf = None
    _model_title_tfidf = None
    _l_model_body_classifier = None
    _l_model_title_classifier = None

    # Mes Logs
    # _logs = None
    _logs_param = 'Prediction : '

    @classmethod
    def __init__(self, nom_fichier=None):
        self._logs = Logs()
        self._logs('__init__ : DEBUT', param=self._logs_param)

        if nom_fichier is not None:
            self._nom_fichier = nom_fichier
        
        import os
        print("ATTENTION nous sommes ici ", os.getcwd())
        # Chargement des vocabulaires
        if not(self._file_vocabulary_loaded):
            msg = "__init__ : DEBUT CHARGEMENT DES VOCABULAIRES"
            print(msg)
            self._logs(msg, param=self._logs_param)

            try:
                self.load_vocabulary()
                self._file_vocabulary_loaded = True
            except Exception as e:
                msg = '__init__ : ERREUR CHARGEMENT DES VOCAB {}'.format(
                        e)
                print(msg)
                self._logs(msg, param=self._logs_param)
            finally:
                pass
        else:
            msg = '-- VOCABULAIRES : FICHIERS DEJA CHARGE'
            print(msg)
            self._logs(msg, param=self._logs_param)

        # Chargement des models
        if not(self._file_models_loaded):
            msg = "__init__ : DEBUT CHARGEMENT DES MODELS"
            print(msg)
            self._logs(msg, param=self._logs_param)

            try:
                self.load_models(nb_features=len(self._voc_tags))
                self._file_models_loaded = True
            except Exception as e:
                msg = '__init__ : ERREUR CHARGEMENT DES MODELS \n\t{}'.format(
                        e)
                print(msg)
                self._logs(msg, param=self._logs_param)
            finally:
                pass
        else:
            msg = '-- MODELS : FICHIER DEJA CHARGE'
            print(msg)
            self._logs(msg, param=self._logs_param)

    @classmethod
    def load_vocabulary(self):
        try:
            self._voc_body = self.load_dict(
                    self, m_path='prog_tags/data/MODEL/vocabulary_body.voc')
            self._voc_title = self.load_dict(
                    self, m_path='prog_tags/data/MODEL/vocabulary_title.voc')
            self._voc_tags = self.load_dict(
                    self, m_path='prog_tags/data/MODEL/vocabulary_tags.voc')
        except Exception as e:
            msg = '-- load_vocabulary : ERREUR {}'.format(e)
            print(msg)
            self._logs(msg, param=self._logs_param)

        self._stopwords = self.load_stop_word(
                self, m_path='prog_tags/data/MODEL/stopword.lst')

    @classmethod
    def load_models(self, nb_features):
        '''Fonction de chargement des models'''

        # Les fichiers individuels
        # ###pattern_files = 'prog_tags/data/MODEL/PROD/{FILE}'
        pattern_files = 'prog_tags/data/MODEL/{FILE}'

        try:
            self._model_body_vectCount = self.load_one_model(
                    self,
                    nom_fichier=pattern_files.format(
                            FILE='body_vecCount.mod'))
            self._model_title_vectCount = self.load_one_model(
                    self,
                    nom_fichier=pattern_files.format(
                            FILE='title_vecCount.mod'))
            self._model_body_tfidf = self.load_one_model(
                    self,
                    nom_fichier=pattern_files.format(
                            FILE='body_tfidf.mod'))
            self._model_title_tfidf = self.load_one_model(
                    self,
                    nom_fichier=pattern_files.format(
                            FILE='title_tfidf.mod'))
            self._l_model_body_classifier = self.load_one_model(
                    self,
                    nom_fichier=pattern_files.format(
                            FILE='body_classifier.mod'))
            self._l_model_title_classifier = self.load_one_model(
                    self,
                    nom_fichier=pattern_files.format(
                            FILE='title_classifier.mod'))

        except Exception as e:
            msg = '-- load_models -- : ERREUR Fichier individuels {}'.format(e)
            self._logs(msg)
            raise Exception(msg)

#        # Les listes de models
#        try:
#            self._l_model_body_classifier = self.load_list_model(
#                    self, subPath='BODY', nb_models=nb_features)
#
#        except Exception as e:
#            msg = '-- load_models -- : ERREUR liste BODY {} '.format(e)
#            self._logs(msg)
#            raise Exception(msg)
#
#        try:
#            self._l_model_title_classifier = self.load_list_model(
#                    self, subPath='TITLE', nb_models=nb_features)
#
#        except Exception as e:
#            msg = '-- load_models -- : ERREUR liste TITLE {}'.format(e)
#            self._logs(msg)
#            raise Exception(msg)

    def load_one_model(self, nom_fichier):
        '''Charge un fichier dont le nom est en paramètre'''
        #    from sklearn.externals import joblib
        if isfile(nom_fichier):
            # On copie le fichier
            try:
                m_mod = joblib.load(nom_fichier)
                return m_mod
            except Exception as e:
                msg = '\n\t-- load_one_model -- : [ Fichier : {} ] -- ERREUR {} '.format(nom_fichier, e)
                self._logs(msg)
                raise Exception(msg)
                return None
        else:
            msg = '\n\t-- load_one_model -- : ERREUR Fichier {} absent'.format(
                    nom_fichier)
            self._logs(msg)
            raise Exception(msg)

    def load_list_model(self, subPath='BODY', nb_models=0):
        '''Charge la liste des models d'un sous repertoire de data
        INPUT:
        -----
            - subPath : sous repertoire BODY/TITLE
            - nb_models : nb de models à charger
        '''

        # pattern_name = 'prog_tags/data/MODEL/{SUB_PATH}/PROD/model_{no}.mod'
        pattern_dir = 'prog_tags/data/MODEL/{SUB_PATH}/PROD/'
        dir_path = pattern_dir.format(SUB_PATH=subPath)

        if not(os.path.isdir(dir_path)):
            self._logs('load_list_model : DIR prog_tags/data/MODEL absent')

        if not(os.path.isdir(dir_path)):
            raise Exception('-- load_list_model -- : Repertoire {} absent'
                            .format(dir_path))

        file_pattern = dir_path + 'model_{no}.mod'
        l_model = []
        for idx in range(nb_models):
            l_model.append(self.load_one_model(
                    self, nom_fichier=file_pattern.format(no=idx)))

        return l_model

    # Chargement des fichiers individuels
    def load_stop_word(self, m_path=None):
        '''Chargement des stopWord du fichier stopword.lst
        INPUT:
        ------
            - path (optionnel) : path different de ./stopword.lst
        OUTPUT:
        ------
            - sw: liste des stopword
        '''

        if m_path is None:
            m_path = 'stopword.lst'

        try:
            with open(m_path, 'r') as f:
                sw_out = f.readlines()
        except Exception as e:
            # #####print('ERREUR sur {} -> {}'.format(m_path, e))
            # TODO : ajouter diff type d'ERREURS : 
            #  - Fichier Inexistant
            #  - Pb de Droits
            raise Exception('\n\t-- load_stop_word -- : '
                            '\n\tERREUR CHARGEMENT FICHIER {}'
                            '\n\t --- {} ---'.format(m_path, e))
            return None

        return [x.replace('\n', '') for x in sw_out]

    # Chargement de DICTIONNAIRES
    def load_dict(self, m_path):
        '''Chargement d'un dictionnaire d'un fichier'''
        try:
            with open(m_path) as data_file:
                data_loaded = json.load(data_file)
        except Exception as e:
            # #####print('ERREUR sur {} -> {}'.format(m_path, e))
            # TODO : ajouter diff type d'ERREURS : 
            #  - Fichier Inexistant
            #  - Pb de Droits
            raise Exception(
                    '-- load_dict -- : ERREUR CHARGEMENT FICHIER {}'.format(
                            m_path))
            return None
        return data_loaded

    # Fonction de prediction du Body
    def predict_body_from_text(self, texte):
        df_pred = self.predict_body_df_from_text(texte)

        return list(filter(
                lambda x:
                    x if df_pred.loc['my_prediction', x] == 1
                    else 0, df_pred.columns.tolist()))

    # Fonction de prediction du Title
    def predict_title_from_text(self, texte):
        df_pred = self.predict_title_df_from_text(texte)

        return list(filter(
                lambda x:
                    x if df_pred.loc['my_prediction', x] == 1
                    else 0, df_pred.columns.tolist()))

    # Fonction de prediction DF du Body
    def predict_body_df_from_text(self, texte):
        '''Fonction de prediction du body
        INPUT:
        ------
            - texte: un string sous forme de html
        OUTPUT:
            - DataFrame de resultat des tags
        '''
    # 1 - On transforme text1 via le VectCount
        X_test_v0 = self._model_body_vectCount.transform([texte])

        # 2 - On le tranforme via le tfidf
        X_test = self._model_body_tfidf.transform(X_test_v0)

        # 3 - On prédit chaque valeur
        # 3.1 On créé un Y_pred de sortie
        l_tags_features = sorted(list(self._voc_tags))

        Y_pred = pd.DataFrame(
            columns=l_tags_features,
            index=['my_prediction']
        )

        for m_model, m_feature in zip(self._l_model_body_classifier,
                                      l_tags_features):
            Y_pred[m_feature] = m_model.predict(X_test)

        # 4 - On renvoie la liste des tags
        return Y_pred

    # Fonction de prediction DF du Title
    def predict_title_df_from_text(self, texte):
        '''Fonction de prediction du body
        INPUT:
        ------
            - texte: un string sous forme de html
        OUTPUT:
            - DataFrame de resultat des tags
        '''
    # 1 - On transforme text1 via le VectCount
        X_test_v0 = self._model_title_vectCount.transform([texte])

        # 2 - On le tranforme via le tfidf
        X_test = self._model_title_tfidf.transform(X_test_v0)

        # 3 - On prédit chaque valeur
        # 3.1 On créé un Y_pred de sortie
        l_tags_features = sorted(list(self._voc_tags))

        Y_pred = pd.DataFrame(
            columns=l_tags_features,
            index=['my_prediction']
        )

        for m_model, m_feature in zip(self._l_model_title_classifier,
                                      l_tags_features):
            Y_pred[m_feature] = m_model.predict(X_test)

        # 4 - On renvoie la liste des tags
        return Y_pred

    # Foncion de prodiction Generale
    def predict(self, texte_body, texte_title):
        '''Fonction de prediction generale
        INPUT:
        ------
            - texte_body: un string sous forme de html du Body
            - texte_title: un string sous forme de html du Title
        OUTPUT:
        -------
            - DataFrame de resultat des tags
        '''
        pred_body = self.predict_body_df_from_text(texte_body)
        pred_title = self.predict_title_df_from_text(texte_title)

        pred = np.logical_or(pred_body, pred_title)
        return list(filter(
                lambda x:
                    x if pred.loc['my_prediction', x]
                    else 0, pred.columns.tolist()))

        return pred


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    