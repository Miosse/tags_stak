#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 00:28:58 2018

@author: seb
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

import re
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()

# =============================================================================
# TOKENIZER
# =============================================================================
tokenizer = nltk.RegexpTokenizer(r'\w+')

# =============================================================================
# FONCTIONS DE NETTOYAGE
# =============================================================================

# =============================================================================
# SUPPRESSION DE PONCTUATIONS
# =============================================================================


def suppression_ponctuations_simple(text_in):
    '''
    Fonction qui supprime toute ponctuation située avant, après un espace,
        ou entourée d'espace:
    PRIS EN COMPTE:
        . , ; ) ( : ? ! & @ / + = - _ < >
    EXEMPLE pour ',':
        - 'mot1 , mot2' : devient 'mot1 mot2'
        - 'mot1, mot2' : devient 'mot1 mot2'
        - 'mot1 ,mot2' : devient 'mot1 mot2'
        Mais
        - 'mot1,mot2' : reste 'mot1,mot2'
    '''

    # Suppresion des ponctuation en fin de ligne
    t1 = re.sub(
            r'([\.\,\;\)\(\:\?\!\&\@\/\+\=\-\_\<\>]$)', '', text_in)
    # Suppresion en début de ligne
    t1 = re.sub(
            r'(^[\.\,\;\)\(\:\?\!\&\@\/\+\=\-\_\<\>])', '', t1)
    # Suppression si on a un espace après
    t1 = re.sub(
            r'([\.\,\;\)\(\:\?\!\&\@\/\+\=\-\_\<\>]\s)', ' ', t1)
    # Suppression si on a un espace avant
    t1 = re.sub(
            r'(\s[\.\,\;\)\(\:\?\!\&\@\/\+\=\-\_\<\>])', ' ', t1)
    # Suppression si on a un espace autour
    t1 = re.sub(
            r'\s[\.\,\;\)\(\:\?\!\&\@\/\+\=\-\_\<\>]\s', ' ', t1)

    # On renvoie le résultat
    return t1


def suppression_mot_une_lettre(text_in):
    return re.sub(r'(\s\w\s)', ' ', text_in)


def suppression_valeur_numerique(text_in):
    return re.sub(r'(\s\d*\s)', ' ', text_in)


def suppression_espaces_multiples(text_in):
    return re.sub(r'(\s+)', ' ', text_in)


def fonction_nettoyage(html_in):
    '''Nettoyage des ponctuations, et suppression du bloc 'Code'
    On renvoie la version nettoyées des balise html en minuscule'''
    html_in_soup = BeautifulSoup(html_in, 'html.parser')

    # Suppression des blocs Code (en ligne)
    for h_code in html_in_soup.find_all('code'):
        h_code.decompose()

    # Il faut supprimer les balises html
    t1 = suppression_ponctuations_simple(html_in_soup.get_text()).lower()
    # On supprime les mot d'une seule lettre et les valeurs numériques
    t1 = suppression_mot_une_lettre(suppression_valeur_numerique(t1))

    # On supprime les espaces multiples
    return suppression_espaces_multiples(t1)
    # return suppression_ponctuations_simple(html_in_soup.get_text()).lower


# =============================================================================
# FONCTIONS SUR LES TAGS
# =============================================================================
# Fabrication d'un Counter des mots
def most_common_tags(tags):
    words = []
    for tag in tags:
        words += re.findall(r'<(\w+|[0-9a-zA-Z-\-\.\+\#]+)>', tag)
    counter = Counter(words)
    return counter


# Ajout de features necessaires : Body_clean / Tags2
def ajoute_features(data):
    def transform_tags(value):
        return ' '.join(re.findall(r'<(\w+|[0-9a-zA-Z-\-\.\+\#]+)>', value))

    # Ajout de la feature "Body_clean"
    data['Body_clean'] = data['Body'].apply(
            lambda x: fonction_nettoyage(x))

    # Ajout de la feature 'Tags': retourne les Tags avec des espaces entre eux
    data['Tags2'] = data['Tags'].apply(lambda x: transform_tags(x))


# Generation d'une matrice des Tags sous forme de dummy
def genere_target_dummy(data, min_df=20):
    '''Fonction qui genere un dataFrame des tags
    INPUT:
    ------
        - data : le dataset initial
        - min_df (=20) : nb minimum d'apparition du tag pour prise en compte
    OUTPUT:
    -------
        - data_target : sparse matrix des dummy features des tags
        - features    : liste des features en colonne
        - index       : index du DataSet initial
    '''
    if 'Tags2' not in data.columns:
        print('/!\ Ajout des features en cours')
        ajoute_features(data)

    m_count_vect = CountVectorizer(
        min_df=min_df,
        token_pattern=r'<(\w+|[0-9a-zA-Z-\-\.\+\#]+)>')
    Y_tags = m_count_vect.fit_transform(data.Tags)

    return pd.SparseDataFrame(
        data=Y_tags,
        columns=m_count_vect.get_feature_names(),
        index=data.index,
        default_fill_value=0
        )

    return {
        'data_target': Y_tags,
        'features': m_count_vect.get_feature_names(),
        'index': data.index
    }

# Generation d'une matrice des Tags sous forme de dummy et le vocabulaire


def genere_target_dummy_and_vocabulary(data, min_df=20):
    '''Fonction qui genere un dataFrame des tags
    INPUT:
    ------
        - data : le dataset initial
        - min_df (=20) : nb minimum d'apparition du tag pour prise en compte
    OUTPUT:
    -------
        - data_target : sparse matrix des dummy features des tags
        - features    : liste des features en colonne
        - index       : index du DataSet initial
    '''
    if 'Tags2' not in data.columns:
        print('/!\ Ajout des features en cours')
        ajoute_features(data)

    m_count_vect = CountVectorizer(
        min_df=min_df,
        token_pattern=r'<(\w+|[0-9a-zA-Z-\-\.\+\#]+)>')
    Y_tags = m_count_vect.fit_transform(data.Tags)

    return pd.SparseDataFrame(
        data=Y_tags,
        columns=m_count_vect.get_feature_names(),
        index=data.index,
        default_fill_value=0
        ), m_count_vect.vocabulary_

# Generation d'un Dataset des tags avec leur Fréquence de sortie


def genere_df_target_tags_freq(data, min_df=20):
    '''Fonction genere la liste des tags
    valeurs de tags en index et Frequence en valeur
    - INPUT:
    --------
        - data
        - min_df (20): nb minimum d'apparition du tag pour prise en compte
    '''
    if 'Tags2' not in data.columns:
        print('/!\ Ajout des features en cours')
        ajoute_features(data)

    m_tags = most_common_tags(data.Tags.values)

    df_m_tags = pd.DataFrame.from_dict(data=dict(m_tags), orient='index')
    df_m_tags.columns = ['Freq']
    df_m_tags.sort_values(by=['Freq'], ascending=False, inplace=True)

    # Filtrage des données et renvoie
    return df_m_tags[df_m_tags.Freq >= min_df]


# Extrait la colonne col_name du Dummy Tags
def get_df_tags_from_feature(dict_tags, col_name):
    '''Retourne la colonne à partir d'un nom de colonne
    APPEL:
    ------
        - get_df_tags_from_feature(dict, 'python')
    '''
    if 'features' not in dict_tags.keys():
        print("Attention dict_tags doit contenir 'features'")
        return None

    l_features = dict_tags['features']

    if (col_name not in l_features):
        print("ERROR : Colonne '{}' non reconnue".format(col_name))
        return None

    return pd.DataFrame(
        dict_tags['data_target'].getcol(l_features.index(col_name)).toarray(),
        columns=[col_name],
        index=dict_tags['index']
    )

# ### A CONSERVER
# ### cleantext = BeautifulSoup(raw_html, "lxml").text


# =============================================================================
# ETAPES DE PROGRAMME
# =============================================================================

# FONCTION ETAPE1 : COMPTAGE DES MOTS PAR POST
def etape1_ajout_counterWord1(data):
    '''Fonction qui fait un comptage par post
    INPUT:
    ------
        - data
    OUTPUT:
    -------
        - data + colonne de counter (solution choisie)
        OU
        - liste des counter (en stand by : a voir pb de perfs)
    '''
    # tokenizer = nltk.RegexpTokenizer(r'\w+')
    # tokenizer = nltk.RegexpTokenizer(r'<(\w+|[0-9a-zA-Z-\-\.\+\#]+)>')
    tokenizer = nltk.RegexpTokenizer(r'\w+|[0-9a-zA-Z-\-\.\+\#]+')

    data_text = data['Body_clean']

    # On ajoute la nouvelle feature
    data.at[:, 'Counter_WORD1'] = None

    for m_id, m_text in data_text.iteritems():
        # print('--{}-- || --{}--'.format(m_id, m_text))
        # # On met des Token dans la nouvelle colonne
        data.at[m_id, 'Counter_WORD1'] = tokenizer.tokenize(m_text.lower())


def etape2_get_freq_word_and_stop_word(data, nb_mots_max=100):
    '''Fonction qui ressort un counter de mots les plus fréquent
    INPUT:
    ------
        - data
    OUTPUT:
    -------
        ###- counter : reprend les mots les plus sités
        - freq_words
        - stopword (english + spécifiques)
    PREREQUIS:
    ----------
        - il faut que data ait la colonne 'Counter_WORD1' créée dans l'étape1
    APPEL FONCTION:
    ---------------
        freq_words, sw = etape2(df2, nb_mots_max=10)
    '''
    freq_totale = nltk.Counter()

    import datetime
    debut = datetime.datetime.now()
    print('Etape2 : DEB {}'.format(debut))

    data_text = data['Counter_WORD1']

    for m_id, token in data_text.iteritems():
        freq_totale += nltk.FreqDist(token)

    most_freq = list(zip(*freq_totale.most_common(nb_mots_max)))[0]
    # #####return most_freq

    # On créé notre set de stopwords final qui cumule ainsi les 100 mots
    # les plus fréquents du corpus ainsi que l'ensemble de stopwords par défaut
    # présent dans la librairie NLTK
    # ##sw.update(tuple(nltk.corpus.stopwords.words('french')))

    # ## Maintenant on peut fabriquer nos STOPWORD sw
    from nltk.corpus import stopwords
    nltk.data.path.append(
            r"/Users/seb/Workspace/Dev/Formation-OC/LIBRAIRIES/nltk_data")

    sw = set()
    # ######sw.update(tuple(nltk.corpus.stopwords.words('english')))
    # sw1.update(stopwords.words('english'))
    sw.update(stopwords.words('english'))
    sw.update(most_freq)
    fin = datetime.datetime.now()
    print('Etape2 : FIN : {}'.format(fin-debut))

    return freq_totale, sw
    return most_freq, sw


def etape3_get_counterWord2_with_sw(data, stopword):
    '''Fonction qui fait un comptage par post avec prise en compte des stopwords
    INPUT:
    ------
        - data
    OUTPUT:
    -------
        - data + colonne de counter (solution choisie)
        OU
        - liste des counter (en stand by : a voir pb de perfs)
    APPEL FONCTION:
    ---------------
        - etape3(df2, sw)
    '''
    data_text = data['Body_clean']

    # On initialise la nouvelle feature
    data['Counter_WORD2'] = None

    for m_id, m_text in data_text.iteritems():
        # # On liste les tokens
        tokens = tokenizer.tokenize(m_text.lower())

        # On sélectionne les token qui ne sont pas des stopWords
        #  et on met à la jour la nouvelle colonne
        data.at[m_id, 'Counter_WORD2'] = [w for w in tokens
               if w not in list(stopword)]


def etape3_bis_get_counterWord2_with_sw_and_stemmer(data, stopword):
    '''Fonction qui fait un comptage par post avec prise en compte des stopwords
    /!\ On ajoute le stemmer
    INPUT:
    ------
        - data
    OUTPUT:
    -------
        - data + colonne de counter (solution choisie)
        OU
        - liste des counter (en stand by : a voir pb de perfs)
    APPEL FONCTION:
    ---------------
        - etape3(df2, sw)
    '''
    from nltk.stem.snowball import EnglishStemmer
    stemmer = EnglishStemmer()

    data_text = data['Body_clean']

    # On initialise la nouvelle feature
    data['Counter_WORD2'] = None

    for m_id, m_text in data_text.iteritems():
        # On liste les tokens
        tokens = tokenizer.tokenize(m_text.lower())

        # On sélectionne les token qui ne sont pas des stopWords
        # et on met à la jour la nouvelle colonne
        # data.at[m_id, 'Counter_WORD2'] =
        #      [w for w in tokens if not w in list(stopword)]
        data.at[m_id, 'Counter_WORD2'] = [stemmer.stem(w)
            for w in tokens if w not in list(stopword)]

# =============================================================================
# SAVE & LOAD
# =============================================================================

# STOP-WORD


def save_stop_word(sw, path=None):
    '''Sauvegarde des stopWord dans le fichier stopword.lst
    INPUT:
    ------
        - sw : liste des stopword
        - path (optionnel) : path different de ./stopword.lst
    '''

    if path is None:
        path = 'stopword.lst'

    with open(path, 'w') as f:
        f.write("\n".join(sw))


def load_stop_word(path=None):
    '''Chargement des stopWord du fichier stopword.lst
    INPUT:
    ------
        - path (optionnel) : path different de ./stopword.lst
    OUTPUT:
    ------
        - sw: liste des stopword
    '''

    if path is None:
        path = 'stopword.lst'

    with open(path, 'r') as f:
        sw_out = f.readlines()

    return [x.replace('\n', '') for x in sw_out]

# DICTIONNAIRES


def save_dict(m_dict, path):
    '''Sérialisation d'un dictionnaire dans un fichier'''
    import json
    with open(path, 'w') as outfile:
        json.dump(m_dict, outfile)


def load_dict(path):
    '''Chargement d'un dictionnaire d'un fichier'''
    import json
    try:
        with open(path) as data_file:
            data_loaded = json.load(data_file)
    except Exception:
        print('ERREUR sur {} -> {}'.format(path, Exception))
        # TODO : ajouter diff type d'ERREURS : 
        #  - Fichier Inexistant
        #  - Pb de Droits
        return None
    return data_loaded


# =============================================================================
# LAST VERSION MODELISATION
# =============================================================================

# Recuperation du vocabulaire

def get_vocabulary_with_nGrams(data, col, sw=[], min_df=1, isCodeRemoved=True):
    '''
    Cette fonction va renvoyer le vocabulaire du DataSet.
    Nous mettons en paramètre les STopWords identifiés
    Nous faisons une recherche sur les N-Grams de 1 à 6
    INPUT:
    ------
        - data : dataSet à analyser
        - col : le nom de la colonne où se trouve le texte
        - sw : liste des stopwords
        - min_df=1 : nb minimum d'occurence des mots pour être pris en compte
    OUTPUT:
    -------
        - vocabulaire identifié
    INFORMATION:
    ------------
        - On utilise le stemming pour lemmatization
        -> Nous pourrons remplacer cela par la lemmatization de 'Porter'
    '''
    # Nous chargeons la fonction de comptage des features
    from sklearn.feature_extraction.text import CountVectorizer

    # Partie de lemmatization
    # On charge la librairie de Stemmer en Anglais
    from nltk.stem.snowball import EnglishStemmer
    stemmer = EnglishStemmer()

    # Réalisation du stemming (on coupe les racines)
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    # La tokenisation :
    def tokenize(text):
        m_token_pattern = r"((?:(?:(?:[0-9a-zA-Z])\.){2,}[a-zA-Z])" +\
            "|(?:(?:[0-9a-zA-Z]){2,}\.(?:[0-9a-zA-Z]){2,}" +\
            "|(?:\.(?:[0-9a-zA-Z]){2,}))" +\
            "|[0-9a-zA-Z-\-\+\#]{2,}|w+)"

        from nltk.tokenize import RegexpTokenizer
        # Nous allons utiliser le pattern pour identifier les mots
        tokenizer = RegexpTokenizer(m_token_pattern)

        # Nous lançons la séparation des mots
        tokens = tokenizer.tokenize(text)

        # On fait appel au stemming pour rapprocher les mots de même racine
        stems = stem_tokens(tokens, stemmer)

        # Etape de nettoyage des valeurs :
        # Nous allons supprimer les nombres sans texte,
        def suppress_nb(x):
            import re
            if x is None:
                return None
            pattern = r'(^[\#\-\+]*[0-9]*$|' +\
                    '^[0-9]*[\#\-\+]*$|' +\
                    '^[0-9]*[\#\-\+]?[0-9]*$|' +\
                    '^[0-9\#\-\+][a-z]$|' +\
                    '^[a-z][0-9\#\-\+]$|' +\
                    '^[0-9]*\.[0-9]*$)'
            if not(re.match(pattern, x)):
                return x

        def nettoie_points(x):
            import re
            if x is None:
                return None

            if (re.match(r'(^[\.\-\#][a-z]*$)', x)):
                return ''.join(list(x)[1:])
            else:
                return x

        # Nous filtrons les nombres seuls
        stems = list(filter(lambda x: suppress_nb(x), stems))
        stems = [nettoie_points(x) for x in stems]
        return stems

    # Le préprocessing :
    #   - nettoie le html
    #   - supprime les blocs de Code
    #   - ne prend que la partie textuelle du html
    #   - renvoie ce texte en minuscule

    def preProcess_remove_Code(html_in):
        # Chargement du module BeautifulSoup pour le parsing des données HTML
        from bs4 import BeautifulSoup
        html_in_soup = BeautifulSoup(html_in, 'html.parser')

        # Suppression des blocs de Code
        for h_code in html_in_soup.find_all('code'):
            h_code.decompose()

        return html_in_soup.get_text().lower()

    def preProcess_keep_Code(html_in):
        # Chargement du module BeautifulSoup pour le parsing des données HTML
        from bs4 import BeautifulSoup
        html_in_soup = BeautifulSoup(html_in, 'html.parser')

        return html_in_soup.get_text().lower()

    if isCodeRemoved:
        preProcess = preProcess_remove_Code
    else:
        preProcess = preProcess_keep_Code

    # Fabrication du Bag Of Words (BOW) via CountVectorizer

    vectorizer = CountVectorizer(
        analyzer="word",
        # On fait varier la fréquence minimum pour la prise en compte des mots
        min_df=min_df,
        tokenizer=tokenize,
        # token_pattern=m_token_pattern,
        preprocessor=preProcess,
        stop_words=sw,        # Pas de stopWord car nous les cherchons
        ngram_range=(1, 6),    # Nous ne prenons que le 1-Gram
        # max_features = 1000  # Nous cherchons les nombres maximum
    )

    import datetime
    debut = datetime.datetime.now()

    # On entraine les données pour fabriquer le Bow
    vectorizer.fit(data[col])

    fin = datetime.datetime.now()
    print("[ Vectorizer : {}]".format(fin-debut))

    return vectorizer.vocabulary_


def create_matrix_tfidf(data, col, vocabulary, sw=[], min_df=1):
    '''
    Cette fonction va renvoyer la frequence des mots issues du DataSet.
    Nous mettons en paramètre les STopWords identifiés
    Nous faisons une recherche sur les N-Grams de 1 à 6
    INPUT:
    ------
        - data : dataSet à analyser
        - col : le nom de la colonne où se trouve le texte
        - sw : liste des stopwords
        - min_df=1 : nb minimum d'occurence des mots pour être pris en compte
    OUTPUT:
    -------
        - Matrice normalisée TfiDf
    INFORMATION:
    ------------
        - On utilise le stemming pour lemmatization
        -> Nous pourrons remplacer cela par la lemmatization de 'Porter'
    '''
    # Nous chargeons la fonction de comptage des features
    from sklearn.feature_extraction.text import CountVectorizer

    # Partie de lemmatization
    # On charge la librairie de Stemmer en Anglais
    from nltk.stem.snowball import EnglishStemmer
    stemmer = EnglishStemmer()

    # Réalisation du stemming (on coupe les racines)
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    # La tokenisation :
    def tokenize(text):
        m_token_pattern = r"((?:(?:(?:[0-9a-zA-Z])\.){2,}[a-zA-Z])" +\
            "|(?:(?:[0-9a-zA-Z]){2,}\.(?:[0-9a-zA-Z]){2,}" +\
            "|(?:\.(?:[0-9a-zA-Z]){2,}))" +\
            "|[0-9a-zA-Z-\-\+\#]{2,}|w+)"

        from nltk.tokenize import RegexpTokenizer
        # Nous allons utiliser le pattern pour identifier les mots
        tokenizer = RegexpTokenizer(m_token_pattern)

        # Nous lançons la séparation des mots
        tokens = tokenizer.tokenize(text)

        # Appel au stemming pour rapprocher les mots de même racine
        stems = stem_tokens(tokens, stemmer)

        # Etape de nettoyage des valeurs :
        # Nous allons supprimer les nombres sans texte,
        def suppress_nb(x):
            import re
            if x is None:
                return None
            pattern = r'(^[\#\-\+]*[0-9]*$|' +\
                    '^[0-9]*[\#\-\+]*$|' +\
                    '^[0-9]*[\#\-\+]?[0-9]*$|' +\
                    '^[0-9\#\-\+][a-z]$|' +\
                    '^[a-z][0-9\#\-\+]$|' +\
                    '^[0-9]*\.[0-9]*$)'
            if not(re.match(pattern, x)):
                return x

        def nettoie_points(x):
            import re
            if x is None:
                return None

            if (re.match(r'(^[\.\-\#][a-z]*$)', x)):
                return ''.join(list(x)[1:])
            else:
                return x

        # Nous filtrons les nombres seuls
        stems = list(filter(lambda x: suppress_nb(x), stems))
        stems = [nettoie_points(x) for x in stems]
        return stems

    # Le préprocessing :
    #   - nettoie le html
    #   - supprime les blocs de Code
    #   - ne prend que la partie textuelle du html
    #   - renvoie ce texte en minuscule
    def preProcess(html_in):
        # Chargement du module BeautifulSoup pour le parsing des données HTML
        from bs4 import BeautifulSoup
        html_in_soup = BeautifulSoup(html_in, 'html.parser')

        # Suppression des blocs de Code
        for h_code in html_in_soup.find_all('code'):
            h_code.decompose()

        return html_in_soup.get_text().lower()

    # Fabrication du Bag Of Words (BOW) via CountVectorizer

    vectorizer = CountVectorizer(
        analyzer="word",
        # On fait varier la fréquence minimum pour la prise en compte des mots
        min_df=min_df,
        tokenizer=tokenize,
        # token_pattern=m_token_pattern,
        preprocessor=preProcess,
        stop_words=sw,        # Pas de stopWord car nous les cherchons
        ngram_range=(1, 6),   # Nous ne prenons que le 1-Gram
        vocabulary=vocabulary,
        # max_features = 1000  # Nous cherchons les nombres maximum
    )

    import datetime
    debut = datetime.datetime.now()

    # On entraine les données pour fabriquer le Bow
    train_data_features = vectorizer.fit_transform(data[col])

    fin = datetime.datetime.now()
    print("[ Vectorizer : {}]".format(fin-debut))

    # Nous allons renvoie une matrice normalisée
    #  des fréquences des mots dans chaque POST
    from sklearn.feature_extraction.text import TfidfTransformer
    debut = datetime.datetime.now()

    # Attention nous ne voulons pas la forme TFiDF donc use_idf=False
    tf = TfidfTransformer(use_idf=True)
    train_data_features_fitted = tf.fit_transform(train_data_features)

    fin = datetime.datetime.now()
    print("[ TfidfTransformer : {}]".format(fin-debut))

    # On va retourne la matrice dense de résultat
    debut = datetime.datetime.now()

    X = pd.SparseDataFrame(
        data=train_data_features_fitted,
        columns=vectorizer.get_feature_names(),
        default_fill_value=0
    )
    fin = datetime.datetime.now()
    print("[ SparseDataFrame {} ]".format(fin-debut))

    return X


# # ON PERMET DE PRENDRE EN COMPTE LE CODE
def create_matrix_tfidf_V2(data, col, vocabulary,
                           sw=[], min_df=1,
                           isCodeRemoved=True):
    X, model_vec, model_tfidf = create_matrix_tfidf_V2_inside(
            data, col, vocabulary, sw, min_df, isCodeRemoved)

    # On ne retourne que la matrice
    return X


def create_matrix_tfidf_V2_and_models(data, col, vocabulary,
                                      sw=[], min_df=1,
                                      isCodeRemoved=True):
    X, model_vec, model_tfidf = create_matrix_tfidf_V2_inside(
            data, col, vocabulary, sw, min_df, isCodeRemoved)

    # On retourne toutes les valeurs
    return X, model_vec, model_tfidf


# =============================================================================
# FONCTIONS DE PREPROCESSING
# =============================================================================

# Le préprocessing :
#   - nettoie le html
#   - supprime les blocs de Code
#   - ne prend que la partie textuelle du html
#   - renvoie ce texte en minuscule
def preProcess_remove_Code(html_in):
    # Chargement du module BeautifulSoup pour le parsing des données HTML
    from bs4 import BeautifulSoup
    html_in_soup = BeautifulSoup(html_in, 'html.parser')

    # Suppression des blocs de Code
    for h_code in html_in_soup.find_all('code'):
        h_code.decompose()

    return html_in_soup.get_text().lower()


def preProcess_keep_Code(html_in):
    # Chargement du module BeautifulSoup pour le parsing des données HTML
    from bs4 import BeautifulSoup
    html_in_soup = BeautifulSoup(html_in, 'html.parser')

    return html_in_soup.get_text().lower()

# Partie de lemmatization
# On charge la librairie de Stemmer en Anglais


# Réalisation du stemming (on coupe les racines)
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# La tokenisation :


def tokenize(text):
    m_token_pattern = r"((?:(?:(?:[0-9a-zA-Z])\.){2,}[a-zA-Z])" +\
        "|(?:(?:[0-9a-zA-Z]){2,}\.(?:[0-9a-zA-Z]){2,}" +\
        "|(?:\.(?:[0-9a-zA-Z]){2,}))" +\
        "|[0-9a-zA-Z-\-\+\#]{2,}|w+)"

    from nltk.tokenize import RegexpTokenizer
    # Nous allons utiliser le pattern pour identifier les mots
    tokenizer = RegexpTokenizer(m_token_pattern)

    # Nous lançons la séparation des mots
    tokens = tokenizer.tokenize(text)

    # Appel au stemming pour rapprocher les mots de même racine
    stems = stem_tokens(tokens, stemmer)

    # Etape de nettoyage des valeurs :
    # Nous allons supprimer les nombres sans texte,
    def suppress_nb(x):
        import re
        if x is None:
            return None
        pattern = r'(^[\#\-\+]*[0-9]*$|' +\
                '^[0-9]*[\#\-\+]*$|' +\
                '^[0-9]*[\#\-\+]?[0-9]*$|' +\
                '^[0-9\#\-\+][a-z]$|' +\
                '^[a-z][0-9\#\-\+]$|' +\
                '^[0-9]*\.[0-9]*$)'
        if not(re.match(pattern, x)):
            return x

    def nettoie_points(x):
        import re
        if x is None:
            return None

        if (re.match(r'(^[\.\-\#][a-z]*$)', x)):
            return ''.join(list(x)[1:])
        else:
            return x

    # Nous filtrons les nombres seuls
    stems = list(filter(lambda x: suppress_nb(x), stems))
    stems = [nettoie_points(x) for x in stems]
    return stems


def create_matrix_tfidf_V2_inside(data, col, vocabulary,
                                  sw=[], min_df=1,
                                  isCodeRemoved=True):
    '''
    Cette fonction va renvoyer la frequence des mots issues du DataSet.
    Nous mettons en paramètre les STopWords identifiés
    Nous faisons une recherche sur les N-Grams de 1 à 6
    INPUT:
    ------
        - data : dataSet à analyser
        - col : le nom de la colonne où se trouve le texte
        - sw : liste des stopwords
        - min_df=1 : nb minimum d'occurence des mots pour être pris en compte
    OUTPUT:
    -------
        - Matrice normalisée TfiDf
    INFORMATION:
    ------------
        - On utilise le stemming pour lemmatization
        -> Nous pourrons remplacer cela par la lemmatization de 'Porter'
    '''
    # Nous chargeons la fonction de comptage des features
    from sklearn.feature_extraction.text import CountVectorizer

    if isCodeRemoved:
        # preProcess = preProcess_remove_Code
        vectorizer = CountVectorizer(
            analyzer="word",
            # On fait varier la fréquence min pour la prise en compte des mots
            min_df=min_df,
            tokenizer=tokenize,
            # token_pattern=m_token_pattern,
            preprocessor=preProcess_remove_Code,
            stop_words=sw,        # Pas de stopWord car nous les cherchons
            ngram_range=(1, 6),   # Nous ne prenons que le 1-Gram
            vocabulary=vocabulary,
            # max_features = 1000  # Nous cherchons les nombres maximum
        )
    else:
        # preProcess = preProcess_keep_Code
        vectorizer = CountVectorizer(
            analyzer="word",
            # On fait varier la fréquence min pour la prise en compte des mots
            min_df=min_df,
            tokenizer=tokenize,
            # token_pattern=m_token_pattern,
            preprocessor=preProcess_keep_Code,
            stop_words=sw,        # Pas de stopWord car nous les cherchons
            ngram_range=(1, 6),   # Nous ne prenons que le 1-Gram
            vocabulary=vocabulary,
            # max_features = 1000  # Nous cherchons les nombres maximum
        )

    # Fabrication du Bag Of Words (BOW) via CountVectorizer

#    vectorizer = CountVectorizer(
#        analyzer="word",
#        # On fait varier la fréquence minimum pour la prise en compte des mots
#        min_df=min_df,
#        tokenizer=tokenize,
#        # token_pattern=m_token_pattern,
#        preprocessor=preProcess,
#        stop_words=sw,        # Pas de stopWord car nous les cherchons
#        ngram_range=(1, 6),   # Nous ne prenons que le 1-Gram
#        vocabulary=vocabulary,
#        # max_features = 1000  # Nous cherchons les nombres maximum
#    )

    import datetime
    debut = datetime.datetime.now()

    # On entraine les données pour fabriquer le Bow
    train_data_features = vectorizer.fit_transform(data[col])

    fin = datetime.datetime.now()
    print("[ Vectorizer : {}]".format(fin-debut))

    # Nous allons renvoie une matrice normalisée
    #  des fréquences des mots dans chaque POST
    from sklearn.feature_extraction.text import TfidfTransformer
    debut = datetime.datetime.now()

    # Attention nous ne voulons pas la forme TFiDF donc use_idf=False
    tf = TfidfTransformer(use_idf=True)
    train_data_features_fitted = tf.fit_transform(train_data_features)

    fin = datetime.datetime.now()
    print("[ TfidfTransformer : {}]".format(fin-debut))

    # On va retourne la matrice dense de résultat
    debut = datetime.datetime.now()

    X = pd.SparseDataFrame(
        data=train_data_features_fitted,
        columns=vectorizer.get_feature_names(),
        default_fill_value=0
    )
    fin = datetime.datetime.now()
    print("[ SparseDataFrame {} ]".format(fin-debut))

    return X, vectorizer, tf

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
# #  dont le nom est créé à partir des paramètres

def load_model(path_directory, id_compagnie, type_modelisation):
    from sklearn.externals import joblib
    from os.path import isfile

    nom_fichier = '{}model_{}_{}.pkl'.format(
            path_directory,
            id_compagnie,
            type_modelisation
            )

    # # On ajoute un contrôle d'existence du fichier
    if (isfile(nom_fichier)):
        # # On charge le fichier
        model = joblib.load(nom_fichier)
        print('chargement du fichier {}'.format(nom_fichier))
    else:
        print("Attention le fichier {} n'existe pas".format(nom_fichier))

    # # On retourne le fichier
    return model


# LES DERNIERS


def save_one_model_BAK(path_file, model):
    from sklearn.externals import joblib
    # ##import os

    # On copie le fichier
    # #  file = open(“testfile.txt”,”w”)
    joblib.dump(model, open(path_file, 'wb'))

    # joblib.dump(model, path_file)
    # ####print('sauvegarde du fichier {}'.format(os.path.basename(path_file)))


def save_one_model(path_file, model):
    from sklearn.externals import joblib
    # ##import os

    # On copie le fichier
    # #  file = open(“testfile.txt”,”w”)
    with open(path_file, 'wb') as f:
        joblib.dump(model, f)

    # joblib.dump(model, path_file)
    # ####print('sauvegarde du fichier {}'.format(os.path.basename(path_file)))


def save_list_model(l_model, subPath='BODY', isDated=False):
    '''Stocke les model dans un sous repertoire de data
    INPUT:
    -----
        - l_model : la liste des models à sauvegarder
        - subPath : sous repertoire BODY/TITLE
        - isDated : date du sous rep pour permettre des tests
    '''
    import os
    import datetime

    if isDated:
        # On date le repertoire de destination (Utile pour tests)
        u1 = datetime.datetime.now()
        m_date = '{AA}-{MM}-{JJ}_{H}{M}{S}'.format(
            AA=u1.year, MM=u1.month, JJ=u1.day,
            H=u1.hour, M=u1.minute, S=u1.second
            )
    else:
        m_date = 'PROD'

    m_path = 'data/MODEL/{SUB_PATH}/{DATE}/'.format(
            SUB_PATH=subPath, DATE=m_date)

    # On créé le repertoire des données s'il est manquant
    if not(os.path.isdir(m_path)):
        os.makedirs(m_path)

    pattern_name = m_path + 'model_{no}.mod'

    for idx, clf in enumerate(l_model):
        save_one_model(pattern_name.format(no=idx), clf)

# =============================================================================
# MESURES
# =============================================================================


def my_mesure_moyenne_accuracy(y_test, y_pred):
    '''On mesure la moyenne des bons tags'''
    diff = (y_test == y_pred)
    return np.round(a=np.mean(np.sum(diff, axis=0)/diff.shape[0]), decimals=4)


def my_mesure_moyenne_accuracy_true_positives(y_test, y_pred):
    '''On mesure la performance a trouver qu'un tag est present'''
    mask1 = (y_test == 1)
    nb_valeurs = np.sum(np.sum(mask1, axis=0))
    diff = ((y_test == y_pred) & (y_test == 1))

    return np.round(a=np.sum(np.sum(diff, axis=0))/nb_valeurs, decimals=4)


def my_mesure_moyenne_accuracy_false_positives(y_test, y_pred):
    '''On mesure la performance a trouver qu'un tag est absent'''
    mask1 = (y_test == 0)
    nb_valeurs = np.sum(np.sum(mask1, axis=0))
    diff = ((y_test == y_pred) & (y_test == 0))

    return np.round(a=np.sum(np.sum(diff, axis=0))/nb_valeurs, decimals=4)


def my_mesure_moyenne_accuracy_all_tags(y_test, y_pred):
    '''On mesure la moyenne des posts ayant tous les bons tags'''
    diff = (y_test == y_pred)
    return np.mean(np.sum(diff, axis=1)/y_test.shape[1] == 1)
    return np.mean(diff, axis=0)

# ## PAR TAG


def my_mesure_moyenne_accuracy_per_tag(y_test, y_pred):
    diff = (y_test == y_pred)
    return np.mean(diff, axis=0)


def my_mesure_moyenne_accuracy_true_positives_per_tag(y_test, y_pred):
    '''On mesure la performance a trouver qu'un tag est present par tag'''
    mask1 = (y_test == 1)
    nb_valeurs = np.sum(mask1, axis=0)
    diff = ((y_test == y_pred) & (y_test == 1))

    return np.round(a=np.sum(diff, axis=0) / nb_valeurs, decimals=4)


def my_mesure_moyenne_accuracy_false_positives_per_tag(y_test, y_pred):
    '''On mesure la performance a trouver qu'un tag est absent par tag'''
    mask1 = (y_test == 0)
    nb_valeurs = np.sum(mask1, axis=0)
    diff = ((y_test == y_pred) & (y_test == 0))

    return np.round(a=np.sum(diff, axis=0)/nb_valeurs, decimals=4)


# ### AFFICHAGE GLOBAL

def affichage_mesures(y_test2, y_pred5):
    '''Affiche une liste de mesures clefs'''
    print('- Moyenne globale \t\t\t= {}\n'
          '- Moyenne globale [python]\t\t= {}\n'
          '- Moyenne globale [java]\t\t= {}\n'
          '- Moyenne globale [c++]\t\t\t= {}\n'

          '- Moyenne pour chaque Post\t\t= {}\n\n'
          '- Moyenne des Vrai positifs\t\t= {}\n'
          '- Moyenne des Vrai positifs [python]\t= {}\n'
          '- Moyenne des Vrai positifs [java]\t= {}\n'
          '- Moyenne des Vrai positifs [c++]\t= {}\n'.format(
              my_mesure_moyenne_accuracy(y_test=y_test2, y_pred=y_pred5),
              my_mesure_moyenne_accuracy_per_tag(
                      y_test=y_test2, y_pred=y_pred5)['python'],
              my_mesure_moyenne_accuracy_per_tag(
                      y_test=y_test2, y_pred=y_pred5)['java'],
              my_mesure_moyenne_accuracy_per_tag(
                      y_test=y_test2, y_pred=y_pred5)['c++'],
              my_mesure_moyenne_accuracy_all_tags(
                      y_test=y_test2, y_pred=y_pred5),
              my_mesure_moyenne_accuracy_true_positives(
                      y_test=y_test2, y_pred=y_pred5),
              my_mesure_moyenne_accuracy_true_positives_per_tag(
                      y_test=y_test2, y_pred=y_pred5)['python'],
              my_mesure_moyenne_accuracy_true_positives_per_tag(
                      y_test=y_test2, y_pred=y_pred5)['java'],
              my_mesure_moyenne_accuracy_true_positives_per_tag(
                      y_test=y_test2, y_pred=y_pred5)['c++']
          ))
