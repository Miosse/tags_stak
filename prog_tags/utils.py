# -*- coding: utf-8 -*-

#import pandas as pd

from .models import Prediction, Post

# from .prediction_classe import Prediction
# Pred = Prediction(MODELS_DIRECTORY_PATH)

# Chargement des données
# file1 = 'prog_tags/data/QueryResults3.csv'
# Post = pd.read_csv(file1, index_col=['Id'])

# Fonction qui renvoie la prédiction pour un titre et un corps de texte

#  #########################
#   #####   Récupération de la liste des mots prédits
#   #########################


def get_prediction(t_title, t_body):
    '''Renvoie la liste de prédiction pour le Titre t_title,
    et le Corps du texte t_body'''
    return Prediction.predict(
            texte_body=t_body,
            texte_title=t_title)

#  #########################
#   #####   Récupération de la liste des mots prédits: idem
#   #########################


def get_prediction2(val1, val2):
    '''Renvoie la liste de prédiction pour le Titre t_title,
    et le Corps du texte t_body'''
    return Prediction.predict(
            texte_body=val1,
            texte_title=val2)


#  #########################
#   #####   Récupération des informations d'un post existant
#   #########################

def recupere_post_id(post_id=None):
    '''Récupère les informations nécessaires pour un Id en entrée'''
    import re
    from bs4 import BeautifulSoup

    if post_id is None:
        # Alors on génère un id aléatoirement
        import random
        secure_random = random.SystemRandom()
        post_id = secure_random.choice(Post.index)
    # On verifie que l'id est dans la liste disponible
    elif post_id not in Post.index:
        import random
        secure_random = random.SystemRandom()
        post_id = secure_random.choice(Post.index)

    u = Post[Post.index == post_id][['Tags', 'Body', 'Title']]

    m_tags = re.findall(r'<(\w+|[0-9a-zA-Z-\-\.\+\#]+)>', u['Tags'].values[0])

    # On ajoute une balise <em> si le mot n'est pas dans les tags
    #  sélectionnés dans le modèle
    m_tags = [check_tag_word(w) for w in m_tags]

    html_in_soup = BeautifulSoup(u['Body'].values[0], 'html.parser')
    html_in_soup.get_text()

    res = {
        'Title': u['Title'].values[0],
        'Body': html_in_soup.get_text(),
        'Tags': m_tags
    }
    return res


#  #########################
#   #####   Test si un terme est dans la liste du vocabulaire des Tags
#   #########################

def check_tag_word(tag_in):
    if tag_in in Prediction._voc_tags:
        # Si le tag est dans les tags utilisés on le retourne
        return tag_in
    else:
        # Si le tag n'est pas dans les tags utilisés alors on
        # le met en valeur en l'entourrant de balise <em>
        return '<em>{}<em>'.format(tag_in)


if __name__ == "__main__":
    pass
