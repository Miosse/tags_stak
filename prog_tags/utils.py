# -*- coding: utf-8 -*-

import pandas as pd

from .prediction_classe import Prediction

Pred = Prediction()

# Chargement des données
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
        'Body': html_in_soup.get_text(),
        'Tags': m_tags
    }
    return res


if __name__ == "__main__":
    pass
