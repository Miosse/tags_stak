from .views import app
from .prediction_classe import Prediction

import pandas as pd

# Initialisation de la prédiction
Prediction = Prediction(app.config['MODELS_DIRECTORY_PATH'])

file1 = app.config['DATA_DIRECTORY_PATH'] + '/QueryResults3.csv'
Post = pd.read_csv(file1, index_col=['Id'])
