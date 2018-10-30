from .views import app
from .prediction_classe import Prediction

# Initialisation de la prédiction
Prediction = Prediction(app.config['MODELS_DIRECTORY_PATH'])
