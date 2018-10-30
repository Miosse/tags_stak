import os
SECRET_KEY = '#d#JCqTTW\nilK\\7m\x0bp#\tj~#H'

# Database initialization
basedir = os.path.abspath(os.path.dirname(__file__))

MODELS_DIRECTORY_PATH = basedir + '/prog_tags/data/MODEL/'

DATA_DIRECTORY_PATH = basedir + '/prog_tags/data/'

# SQLALCHEMY_TRACK_MODIFICATIONS = True
# SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')

# AEROPORT_DATABAS_URI = os.path.join(basedir,
#    'retards_avions/datas/aeroports_nettoyes.csv')

# AEROPORT_DATABAS_URI = os.path.join(
#        basedir, 'retards_avions/datas/aeroports_nettoyes.csv')
# VOLS_DATABAS_URI = os.path.join(
#       basedir, 'retards_avions/datas/datas_total_nettoyees-Q4.csv')
# REPERTOIRE_STOCKAGE_MODELS_URI = os.path.join(
#       basedir, 'retards_avions/datas/MODELS/')
