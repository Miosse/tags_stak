from flask_sqlalchemy import SQLAlchemy
import logging as lg
import pandas as pd

from .views import app
from .mes_classes import Aeroport, Vols

# Create database connection object
db = SQLAlchemy(app)

#Aeroport2 = Aeroport(app.config['AEROPORT_DATABAS_URI'])

class Aeroport(Aeroport):
    _nom_fichier = app.config['AEROPORT_DATABAS_URI']

class Vols(Vols):
    _nom_fichier = app.config['VOLS_DATABAS_URI']


Aeroport = Aeroport()
Vols = Vols()

Vols.aeroport = Aeroport
Aeroport.vols = Vols

class Content(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	description = db.Column(db.String(200), nullable=False)
	gender = db.Column(db.Integer(), nullable=False)

	def __init__(self, description, gender):
		self.description = description
		self.gender = gender

def charge_df():
    FILE = 'datas/aeroports_nettoyes.csv'
    df = pd.read_csv(FILE)
     
    return df


def init_db():
    #Aeroport = Aeroport()
    
	#db.drop_all()
	#db.create_all()
	#db.session.add(Content("Super", 1))
	#db.session.add(Content("Cool2", 0))
	#db.session.add(Content("Toc is the best",1))
	#db.session.commit()
    lg.warning('Fichier Aeroport chargé {state}'\
               .format(state=Aeroport.state)) 


######db.create_all()
