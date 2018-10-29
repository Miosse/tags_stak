import pandas as pd
from os import path

class Aeroport():
    """Classe de Stockage de la liste des Aeroports"""
    _nom_fichier = 'aeroports_nettoyes.csv'
    _file_loaded = False

    # Lien sur les vols
    _vols = None
    
    @classmethod
    def __init__(self, nom_fichier = None):
        if nom_fichier!=None:
            self._nom_fichier = nom_fichier
            
        if not(self._file_loaded):
            OPERATION_VALIDE = True
            try:
                self._data = pd.read_csv(self._nom_fichier,index_col='AIRPORT_SEQ_ID')
            except:
                OPERATION_VALIDE = False
                print('-- AEROPORT : Erreur de chargement de ',self._nom_fichier)
            finally:
                if (OPERATION_VALIDE):
                    self._file_loaded = True
                    path_dir = path.dirname(self._nom_fichier)
                    path_basename = path.basename(self._nom_fichier)
                    print("-- AEROPORT : dirname = '{}'".format(path_dir))
                    print("-- AEROPORT : Fichier '{}' chargé --> Taille : {}".format(
                        path_basename, len(self._data)))
        else:
            print('-- AEROPORT : FICHIER DEJA CHARGE')
    
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

    @classmethod
    def __init__(self, nom_fichier = None):
        if nom_fichier!=None:
            self._nom_fichier = nom_fichier
            
        if not(self._file_loaded):
            OPERATION_VALIDE = True
            try:
                self._data = pd.read_csv(self._nom_fichier)
            except:
                OPERATION_VALIDE = False
                print('-- VOLS : Erreur de chargement de ',self._nom_fichier)
            finally:
                if (OPERATION_VALIDE):
                    self._file_loaded = True
                    path_dir = path.dirname(self._nom_fichier)
                    path_basename = path.basename(self._nom_fichier)
                    print("-- VOLS : dirname = '{}'".format(path_dir))
                    print("-- VOLS : Fichier '{}' chargé --> Taille : {}".format(
                        path_basename, len(self._data)))
        else:
            print('-- VOLS : FICHIER DEJA CHARGE')

        # On créer aussi les données indexées
        self.create_indexed_flies(self)
    
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

    ## Methodes pour la gestion des vols
    def get_vols(self, ville_origine = 'Dallas, TX',
                 ville_arrivee = 'Chicago, IL',
                 colums_sortie = ['ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 
                                  'FL_DATE', 'AIRLINE_ID', 'ARR_HOUR', 'DEP_HOUR'],
                 data_recherche = {'date_depart': None, 'airline_id': None, 
                                   'airline_id': None, 
                                   'heure_dep': None, 'heure_arr': None}
                ):
        l_vols_possibles = []
        get_code = lambda x: x['code']
        
        l_ville_dep = self._aeroports.get_ville(ville=ville_origine, with_code = True)
        l_ville_arr = self._aeroports.get_ville(ville=ville_arrivee, with_code = True)
        
        l_ville_dep = [get_code(x) for x in l_ville_dep]
        l_ville_arr = [get_code(x) for x in l_ville_arr]
        
        subset = self[l_ville_dep, l_ville_arr, :]
        subset = subset.reset_index().groupby(colums_sortie
                                   ).size().reset_index(name='Freq')
        return subset



    

