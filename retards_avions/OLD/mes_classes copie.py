import pandas as pd
from os import path

class Aeroport():
    """Classe de Stockage de la liste des Aeroports"""
    _nom_fichier = 'aeroports_nettoyes.csv'
    _file_loaded = False
    
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
        if self._file_loaded:
            return True
        else:
            return False
    
    def _get_data(self):
        return self._data
    
    #@staticmethod
    def _get_aeroports(self):
        return self._aeroports
    
    #@staticmethod
    def _set_aeroports(self, Aeroport):
        self._aeroports = Aeroport
        
    state = property(_get_state)
    data = property(_get_data)
    aeroport = property(_get_aeroports, _set_aeroports)