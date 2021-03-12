from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

from scipy import stats

from sklearn import datasets, linear_model
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from collections import Counter

from yellowbrick.cluster import KElbowVisualizer
from scipy.stats import norm
import os

def generate_cluster(filename):
    df_users=pd.read_csv(filename)
    df_users.dropna(inplace=True)
    pruebas=df_users[['votos_utiles', 'votos_graciosos', 'votos_geniales','cantidad_fans', 'promedio_estrellas_reseñas', 'halagos_hot',
       'halagos_mas', 'halagos_perfil', 'halagos_adorable', 'halagos_lista',
       'halagos_nota', 'halagos_simple', 'halagos_genial', 'halagos_gracioso',
       'halagos_escritor', 'halagos_foto']]
    min_max_scaler=MinMaxScaler()

    pruebas_escalada=min_max_scaler.fit(pruebas)
    pruebas_escalada=min_max_scaler.transform(pruebas)

    filename_model_cluster="../serialized_data/gm_model.pk"
    file_model=open(filename_model_cluster,"rb")
    model_gm=pickle.load(file_model)
    file_model.close()

    y=model_gm.predict(pruebas_escalada)
    df_user_escalado=pd.DataFrame(pruebas_escalada,columns=['votos_utiles', 'votos_graciosos', 'votos_geniales','cantidad_fans', 'promedio_estrellas_reseñas', 'halagos_hot',
       'halagos_mas', 'halagos_perfil', 'halagos_adorable', 'halagos_lista',
       'halagos_nota', 'halagos_simple', 'halagos_genial', 'halagos_gracioso',
       'halagos_escritor', 'halagos_foto'])

    df_user_escalado["BICModel"]=y
    pruebas["BICModel"]=y

    if not(os.path.isdir("../datasets/")):
        os.makedirs('../datasets', exist_ok=True)
    pruebas.to_csv("../datasets/dataset_with_bic_model")
    return (pruebas,model_gm,pruebas_escalada)

