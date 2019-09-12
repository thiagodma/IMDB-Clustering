import numpy as np
import pickle
from scipy.cluster import hierarchy
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

X = np.load('X.npy')
y = np.load('y.npy')
with open("texts.txt", "rb") as fp:
    texts = pickle.load(fp)

X = preprocessing.scale(X)

X_aux = np.zeros((1,1200))
for i in range(X.shape[0]):
    Xi_norm = X[i,:]/np.linalg.norm(X[i,:])
    Xi_norm.shape=(1,1200)
    X_aux = np.append(X_aux,Xi_norm,axis=0)

X_aux = np.delete(X_aux, (0), axis=0)

clusters_por_cosseno = hierarchy.linkage(X_aux,"average", metric="cosine")
#plt.figure()
#dn = hierarchy.dendrogram(clusters_por_cosseno)
#plt.savefig('dendogram.jpg')


#kmeans = KMeans(n_clusters=2, random_state=0,)
#id_clusters = kmeans.fit_predict(X_aux)

clusters_por_cosseno = hierarchy.linkage(X,"average", metric="cosine")
limite_dissimilaridade = 1
id_clusters = hierarchy.fcluster(clusters_por_cosseno, limite_dissimilaridade, criterion="distance")


df = pd.DataFrame(list(zip(id_clusters,y)), columns=['cluster_id','text'])
