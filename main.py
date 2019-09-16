import numpy as np
import pickle
from scipy.cluster import hierarchy
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import re
from sklearn.ensemble import RandomForestClassifier

#Loads the data
X = np.load('X.npy')
y = np.load('y.npy')
with open("texts.txt", "rb") as fp:
    texts = pickle.load(fp)

#Cleans the text so that it is more readable
texts_clean = [re.sub(r'xxbos|xxmaj|xxunk|xxeos|','',text) for text in texts]

#Normalizes to mean zero and std 1
X = preprocessing.scale(X)

#I train a RandomForestClassifier so that i can discard the useless features
X_train = X[0:799,:]
y_train = y[0:799]
X_valid= X[799:,:]
y_valid = y[799:]
clf = RandomForestClassifier(n_estimators=20,max_depth=10)
clf.fit(X_train,y_train)
print(clf.score(X_valid,y_valid))
fi = clf.feature_importances_

idx = np.where(fi>0)[0]
#I want all the features that had some relevance
X = X[:,idx]

clusters_por_cosseno = hierarchy.linkage(X,"average", metric="cosine")
#plt.figure()
#dn = hierarchy.dendrogram(clusters_por_cosseno)
#plt.savefig('dendogram.jpg')


limite_dissimilaridade = 0.9
id_clusters = hierarchy.fcluster(clusters_por_cosseno, limite_dissimilaridade, criterion="distance")

#Colocando o resultado em dataframes
clusters = np.unique(id_clusters)
n_normas = np.zeros(len(clusters)) #numero de normas pertencentes a uma cluster
for cluster in clusters:
    idxs = np.where(id_clusters == cluster) #a primeira cluster não é a 0 e sim a 1
    n_normas[cluster-1] = len(idxs[0])

cluster_nnormas = pd.DataFrame(list(zip(clusters,n_normas)),columns=['cluster_id','n_normas'])

df = pd.DataFrame(list(zip(id_clusters,y,texts_clean)), columns=['cluster_id','y','text'])
