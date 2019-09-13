import numpy as np
import pickle
from scipy.cluster import hierarchy
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

X = np.load('X.npy')
y = np.load('y.npy')
with open("texts.txt", "rb") as fp:
    texts = pickle.load(fp)

X = preprocessing.scale(X)

X_train = X[0:799,:]
y_train = y[0:799]
X_valid= X[799:,:]
y_valid = y[799:]

#import pdb; pdb.set_trace()

clf = MLPClassifier(alpha=0.85)
clf.fit(X_train,y_train)
print(clf.score(X_valid,y_valid))
