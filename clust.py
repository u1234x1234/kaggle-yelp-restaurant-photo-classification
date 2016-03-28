# -*- coding: utf-8 -*-
import numpy as np
from sklearn import manifold, decomposition
from sklearn import cluster

n=300000
X = np.fromfile('train_feat', dtype=np.float32, count=n*2048).reshape(-1, 2048)
X_test = np.fromfile('test_feat', dtype=np.float32, count=500000*2048).reshape(-1, 2048)
print (X.shape, X_test.shape)
#XX = np.concatenate([X, X_test])
#print (XX.shape)
mean = X.mean(axis=0)
print(mean.shape)
X -= mean
X_test -= mean

pca = decomposition.TruncatedSVD(n_components=128)

pca.fit(X)
X = pca.transform(X).astype(np.float32)
X.tofile('train_feat_128')
X_test = pca.transform(X_test).astype(np.float32)
X_test.tofile('test_feat_128')
#print (X.shape)
#
#print (pca.explained_variance_ratio_.sum())
#print (pca.components_.dtype)
#
#print (pca.components_.shape)
#pca.mean_.tofile('mean')
#pca.components_.tofile('comp')


#X = np.fromfile('train_feat_128', dtype=np.float32, count=n*128).reshape(-1, 128)
#print (X.shape)
#cl = cluster.KMeans(n_clusters=64)
##cl = cluster.MiniBatchKMeans(n_clusters=64, batch_size=100000, init_size=100000, n_init=10)
#
#cl.fit(X)
#
#print (cl.inertia_)
#cl.cluster_centers_ = cl.cluster_centers_.astype(np.float32)
#print (cl.cluster_centers_.dtype)
#
#cl.cluster_centers_.tofile('centers')