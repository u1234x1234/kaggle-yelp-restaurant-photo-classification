# -*- coding: utf-8 -*-
import numpy as np
import os
from sklearn import preprocessing

root_path = '/mnt/disk/data/'
model_name1 = root_path + 'test_feat_21k77_6crop'
feature_dim = 50176

#x = np.fromfile(model_name1, dtype=np.float32, count=feature_dim*30000).reshape(-1, feature_dim)
#
#print ('train: ', x.shape)
#pre = preprocessing.Normalizer(norm='l2')
#preprocessing.normalize(x, copy=False)
#
#n_comp = 2048
#mean = np.mean(x, axis=0)
#x -= mean
#svd = decomposition.TruncatedSVD(n_components=n_comp, algorithm='arpack')
#svd.fit(x)
#print('explained: ', svd.explained_variance_ratio_.sum())
#x = svd.transform(x)
#
#np.save('online_mean_2048.npy', mean)
#np.save('online_components_2048.npy', svd.components_)

mean = np.load('online_mean_2048.npy')
comp = np.load('online_components_2048.npy')

batch_size = 10000
filename = root_path + 'feat_21k77_6crop_2048' + '_test'
try:
    os.remove(filename)
except:
    pass
f_out = open(filename, 'ab')

for i in range(10000):
    f = open(model_name1, 'rb')
    f.seek(i * batch_size * feature_dim * 4, os.SEEK_SET)
    x = np.fromfile(f, dtype=np.float32, count=feature_dim*batch_size).reshape(-1, feature_dim)
    preprocessing.normalize(x, copy=False)
    x = x - mean
    x = x.dot(comp.T)
    x.tofile(f_out)
    print(i, x.shape)
    if x.shape[0] < batch_size:
        break


