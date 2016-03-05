# -*- coding: utf-8 -*-
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing

model_name1 = 'model/inception-v3/feat'
model_name2 = 'model/21k/feat_21k_1'
model_name3 = 'model/fb.resnet.torch/pretrained/features101'

x1 = np.fromfile(model_name1+'_train', dtype=np.float32).reshape(-1, 2048)
x2 = np.fromfile(model_name2+'_train', dtype=np.float32).reshape(-1, 1024)
x3 = np.fromfile(model_name3+'_train', dtype=np.float32).reshape(-1, 2048)

print (x1[0, :10])
print (x2[0, :10])
print (x3[0, :10])
print ('train: ', x1.shape, x2.shape, x3.shape)

x = np.concatenate([x2], axis=1)
pre = preprocessing.Normalizer(norm='l2')
x = pre.transform(x)
#print (x.shape)
#x = x[:6000]
#print (x.shape)

#n_comp = 256

#mean = np.mean(x, axis=0)
#x = x - mean
#svd = decomposition.TruncatedSVD(n_components=n_comp
#    , algorithm='arpack'
#    )
#svd.fit(x[:100000])
#print (svd.explained_variance_ratio_.sum())
#x = svd.transform(x).astype(np.float32)
#x.tofile('/home/dima/yelp/train_feat')

#nmf = decomposition.NMF(n_components=n_comp, max_iter=50)
#nmf.fit(x[:10000])
#print(nmf.reconstruction_err_)
#comp = nmf.components_.astype(np.float32).T
#print (comp.shape)
#nmf.transform(x).tofile('/home/dima/yelp/train_feat')
#x.dot(comp).tofile('/home/dima/yelp/train_feat')
x.tofile('/home/dima/yelp/train_feat')

#qwe
# test compressing

del x1, x2, x3, x
x1t = np.fromfile(model_name1+'_test', dtype=np.float32).reshape(-1, 2048)
x2t = np.fromfile(model_name2+'_test', dtype=np.float32).reshape(-1, 1024)
x3t = np.fromfile(model_name3+'_test', dtype=np.float32).reshape(-1, 2048)
xt = np.concatenate([x2t], axis=1)
xt = pre.transform(xt)

print ('test: ', x1t.shape, x2t.shape, x3t.shape)
#svd.transform(xt - mean).tofile('/home/dima/yelp/test_feat')
xt.tofile('/home/dima/yelp/test_feat')
