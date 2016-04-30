1# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing
from scipy.linalg import qr
model_names = ['model/inception-v3/feat_v3_6crop2',
                 'model/21k/feat_21k_6',
                 'model/fb.resnet.torch/pretrained/features101',
                 'color',
                 '/mnt/disk/data/feat_21k_50k_6crop_2048'
]

train_size = 234842

#x1 = np.fromfile(model_names[0]+'_train', dtype=np.float32).reshape(train_size, -1)
x2 = np.fromfile(model_names[1]+'_train', dtype=np.float32).reshape(train_size, -1)
#x3 = np.fromfile(model_names[2]+'_train', dtype=np.float32).reshape(train_size, -1)
#x4 = np.fromfile(model_names[3]+'_train', dtype=np.int64).astype(np.float32).reshape(train_size, -1)
#x = np.fromfile(model_names[4]+'_train', dtype=np.float32).reshape(train_size, -1)

#x = x[:, :256]
preprocessing.normalize(x2, copy=False)

x = np.concatenate([x2], axis=1)
#n_comp = 64
#mean = np.mean(x, axis=0)
#x = x - mean
#svd = decomposition.TruncatedSVD(n_components=n_comp, algorithm='arpack')
#svd.fit(x)
#print('explained: ', svd.explained_variance_ratio_.sum())
#x = svd.transform(x)

x.tofile('train_feat')
#qwe
#np.save('train', x)
#qwe
# test compressing

#del x1, x2, x3, x
test_size = 237152
#x1 = np.fromfile(model_names[0]+'_test', dtype=np.float32).reshape(test_size, -1)
x2 = np.fromfile(model_names[1]+'_test', dtype=np.float32).reshape(test_size, -1)
#x3 = np.fromfile(model_names[2]+'_test', dtype=np.float32).reshape(test_size, -1)
#x4 = np.fromfile(model_names[3]+'_test', dtype=np.int64).astype(np.float32).reshape(test_size, -1)
#x5 = np.fromfile(model_names[4]+'_test', dtype=np.float32).reshape(test_size, -1)

preprocessing.normalize(x2, copy=False)

xt = np.concatenate([x2], axis=1)
#xt = xt - mean
#xt = svd.transform(xt)

xt.tofile('test_feat')


