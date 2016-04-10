# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import cross_validation, metrics, ensemble, neighbors, decomposition, preprocessing
import sklearn
import xgboost as xgb

features = [
#           ('21k_1024.npy', sklearn.linear_model.LogisticRegression(C=100)),
#           ('v3_2048.npy', sklearn.linear_model.LogisticRegression(C=100)),
#           ('res_full_l2.npy', sklearn.linear_model.LogisticRegression(C=1)),
            ('21k_50k_2048.npy', sklearn.linear_model.LogisticRegression(C=100)),
           ('21k_v3_3072.npy', sklearn.linear_model.LogisticRegression(C=100)),
            ('21k_v3_128.npy', sklearn.linear_model.LogisticRegression(C=50)),
#            ('21k.npy', sklearn.linear_model.LogisticRegression(C=50)),
           ('fisher.npy', sklearn.linear_model.LogisticRegression(C=2)),
            ('v3_full.npy', sklearn.linear_model.LogisticRegression(C=100)),
            ('21k_full.npy', sklearn.linear_model.LogisticRegression(C=100)),
#            ('vlad_2_21k_full.npy', sklearn.linear_model.LogisticRegression(C=1)),
#           ('21k_v3_128.npy', xgb_wrapper()),
#           ('fisher_21k_1024.npy', sklearn.linear_model.LogisticRegression(C=2))
#           ('v3.npy', sklearn.linear_model.LogisticRegression(C=100)),
            
            ('vlad_2_21k_full.npy', xgb.sklearn.XGBClassifier(learning_rate=0.1, n_estimators=100, nthread=8,
                                max_depth=3, subsample=0.8, colsample_bytree=0.8)),
#            ('jo.npy', xgb.sklearn.XGBClassifier(learning_rate=0.1, n_estimators=100, nthread=8,
#                max_depth=4, subsample=0.9, colsample_bytree=0.9))
            ]

def f(weights):
   preds = np.array([])
   for feature, clf in features:    
 
       preds_br = np.load('test/' + feature + '_br.npy')
       preds_nn = np.load('test/' + feature + '_nn.npy')
       preds_cc = np.load('test/' + feature + '_cc.npy')       
#       preds_br = (1*preds_br + 3*preds_nn + 2*preds_cc) / 6
       
       preds = np.concatenate((preds, preds_br[..., np.newaxis]), axis=2) \
           if preds.size else preds_br[..., np.newaxis]
       preds = np.concatenate((preds, preds_nn[..., np.newaxis]), axis=2)
       preds = np.concatenate((preds, preds_cc[..., np.newaxis]), axis=2)
               
   for i in range(21):
       preds[:, :, i] *= weights[i]           

   preds = np.mean(preds, axis=2)
   preds = preds > 0.224

   return preds
       
weights = np.array([1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    4.6])
preds = f(weights)
#n_iter = 1000
f = open('res', 'w')
print('business_id,labels', file=f)
ids = pd.read_csv('sample_submission.csv').values[:, 0]
for biz_id, pred in zip(ids, preds):
    nz = pred.nonzero()
    nz = [str(x) for x in nz[0]]
    print (biz_id + ',' + ' '.join(nz), file=f)