import numpy as np
import xgboost as xgb
import sklearn
import os
import pandas as pd
from classifier_chain import ClassifierChain
from sklearn import cross_validation, metrics, ensemble, neighbors, decomposition, preprocessing
from nn_wrapper import nn_wrapper
from scipy import optimize
from best.nn_joint import nn_joint

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


def f_real(weights):
    kf = cross_validation.KFold(2000, n_folds=10, shuffle=True, random_state=0)
    re = np.array([])
    fold = 0
    for train_index, test_index in kf:
    
       y = np.load('y_train.npy')
       y_val = y[test_index]
       preds = np.array([])
       for feature, clf in features:    
 
           preds_br = np.load('val5/' + str(fold) + '_' + feature + '_br.npy')
           preds_nn = np.load('val5/' + str(fold) + '_' + feature + '_nn.npy')
           preds_cc = np.load('val5/' + str(fold) + '_' + feature + '_cc.npy')       
#           preds_br = (preds_br + 3*preds_nn + 2*preds_cc) / 6
           
           preds = np.concatenate((preds, preds_br[..., np.newaxis]), axis=2) \
               if preds.size else preds_br[..., np.newaxis]
           preds = np.concatenate((preds, preds_nn[..., np.newaxis]), axis=2)
           preds = np.concatenate((preds, preds_cc[..., np.newaxis]), axis=2)
                   
       for i in range(21):
           preds[:, :, i] *= weights[i]
       preds = np.sum(preds, axis=2)
       
       preds = preds > weights[21]
       
       score = metrics.f1_score(y_val, preds, average='samples')
       print('score chain classifier: ', score)
       re = np.hstack([re, score]) if re.size else score
       fold = fold + 1
    return re.mean()

#th = np.array([0.45, 0.42, 0.45, 0.43, 0.43, 0.45, 0.45, 0.42, 0.42])
def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))
features_jo = [
('21k_1024.npy', 1),
('v3_2048.npy', 1),
#('21k_v3_3072.npy', 1),
#('vlad_21k_64_16.npy', 1),
#('res_l2.npy', 1),
#('21k_v3_128.npy', 1),
]
def jo(mode, args):
    x = np.hstack([k * np.load(mode + '_' + features) for (features, k) in args])
    return x
    
clf = nn_joint()
x_j = jo('train', features_jo)

cl = np.zeros(9, dtype=np.float32)
#def f(weights):
kf = cross_validation.KFold(2000, n_folds=10, shuffle=True, random_state=0)
re = np.array([])
fold = 0
weights = np.array([1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    4.6])
for train_index, test_index in kf:

   y = np.load('y_train.npy')
   y_val = y[test_index]
   preds = np.array([])
   for feature, clf in features:    
 
       preds_br = np.load('val6/' + str(fold) + '_' + feature + '_br.npy')
       preds_nn = np.load('val6/' + str(fold) + '_' + feature + '_nn.npy')
       preds_cc = np.load('val6/' + str(fold) + '_' + feature + '_cc.npy')
       
#           print(preds_br.shape, preds_nn.shape, preds_cc.shape)
       preds = np.concatenate((preds, preds_br[..., np.newaxis]), axis=2) \
           if preds.size else preds_br[..., np.newaxis]
       preds = np.concatenate((preds, preds_nn[..., np.newaxis]), axis=2)
       preds = np.concatenate((preds, preds_cc[..., np.newaxis]), axis=2)
               
#       filename = 'val3/joint_' + str(fold) + '.npy'
#       X_train, X_val = x_j[train_index], x_j[test_index]
#       y_train = y[train_index]
#       nn_joint_preds = np.array([])
   
#       if os.path.exists(filename):
#           nn_joint_preds = np.load(filename)
#       else:
#           print('no such model pred: ', filename, 'train..')
#           n_iter = 1
#           for i in range(n_iter):    
#               clf = nn_joint()
#               clf.fit(X_train[:, :1024], X_train[:, 1024:], y_train)
#               r = clf.predict(X_val[:, :1024], X_val[:, 1024:])
#               nn_joint_preds = nn_joint_preds + r if nn_joint_preds.size else r
#           nn_joint_preds /= n_iter
#           np.save(filename, nn_joint_preds)

   for i in range(21):
       preds[:, :, i] *= weights[i]           
   preds = np.mean(preds, axis=2)

#       preds = (preds + nn_joint_preds*0.2) / 2
   
   preds = preds > 0.219
   
   score = metrics.f1_score(y_val, preds, average='samples')
   
#       print(y_val.shape, preds.shape)
   for qq in range(9):
       r = metrics.f1_score(y_val[:, qq], preds[:, qq])
       cl[qq] += r
#           print('class ', qq, ' : ', r, average='binary'))
#       score = metrics.mean_squared_error(y_val, preds)
#       print('score chain classifier: ', score)
   re = np.hstack([re, score]) if re.size else score
   fold = fold + 1
#return re.mean()

#1            ('21k_50k_2048.npy', sklearn.linear_model.LogisticRegression(C=100)),
#2          ('21k_v3_3072.npy', sklearn.linear_model.LogisticRegression(C=100)),
#3            ('21k_v3_128.npy', sklearn.linear_model.LogisticRegression(C=50)),
#4           ('fisher.npy', sklearn.linear_model.LogisticRegression(C=2)),
#5            ('v3_full.npy', sklearn.linear_model.LogisticRegression(C=100)),
#6            ('21k_full.npy', sklearn.linear_model.LogisticRegression(C=100)),
#7            ('vlad_2_21k_full.npy'
cl /= 10
print('cl', cl, cl.mean())
weights = np.array([1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    1/6, 1, 2/6,
                    4.6])
n_iter = 1000
mx = 0
best_w = np.array([])
#for it in range(n_iter):
for th in np.linspace(0.2, 0.3, 0):
#    weights = np.zeros(22)
#    for i in range(7):
#        weights[i*3] = np.random.uniform(0, 2/6)
#        weights[i*3+1] = np.random.uniform(0.8, 1.2)
#        weights[i*3+2] = np.random.uniform(0, 4/6)    
    weights[21] = th
    score = f(weights)
    if score > mx:
        mx = score
        best_w = np.array(weights)
    print(score, mx, best_w[21])
#n = 6
#weights[n*3] = 0
#res = optimize.minimize(f, weights, method='nelder-mead')
#res = optimize.basinhopping(f, weights)
#print(res)
#weights = res.x
print(weights)
print('overall:', f(weights))
    
