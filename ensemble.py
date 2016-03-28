import numpy as np
import xgboost as xgb
import sklearn
import pandas as pd
from classifier_chain import ClassifierChain
from sklearn import cross_validation, metrics, ensemble, neighbors, decomposition, preprocessing
from nn_wrapper import nn_wrapper

param = {'booster':'gblinear',
     'max_depth':5,
     'eta':0.1,
     'silent':1,
     'alpha':0.,
     'lambda':0.,
     'objective':'reg:logistic',
     'subsample':0.9,
      'colsample_bytree': 0.9,
     'eval_metric':'auc'
     }

class xgb_wrapper:

    def __init__(self):
        self.clf = xgb.Booster()

    def fit(self, X, y):
        d = xgb.DMatrix(X, y)
        self.clf = xgb.Booster(param, [d])
        for i in range(10):
            self.clf.update(d, i)

    def predict_proba(self, X):
        d = xgb.DMatrix(X)
        preds = self.clf.predict(d).reshape(-1, 1)       
        return np.hstack([preds, preds])


class svr_wrapper:

    def __init__(self):
        self.clf = sklearn.svm.LinearSVR(C=0.2)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_proba(self, X):
        preds = self.clf.predict(X).reshape(-1, 1)       
        return np.hstack([preds, preds])

nn_clf = nn_wrapper()
        
features = [
#           ('21k_1024.npy', sklearn.linear_model.LogisticRegression(C=100)),
#           ('v3_2048.npy', sklearn.linear_model.LogisticRegression(C=100)),
#           ('res_full_l2.npy', sklearn.linear_model.LogisticRegression(C=1)),
           ('21k_v3_3072.npy', sklearn.linear_model.LogisticRegression(C=100)),
            ('21k_v3_128.npy', sklearn.linear_model.LogisticRegression(C=50)),
#            ('21k.npy', sklearn.linear_model.LogisticRegression(C=50)),
           ('fisher.npy', sklearn.linear_model.LogisticRegression(C=2)),
            ('v3_full.npy', sklearn.linear_model.LogisticRegression(C=100)),
            ('21k_full.npy', sklearn.linear_model.LogisticRegression(C=100)),
            ('vlad_2_21k_full.npy', sklearn.linear_model.LogisticRegression(C=1)),
#           ('21k_v3_128.npy', xgb_wrapper()),
#           ('fisher_21k_1024.npy', sklearn.linear_model.LogisticRegression(C=2))
#           ('v3.npy', sklearn.linear_model.LogisticRegression(C=100)),
            
            ('vlad_2_21k_full.npy', xgb.sklearn.XGBClassifier(learning_rate=0.1, n_estimators=100, nthread=8,
                                max_depth=3, subsample=0.8, colsample_bytree=0.8)),
#            ('jo.npy', xgb.sklearn.XGBClassifier(learning_rate=0.1, n_estimators=100, nthread=8,
#                max_depth=4, subsample=0.9, colsample_bytree=0.9))
            ]
#th = np.array([0.4, 0.45, 0.45, 0.4, 0.4, 0.45, 0.5, 0.4, 0.5])
#n_folds = 5
#kf = cross_validation.KFold(2000, n_folds=n_folds, shuffle=True, random_state=0)
#re = np.array([])
#for train_index, test_index in kf:
#
#    preds = np.array([])
#    for feature, clf in features:    
#        x = np.load('train_' + feature)
#        y = np.load('y_train.npy')
##        print('features: ', feature, 'loaded. shape: ', x.shape)
#        X_train, X_val = x[train_index], x[test_index]
#        y_train, y_val = y[train_index], y[test_index]
#        
#        preds_br = np.zeros((X_val.shape[0], 9))
#        for i in range(0, 9):
#            clf.fit(X_train, y_train[:, i])
#            preds_br[:, i] = clf.predict_proba(X_val)[:, 1]        
#
#        nn_preds = np.array([])
#        n_iter = 1
#        for i in range(n_iter):    
#            nn_clf.fit(X_train, y_train)
#            s_preds = nn_clf.predict_proba(X_val)
#            nn_preds = nn_preds + s_preds if nn_preds.size else s_preds
#        nn_preds = (nn_preds / n_iter)
#        
##        n_chains = 2
##        preds_cc = np.zeros((X_val.shape[0], 9))
##        for i in range(n_chains):
##            cc = ClassifierChain(clf)
##            cc.fit(X_train, y_train)
##            preds_cc = preds_cc + cc.predict(X_val)
##        preds_cc = preds_cc / n_chains
##        preds_br = (preds_br + 2*preds_cc) / 3
#
#        preds_br = (preds_br + nn_preds) / 2
#        
#        preds = np.concatenate((preds, preds_br[..., np.newaxis]), axis=2) \
#            if preds.size else preds_br[..., np.newaxis]
#    
#    if len(preds.shape) == 3:
#        preds = preds.mean(axis=2)
#    
#    preds = preds > 0.42
#    
#    score = metrics.f1_score(y_val, preds, average='samples')
#    
#    print('score chain classifier: ', score)
#    re = np.hstack([re, score]) if re.size else score
#
#print('overall:', re.mean(), re.std())

preds = np.array([])
for feature, clf in features:
    x = np.load('train_' + feature)
    y = np.load('y_train.npy')

    x_test = np.load('test_' + feature)

    preds_br = np.zeros((x_test.shape[0], 9))
    for i in range(0, 9):
        clf.fit(x, y[:, i])
        preds_br[:, i] = clf.predict_proba(x_test)[:, 1]        

    nn_preds = np.array([])
    n_iter = 20
    for i in range(n_iter):    
        nn_clf.fit(x, y)
        s_preds = nn_clf.predict_proba(x_test)
        nn_preds = nn_preds + s_preds if nn_preds.size else s_preds
    nn_preds = (nn_preds / n_iter)
    
    n_chains = 20
    preds_cc = np.zeros((x_test.shape[0], 9))
    for i in range(n_chains):
        cc = ClassifierChain(clf)
        cc.fit(x, y)
        preds_cc = preds_cc + cc.predict(x_test)
    preds_cc = preds_cc / n_chains
#    preds_br = (preds_br + 2*preds_cc) / 3

    preds_br = (preds_br + 2*nn_preds + 2*preds_cc) / 5
    
    preds = np.concatenate((preds, preds_br[..., np.newaxis]), axis=2) \
        if preds.size else preds_br[..., np.newaxis]

if len(preds.shape) == 3:
    preds = preds.mean(axis=2)

preds = preds > 0.42

f = open('res', 'w')
print('business_id,labels', file=f)
ids = pd.read_csv('sample_submission.csv').values[:, 0]
for biz_id, pred in zip(ids, preds):
    nz = pred.nonzero()
    nz = [str(x) for x in nz[0]]
    print (biz_id + ',' + ' '.join(nz), file=f)

