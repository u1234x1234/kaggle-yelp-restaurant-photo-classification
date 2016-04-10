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

def train_predict(fold, feature, clf, X_train, y_train, X_test, y_test):
    preds_br = np.zeros((X_test.shape[0], 9))
    for i in range(0, 9):
        clf.fit(X_train, y_train[:, i])
        preds_br[:, i] = clf.predict_proba(X_test)[:, 1]
    np.save('val3/' + str(fold) + '_' + feature + '_br', preds_br)

    nn_preds = np.array([])
    n_iter = 10
    for i in range(n_iter):
        nn_clf.fit(X_train, y_train)
        s_preds = nn_clf.predict_proba(X_test)
        nn_preds = nn_preds + s_preds if nn_preds.size else s_preds
    nn_preds = (nn_preds / n_iter)
    np.save('val3/' + str(fold) + '_' + feature + '_nn', nn_preds)
    
    n_chains = 10
    preds_cc = np.zeros((X_test.shape[0], 9))
    for i in range(n_chains):
        cc = ClassifierChain(clf)
        cc.fit(X_train, y_train)
        preds_cc = preds_cc + cc.predict(X_test)
    preds_cc = preds_cc / n_chains
    np.save('val3/' + str(fold) + '_' + feature + '_cc', preds_cc)
    
    np.save('val3/' + str(fold) + '_' + feature + '_test', y_test)
        
    preds_br = (preds_br + 3*nn_preds + 2*preds_cc) / 6
    
    return preds_br

kf = cross_validation.KFold(2000, n_folds=5, shuffle=True, random_state=0)
re = np.array([])
fold = 0
for train_index, test_index in kf:

   y = np.load('y_train.npy')
   y_val = y[test_index]
   preds = np.array([])
   for feature, clf in features:    
       x = np.load('train_' + feature)

       X_train, X_val = x[train_index], x[test_index]
       y_train = y[train_index]
       
       preds_br = train_predict(fold, feature, clf, X_train, y_train, X_val, y_val)
       
       preds = np.concatenate((preds, preds_br[..., np.newaxis]), axis=2) \
           if preds.size else preds_br[..., np.newaxis]
   
   if len(preds.shape) == 3:
       preds = preds.mean(axis=2)
   
   preds = preds > 0.44
   
   score = metrics.f1_score(y_val, preds, average='samples')
   print('score chain classifier: ', score)
   re = np.hstack([re, score]) if re.size else score
   fold = fold + 1

print('overall:', re.mean(), re.std())
qwe

preds = np.array([])
for feature, clf in features:
    x = np.load('train_' + feature)
    y = np.load('y_train.npy')

    x_test = np.load('test_' + feature)

    preds_br = train_predict(clf, x, y, x_test)
    
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

