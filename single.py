import numpy as np
import xgboost as xgb
import sklearn
from classifier_chain import ClassifierChain
from sklearn import cross_validation, metrics, ensemble, neighbors, decomposition, preprocessing
from nn_wrapper import nn_wrapper

class xgb_wrapper:

    def __init__(self):
        self.clf = xgb.Booster()

    def fit(self, X, y):
        d = xgb.DMatrix(X, y)
        self.clf = xgb.Booster(param, [d])
        for i in range(50):
            self.clf.update(d, i)

    def predict(self, X):
        d = xgb.DMatrix(X)
        preds = self.clf.predict(d).reshape(-1, 1)       
        return preds.ravel()
#        return np.hstack([preds, preds])



param = {'booster':'gblinear',
     'max_depth':5,
     'eta':0.1,
     'silent':1,
     'alpha':0.,
     'lambda':0.,
     'objective':'reg:logistic',
     'subsample':1.0,
      'colsample_bytree': 1.0,
     'eval_metric':'auc'
     }

features = [
#('v3_256_vlad_8_nc.npy', 0.1),
#('21k_1024.npy', 1),
('21k_50k_2048.npy', 1),
#('v3_2048.npy', 1),
#('21k_color50.npy', 1),
]
def jo(mode, args):
    x = np.hstack([k * np.load(mode + '_' + features) for (features, k) in args])
    return x
    
clf = sklearn.linear_model.LogisticRegression(C=100)
clf = sklearn.linear_model.Ridge(alpha=0.1)
#clf = sklearn.ensemble.BaggingRegressor(clf, n_estimators=8, n_jobs=-1, max_features=0.9, max_samples=0.9, bootstrap=False)

#clf = xgb.sklearn.XGBClassifier(learning_rate=0.2, n_estimators=50, nthread=8,
#                                max_depth=4, subsample=0.8, colsample_bytree=0.8)
#clf = sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
#clf = xgb_wrapper()
clf = nn_wrapper()

x = jo('train', features)   
#x[:, 1024] = np.sqrt(x[:, 1024])
#np.save('train_jo', x)
#x = jo('test', features, features2, features3)
#np.save('test_jo', x)
#qwe
y = np.load('y_train.npy')

print('features: ', features, 'loaded. shape: ', x.shape)

th = np.array([0.4, 0.45, 0.45, 0.4, 0.4, 0.45, 0.5, 0.4, 0.5])
n_folds = 5
kf = cross_validation.KFold(2000, n_folds=n_folds, shuffle=True, random_state=0)
re = np.array([])
for train_index, test_index in kf:
    X_train, X_val = x[train_index], x[test_index]
    y_train, y_val = y[train_index], y[test_index]

#    preds_br = np.zeros((X_val.shape[0], 9))
#    for i in range(0, 9):
#        clf.fit(X_train, y_train[:, i])
#        preds_br[:, i] = clf.predict(X_val)
    
#    multi output
    clf.fit(X_train, y_train)
    preds_br = clf.predict(X_val)

#    nn_preds = np.array([])
#    n_iter = 1
#    for i in range(n_iter):    
#        nn_clf.fit(X_train, y_train)
#        preds = nn_clf.predict_proba(X_val)
#        nn_preds = nn_preds + preds if nn_preds.size else preds
#    nn_preds = (nn_preds / n_iter)
    
#    preds_br = (preds_br + nn_preds)
    preds_br = preds_br > 0.42
 
    score_сс = metrics.f1_score(y_val, preds_br, average='samples')
    re = np.hstack([re, score_сс]) if re.size else score_сс
    print('score chain classifier: ', score_сс)

print('overall:', re.mean(), re.std())

