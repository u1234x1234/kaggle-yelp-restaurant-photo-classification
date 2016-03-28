import numpy as np
import xgboost as xgb
import sklearn
from classifier_chain import ClassifierChain
from sklearn import cross_validation, metrics, ensemble, neighbors, decomposition, preprocessing
from nn_wrapper import nn_wrapper

clf1 = sklearn.linear_model.LogisticRegression(C=5)
clf1vlad = sklearn.linear_model.LogisticRegression(C=1)

clf2 = sklearn.svm.LinearSVR(C=5)
#clf2vlad = sklearn.svm.LinearSVR(C=1)

#clf2 = sklearn.svm.SVR(C=0.1, kernel='linear')
#clf1 = sklearn.linear_model.LogisticRegressionCV(Cs=100)
#clf1 = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
#clf1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=50)
#clf1 = sklearn.svm.SVC(C=10, gamma=0.03, kernel='linear', probability=True)
clf3 = xgb.sklearn.XGBClassifier(learning_rate=0.1, n_estimators=200, nthread=8,
                                max_depth=5, subsample=0.9, colsample_bytree=0.9)

param = {'booster':'gblinear',
     'max_depth':5,
     'eta':0.1,
     'silent':1,
     'alpha':0.,
     'lambda':0.,
     'objective':'reg:logistic',
     'subsample':0.8,
      'colsample_bytree': 0.8,
     'eval_metric':'auc'
     }

features = [
#('v3_256_vlad_8_nc.npy', 0.1),
('res_full_l2.npy', 1),
#('v3_2048.npy', 1),
#('21k_1024.npy', 1),
#('21k_color50.npy', 1),
]
def jo(mode, args):
    x = np.hstack([k * np.load(mode + '_' + features) for (features, k) in args])
    return x
    
clf = sklearn.linear_model.LogisticRegression(C=100)
#clf = xgb.sklearn.XGBClassifier(learning_rate=0.1, n_estimators=200, nthread=8,
#                                max_depth=5, subsample=0.8, colsample_bytree=0.9)
#clf = sklearn.ensemble.RandomForestClassifier(n_jobs=-1)
#clf = nn_wrapper()
nn_clf = nn_wrapper()

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

    preds_br = np.zeros((X_val.shape[0], 9))
#    for i in range(0, 9):
#        clf.fit(X_train, y_train[:, i])
#        preds_br[:, i] = clf.predict_proba(X_val)[:, 1]        

    nn_preds = np.array([])
    n_iter = 1
    for i in range(n_iter):    
        nn_clf.fit(X_train, y_train)
        preds = nn_clf.predict_proba(X_val)
        nn_preds = nn_preds + preds if nn_preds.size else preds
    nn_preds = (nn_preds / n_iter)
    
    preds_br = (preds_br + nn_preds)
    preds_br = preds_br > 0.42
 
    score_сс = metrics.f1_score(y_val, preds_br, average='samples')
    re = np.hstack([re, score_сс]) if re.size else score_сс
    print('score chain classifier: ', score_сс)

print('overall:', re.mean(), re.std())

