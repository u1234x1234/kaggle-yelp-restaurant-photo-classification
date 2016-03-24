import numpy as np
import xgboost as xgb
import sklearn
from classifier_chain import ClassifierChain
from sklearn import cross_validation, metrics, ensemble, neighbors, decomposition, preprocessing

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
clf3vlad = xgb.sklearn.XGBClassifier(learning_rate=0.1, n_estimators=200, nthread=8,
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
          
#features = 'train_fisher.npy'
#features = 'train_21k_l2.npy'
features = 'train_v3_full.npy'
#features = 'train_vlad_32_16.npy'

clf = sklearn.linear_model.LogisticRegression(C=100)
#clf = sklearn.ensemble.RandomForestClassifier()
x = np.load(features)
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
    for i in range(0, 9):
        clf.fit(X_train, y_train[:, i])
        preds_br[:, i] = clf.predict_proba(X_val)[:, 1]        
    preds_br = preds_br > 0.42
 
    score_сс = metrics.f1_score(y_val, preds_br, average='samples')
    re = np.hstack([re, score_сс]) if re.size else score_сс
    print('score chain classifier: ', score_сс)

print('overall:', re.mean(), re.std())

