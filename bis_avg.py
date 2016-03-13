import numpy as np
import xgboost as xgb
from sklearn import cross_validation, metrics
import sklearn
from sklearn import ensemble, neighbors, decomposition, preprocessing
import pandas as pd
from sklearn import decomposition
from collections import defaultdict
import os

def read_data(mode):
    l = 237152
    if mode == 'train':
        l = 234842
    print ('read data ' + mode)
    x = np.fromfile(mode + '_feat', dtype=np.float32).reshape(l, -1)
    id_list = [line.rstrip('\n')[line.rfind('/') + 1:-4] for line in open(mode + '_list')]    
    photo2x = {photo: x_i for photo, x_i in zip(id_list, x[:])}
    df = pd.read_csv(mode + '_photo_to_biz.csv', dtype={0: str, 1:str})
    biz_dict = defaultdict(list)    
    for index, row in df.iterrows():
        biz_dict[row[1]].append(photo2x.get(row[0]))
        if index % 50000 == 0:
            print (index)
    return biz_dict

def read_y():
    y_dict = {}
    for row in pd.read_csv('train.csv').values:
        y = np.zeros(9)
        if str(row[1]) != 'nan':
            for label in row[1].split(' '):
                y[label] = 1
        y_dict[str(row[0])] = y
    return y_dict

def read_vlad(mode):
    l = 10000
    if mode == 'train':
        l = 2000
    vlad_business = np.genfromtxt('vlad_business_' + mode, dtype='str')
    vlad_feat = np.fromfile('vlad_' + mode, dtype=np.float32).reshape((l, -1))
    
#    if mode == 'train':
#        read_vlad.pca.fit(vlad_feat)
    vlad_feat = preprocessing.normalize(vlad_feat)
#    vlad_feat = read_vlad.pca.transform(vlad_feat)
    vlad_business_dict = {idx: feat for idx, feat in  zip(vlad_business, vlad_feat[:])}
    return vlad_business_dict

def pool(biz_dict, vlad_dict, mode):
    if mode == 'train':
        y_dict = read_y()
    y = np.zeros((0, 9))
    x = np.array([])
    x_vlad = np.array([])
    
    for key, value in sorted(biz_dict.items()):
        avg = np.array(value).sum(axis=0) / len(value)
        vlad = vlad_dict.get(key)
#        vlad = preprocessing.normalize(vlad)
#        print(vlad.shape)
#        feat = np.concatenate([avg, vlad], axis=0)
#        feat = preprocessing.Normalizer().fit_transform(feat)
#        feat = avg
        x = np.vstack((x, avg)) if x.size else avg
        x_vlad = np.vstack((x_vlad, vlad)) if x_vlad.size else vlad
        
        if mode == 'train':
            y = np.vstack((y, y_dict.get(key)))        
    return (x, x_vlad, y) if mode == 'train' else (x, x_vlad)

train_biz_dict = read_data('train')
read_vlad.pca = decomposition.PCA(n_components=1024)
vlad_train = read_vlad('train')
#vlad_train = []
x, x_vlad, y = pool(train_biz_dict, vlad_train, 'train')
#np.save('xbin.npy', x)
#np.save('ybin.npy', y)
#np.save('xvladbin.npy', x_vlad)

#x = np.load('xbin.npy') 
#y = np.load('ybin.npy')
#x_vlad = np.load('xvladbin.npy')

print (x.shape, x_vlad.shape, y.shape)

#for i in range(9):
#    q, _ = np.histogram(y[:, i].ravel(), bins=[0, 0.5, 1])
#    print(i + 1, q, q[0] / q[1])

clf1 = sklearn.linear_model.LogisticRegression(C=200)
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

#kf = cross_validation.KFold(x.shape[0], n_folds=5, shuffle=True, random_state=0)
#res = 0
#for i in range(9):
#    res = 0
#    for train_index, test_index in kf:
#        X_train, X_val = x[train_index], x[test_index]
#        y_train, y_val = y[train_index], y[test_index]
#        rrr = np.zeros((X_val.shape[0], 9), dtype=np.int32)
#
#        clf.fit(X_train, y_train[:, i])
#        preds = clf.predict(X_val)
#        rrr[:, i] = preds
##        print (i, metrics.f1_score(y_val[:, i], preds))
##    score = metrics.f1_score(y_val, rrr, average='samples')
#        res += metrics.f1_score(y_val[:, i], preds)
#
#    print (i, res / kf.n_folds)
#
#print (res / kf.n_folds)

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
th = np.array([0.4, 0.45, 0.45, 0.4, 0.4, 0.45, 0.5, 0.4, 0.5])
res = 0
n_folds = 5
#for i in range(0, 9):
#    yi_score = np.zeros((0, 1))
#    kf = cross_validation.StratifiedKFold(y[:, i], n_folds=n_folds, shuffle=True, random_state=0)    
#    for train_index, test_index in kf:        
#        X_train, X_val = x[train_index], x[test_index]
#        X_vlad_train, X_vlad_val = x_vlad[train_index], x_vlad[test_index]
#        y_train, y_val = y[train_index], y[test_index]
##        rrr = np.zeros((X_val.shape[0], 9), dtype=np.int32)    
#
#        clf1.fit(X_train, y_train[:, i])
#        preds1 = clf1.predict_proba(X_val)[:, 1]
#        clf1vlad.fit(X_vlad_train, y_train[:, i])
#        preds1vlad = clf1vlad.predict_proba(X_vlad_val)[:, 1]
#
##        clf2.fit(X_train, y_train[:, i])
##        preds2 = clf2.predict(X_val)
##        clf2vlad.fit(X_vlad_train, y_train[:, i])
##        preds2vlad = clf2vlad.predict(X_vlad_val)
#        
#        clf3.fit(X_train, y_train[:, i])
#        preds3 = clf3.predict_proba(X_val)[:, 1]
#        clf3vlad.fit(X_vlad_train, y_train[:, i])
#        preds3vlad = clf3vlad.predict_proba(X_vlad_val)[:, 1]
#        
##        dtrain = xgb.DMatrix(X_train, y_train[:, i])
##        dval = xgb.DMatrix(X_val, y_val[:, i])
##        bst = xgb.Booster(param, [dtrain, dval])
##        for it in range(30):
##            bst.update(dtrain, it)
##        preds4 = np.array(bst.predict(dval))
#        preds = (preds1 + preds1vlad + preds3 + preds3vlad) > 0.42 * 4
##        preds = clf1.fit(X_train, y_train[:, i]).predict_proba(X_val)[:, 1] > 0.4
##        rrr[:, i] = preds
#        score = metrics.f1_score(y_val[:, i], preds, average='binary')
#        yi_score = np.vstack((yi_score, score))
#    print (i, yi_score.mean())
#    res += yi_score.sum()
#print (res / (n_folds * 9))
#qwe

test_biz_dict = read_data('test')
vlad_test = read_vlad('test')
#vlad_test = []
X_train = x
X_vlad_train = x_vlad
y_train = y

X_val, X_vlad_val = pool(test_biz_dict, vlad_test, 'test')

test_preds = np.zeros((X_val.shape[0], 9), dtype=np.int32)
for i in range(9):

    clf1.fit(X_train, y_train[:, i])
    preds1 = clf1.predict_proba(X_val)[:, 1]
    clf1vlad.fit(X_vlad_train, y_train[:, i])
    preds1vlad = clf1vlad.predict_proba(X_vlad_val)[:, 1]

#        clf2.fit(X_train, y_train[:, i])
#        preds2 = clf2.predict(X_val)
#        clf2vlad.fit(X_vlad_train, y_train[:, i])
#        preds2vlad = clf2vlad.predict(X_vlad_val)
    
    clf3.fit(X_train, y_train[:, i])
    preds3 = clf3.predict_proba(X_val)[:, 1]
    clf3vlad.fit(X_vlad_train, y_train[:, i])
    preds3vlad = clf3vlad.predict_proba(X_vlad_val)[:, 1]
    preds = (preds1 + preds1vlad + preds3 + preds3vlad) > 0.42 * 4

    test_preds[:, i] = preds
    print(i)

f = open('res', 'w')
print('business_id,labels', file=f)
for i, (key, val) in enumerate(sorted(test_biz_dict.items())):
    nz = test_preds[i].nonzero()
    nz = [str(x) for x in nz[0]]
    print (key + ',' + ' '.join(nz), file=f)

