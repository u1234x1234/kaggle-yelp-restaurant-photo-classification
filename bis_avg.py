import numpy as np
import xgboost as xgb
from sklearn import cross_validation, metrics
import misvm
import sklearn
from sklearn import ensemble, neighbors, decomposition
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

train_biz_dict = read_data('train')

y_dict = {}
for row in pd.read_csv('train.csv').values:
    y = np.zeros(9)
    if str(row[1]) != 'nan':
        for label in row[1].split(' '):
            y[label] = 1
    y_dict[str(row[0])] = y

def read_vlad(mode):
    l = 10000
    if mode == 'train':
        l = 2000
    vlad_business = np.genfromtxt('vlad_business_' + mode, dtype='str')
    vlad_feat = np.fromfile('vlad_' + mode, dtype=np.float32).reshape((l, -1))
    if mode == 'train':
        read_vlad.pca.fit(vlad_feat)
    vlad_feat = read_vlad.pca.transform(vlad_feat)
    vlad_business_dict = {idx: feat for idx, feat in  zip(vlad_business, vlad_feat[:])}
    return vlad_business_dict

read_vlad.pca = decomposition.PCA(n_components=128)
#read_vlad.pca = decomposition.NMF(n_components=32)
#vlad_train = read_vlad('train')

y = np.zeros((0, 9))
x = np.array([])
x_test = np.array([])

for key, value in sorted(train_biz_dict.items()):
    avg = np.array(value).sum(axis=0) / len(value)
#    vlad = vlad_train.get(key)
#    feat = np.concatenate([avg, vlad], axis=0)
    feat = avg
    x = np.vstack((x, feat)) if x.size else feat
    y = np.vstack((y, y_dict.get(key)))

print (x.shape, y.shape)

clfs = [sklearn.svm.SVC(C=2., gamma=0.03), 
        sklearn.linear_model.LogisticRegression(C=0.1),
        sklearn.svm.SVC(C=2., gamma=0.03),
        sklearn.svm.SVC(C=1.5, gamma=0.03),
        sklearn.linear_model.LogisticRegression(C=0.1),
        sklearn.svm.SVC(C=1.5, gamma=0.04),
        sklearn.linear_model.LogisticRegression(C=0.1),
        sklearn.svm.SVC(C=2, gamma=0.03),
        sklearn.linear_model.LogisticRegression(C=0.1)]
         
clf1 = sklearn.linear_model.LogisticRegression(C=20)
clf2 = sklearn.svm.LinearSVR(C=10)
#clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
#clf = sklearn.svm.SVC(C=2, gamma=0.03)
#clf = xgb.sklearn.XGBClassifier(learning_rate=0.2, n_estimators=100, nthread=8,
#                                max_depth=5, subsample=0.9, colsample_bytree=0.9)

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

kf = cross_validation.KFold(x.shape[0], n_folds=5, shuffle=True, random_state=0)
res = 0
for train_index, test_index in kf:
    X_train, X_val = x[train_index], x[test_index]
    y_train, y_val = y[train_index], y[test_index]
    rrr = np.zeros((X_val.shape[0], 9), dtype=np.int32)
    for i in range(9):
        clf1.fit(X_train, y_train[:, i])
        preds1 = clf1.predict_proba(X_val)[:, 1]
        clf2.fit(X_train, y_train[:, i])
        preds2 = clf2.predict(X_val)
        preds = (preds1 + preds2) > 0.9
        rrr[:, i] = preds
#        print (i, metrics.f1_score(y_val[:, i], preds))
    score = metrics.f1_score(y_val, rrr, average='samples')
    res += score
    print ('f1: ', score)

print (res / kf.n_folds)
#qwe

test_biz_dict = read_data('test')
#vlad_test = read_vlad('test')

for key, value in sorted(test_biz_dict.items()):
    avg = np.array(value).sum(axis=0) / len(value)    
#    vlad = vlad_test.get(key)
#    feat = np.concatenate([avg, vlad], axis=0)
    feat = avg
    x_test = np.vstack((x_test, feat)) if x_test.size else feat

test_preds = np.zeros((x_test.shape[0], 9), dtype=np.int32)
for i in range(9):
#    clf.fit(x, y[:, i])
    clf1.fit(x, y[:, i])
    preds1 = clf1.predict_proba(x_test)[:, 1]
    clf2.fit(x, y[:, i])
    preds2 = clf2.predict(x_test)
    preds = (preds1 + preds2) > 0.9
    test_preds[:, i] = preds

f = open('res', 'w')
print('business_id,labels', file=f)
for i, (key, val) in enumerate(sorted(test_biz_dict.items())):
    nz = test_preds[i].nonzero()
    nz = [str(x) for x in nz[0]]
    print (key + ',' + ' '.join(nz), file=f)

