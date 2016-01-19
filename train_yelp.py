import numpy as np
import xgboost as xgb
from sklearn import cross_validation, metrics
import pandas as pd

x = np.fromfile('train_biz_feat', dtype=np.float32).reshape(2000, -1)
x_test = np.fromfile('test_biz_feat', dtype=np.float32).reshape(-1, 2048)
test_biz = np.genfromtxt('test_biz', dtype='str')
print ('test shape: ', x_test.shape)

y = np.zeros((2000, 9), dtype=np.int32)
for i in range(9):
    y[:, i] = np.fromfile('train_y_' + str(i), dtype=np.int32)

X_train, X_val, y_train, y_val = cross_validation.train_test_split(x, y, test_size=0.01, random_state=1)
rrr = np.zeros((X_val.shape[0], 9), dtype=np.int32)
test_preds = np.zeros((x_test.shape[0], 9), dtype=np.int32)

for i in range(9):
    clf = xgb.sklearn.XGBClassifier(learning_rate=0.10, n_estimators=150, nthread=8, reg_alpha=0., reg_lambda=1.,
                                max_depth=6, subsample=0.9, colsample_bytree=0.9)
    clf.fit(X_train, y_train[:, i])   

    preds = clf.predict(X_val)
    rrr[:, i] = preds
    print (i, metrics.f1_score(y_val[:, i], preds))
    test_preds[:, i] = clf.predict(x_test)

#for i in range(400):    
#    tp = np.sum(rrr[i] * y_val[i])
#    fp = np.sum((rrr[i] - y_val[i]) == 1)
#    fn = np.sum((y_val[i] - rrr[i]) == 1)
#    
#    pre = tp / (tp + fp)
#    rec = tp / (tp + fn) 
#    f1 = pre * rec / (pre + rec)
#    if (pre + rec) == 0:
#        print('!!!!!!!!!!!!!!!!!')
#        f1 = 0.
#    if np.isnan(pre):
#        print('!!!!!!!!!!!!!!!!!')
#        f1 = 0.
#    f_mean = f_mean + f1

print ('f1: ', metrics.f1_score(y_val, rrr, average='samples'))

f = open('res', 'w')
print('business_id,labels', file=f)
for i in range(test_preds.shape[0]):
    nz = test_preds[i].nonzero()
    nz = [str(x) for x in nz[0]]
    print (test_biz[i] + ',' + ' '.join(nz), file=f)

