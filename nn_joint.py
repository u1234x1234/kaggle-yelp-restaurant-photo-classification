import numpy as np
from sklearn import metrics, ensemble, neighbors, decomposition, preprocessing
import mxnet as mx
import logging

class nn_joint():
    def __init__(self):
        self.pre = preprocessing.StandardScaler()
    
    def get_mlp(self):
        data1 = mx.symbol.Variable('data1')
        data2 = mx.symbol.Variable('data2')
        
        label = mx.symbol.Variable('label')

        x1 = mx.symbol.FullyConnected(data = data1, name='fc11', num_hidden=1024)
        x1 = mx.symbol.BatchNorm(data=x1)
        x1 = mx.symbol.LeakyReLU(data=x1)
        x1 = mx.symbol.Dropout(data = x1, p=0.5)

        x1  = mx.symbol.FullyConnected(data = x1, name = 'fc12', num_hidden=512)
        x1 = mx.symbol.LeakyReLU(data=x1)

        x2 = mx.symbol.FullyConnected(data = data2, name='fc21', num_hidden=1024)
        x2 = mx.symbol.BatchNorm(data=x2)
        x2 = mx.symbol.LeakyReLU(data=x2)
        x2 = mx.symbol.Dropout(data = x2, p=0.5)

        x2  = mx.symbol.FullyConnected(data = x2, name = 'fc22', num_hidden=512)
        x2 = mx.symbol.LeakyReLU(data=x2)

        x = mx.symbol.Concat(*[x1, x2])

#        x  = mx.symbol.FullyConnected(data = x, name = 'fcj1', num_hidden=512)
#        x = mx.symbol.LeakyReLU(data=x)        
#        x = mx.symbol.Dropout(data = x, p=0.5)
        
        x  = mx.symbol.FullyConnected(data = x, name = 'fcj2', num_hidden=350)
        x = mx.symbol.Activation(data = x, act_type="relu")
        x = mx.symbol.Dropout(data = x, p=0.5)
        
        x = mx.symbol.FullyConnected(data = x, name='fc4', num_hidden=9)
#        label  = mx.symbol.Softmax(data = x, name='sofmax', multi_output=True)

        label = mx.symbol.LinearRegressionOutput(data=x, label=label) 
        return label
    
    def logloss(self, label, pred_prob):
        return metrics.log_loss(label, pred_prob)

    def fit(self, X1, X2, y):
#        X = self.pre.fit_transform(X)

        net = self.get_mlp()
        self.model = mx.model.FeedForward(
            ctx                = mx.gpu(),
            symbol             = net,
            num_epoch          = 300,
            learning_rate      = 0.04,
            momentum           = 0.9,
            wd                 = 0.000001
             ,initializer        = mx.init.Xavier(factor_type="in", magnitude=1)
#            ,initializer       = mx.init.Orthogonal(scale=1)
#            ,initializer = mx.init.Xavier(factor_type="in", rnd_type="gaussian", magnitude=1.0)  #MSRA
            )
        
        batch_size = 200
        
        train_iter = mx.io.NDArrayIter({'data1':X1, 'data2':X2}, {'label':y}, batch_size=batch_size, shuffle=True) 
        
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        self.model.fit(X=train_iter,
                  eval_metric=mx.metric.np(self.logloss),
    #                  eval_data=val_iter,
#                      batch_end_callback = mx.callback.Speedometer(batch_size, 4),
    #                epoch_end_callback = do_checkpoint(),
            logger = logger)
        
    def predict_proba(self, X1, X2):
#        X = self.pre.transform(X)
        preds = self.model.predict()
        return preds
#        return np.hstack([preds, preds])
    def predict(self, X1, X2):
#        X = self.pre.transform(X)
        train_iter = mx.io.NDArrayIter({'data1':X1, 'data2':X2}) 

        preds = self.model.predict(train_iter)
        return preds
    
