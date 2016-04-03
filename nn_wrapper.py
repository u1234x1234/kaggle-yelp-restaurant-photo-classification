import sys
sys.path.append('../mxnet/python')
import numpy as np
from sklearn import metrics, ensemble, neighbors, decomposition, preprocessing
import mxnet as mx
import logging

class nn_wrapper():
    def __init__(self):
        self.pre = preprocessing.StandardScaler()
    
    def get_mlp(self):
        data = mx.symbol.Variable('data')
        label = mx.symbol.Variable('label')
        x  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=2048)
        x = mx.symbol.BatchNorm(data=x)
        x = mx.symbol.LeakyReLU(data=x)
        x = mx.symbol.Dropout(data = x, p=0.5)

        x  = mx.symbol.FullyConnected(data = x, name = 'fc2', num_hidden=1024)
        x = mx.symbol.LeakyReLU(data=x)
        
        x  = mx.symbol.FullyConnected(data = x, name = 'fc3', num_hidden=512)
        x = mx.symbol.LeakyReLU(data=x)


#        x  = mx.symbol.FullyConnected(data = x, name = 'fc33', num_hidden=512)
#        x = mx.symbol.LeakyReLU(data=x)
        
#        x  = mx.symbol.FullyConnected(data = x, name = 'fc33', num_hidden=128)
#        x = mx.symbol.BatchNorm(data=x)
#        x = mx.symbol.LeakyReLU(data=x)    
#        x = mx.symbol.Dropout(data = x, p=0.5)
        
    #    x  = mx.symbol.FullyConnected(data = x, name = 'fc22', num_hidden=256)
    #    x = mx.symbol.BatchNorm(data=x)
    #    x = mx.symbol.LeakyReLU(data=x)
    #    x = mx.symbol.Activation(data = x, act_type="relu")
    #    x = mx.symbol.Dropout(data = x, p=0.5)
    #    x  = mx.symbol.FullyConnected(data = x, name = 'fc3', num_hidden = 200)
    #    x = mx.symbol.Activation(data = x, name='relu3', act_type="relu")
    #    x = mx.symbol.Dropout(data = x, p=0.5)
        x  = mx.symbol.FullyConnected(data = x, name='fc4', num_hidden=9)
#        label  = mx.symbol.SoftmaxOutput(data = x, label=label)

        label = mx.symbol.LinearRegressionOutput(data=x, label=label) 
        return label
    
    def logloss(self, label, pred_prob):
        return metrics.log_loss(label, pred_prob)

    def fit(self, X, y):
#        X = self.pre.fit_transform(X)

        net = self.get_mlp()
        self.model = mx.model.FeedForward(
            ctx                = mx.gpu(),
            symbol             = net,
            num_epoch          = 150,
            learning_rate      = 0.05,
            momentum           = 0.9,
            wd                 = 0.000001
             ,initializer        = mx.init.Xavier(factor_type="in", magnitude=1)
#            ,initializer       = mx.initializer.Normal(sigma=0.01)
#            ,initializer = mx.init.Xavier(factor_type="in", rnd_type="gaussian", magnitude=2.0)  #MSRA
            )
        
        batch_size = 256
        
        train_iter = mx.io.NDArrayIter({'data':X}, {'label':y}, batch_size=batch_size, shuffle=True) 
        
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        self.model.fit(X=train_iter,
                  eval_metric=mx.metric.np(self.logloss),
    #                  eval_data=val_iter,
#                      batch_end_callback = mx.callback.Speedometer(batch_size, 4),
    #                epoch_end_callback = do_checkpoint(),
            logger = logger)
        
    def predict_proba(self, X):
#        X = self.pre.transform(X)
        preds = self.model.predict(X)
        return preds
#        return np.hstack([preds, preds])
    def predict(self, X):
#        X = self.pre.transform(X)
        preds = self.model.predict(X)
        return preds
    
