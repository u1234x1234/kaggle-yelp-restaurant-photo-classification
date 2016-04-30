import sys
sys.path.append('/home/dima/mxnet/python')
import cv2
import mxnet as mx
import logging
import numpy as np
import time
import os

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

prefix = "Inception"
num_round = 9
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)

synset = [l.strip() for l in open('synset.txt').readlines()]

def PreprocessImageCV(path, show_img=True):
    img = cv2.imread(path)
#    cv2.imshow('orig', img)
#    cv2.waitKey()
    sf = 224 / min(img.shape[:2])
    resized_img = cv2.resize(img, (0, 0), fx=sf, fy=sf)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    max_side = max(resized_img.shape[:2])
    pos = np.array([[0, 0], [max_side - 224, 0], [(max_side - 224) / 2, 0]])
    if img.shape[0] > img.shape[1]:
        pos = np.swapaxes(pos, 0, 1)
    batch = np.zeros((6, 3, 224, 224), np.float32)
    
    for i, (p, f) in enumerate([(p, f) for p in pos for f in [0, 1]]):
        xx = int(p[0])
        yy = int(p[1])

        crop_img = resized_img[yy : yy + 224, xx : xx + 224]
        if f == 1:
            crop_img = cv2.flip(crop_img, 1)
            
#        cv2.imshow('qwe', crop_img)
#        cv2.waitKey()
        sample = crop_img
        sample = np.swapaxes(sample, 0, 2)
        sample = np.swapaxes(sample, 1, 2)
        
        normed_img = sample - 117.
#        normed_img /= 128.
#        print(normed_img)
        
        batch[i] = normed_img
    return batch

batch = PreprocessImageCV('1.jpg', True).reshape((6, 3, 224, 224,))
prob = model.predict(batch)[0]
pred = np.argsort(prob)[::-1]
top1 = synset[pred[0]]
print("Top1: ", top1)
top5 = [synset[pred[i]] for i in range(5)]
print("Top5: ", top5)

internals = model.symbol.get_internals()
fea_symbol = internals["fc1_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=6,
                                         arg_params=model.arg_params, aux_params=model.aux_params,
                                         allow_extra_params=True)
global_pooling_feature = feature_extractor.predict(batch)
print(global_pooling_feature.shape)
#qwe
cn = 0
ts = time.time()

for mode in ['train']:
    root_path = '/mnt/disk/data/'
    lines = [line.rstrip('\n') for line in open(root_path + mode + '_list')]
    filename = mode + '_feat_21k_21k_6crop'    
    try:
        os.remove(root_path + filename)
    except:
        pass
    f = open(root_path + filename, 'ab')
    
    for filename in lines:
        batch = PreprocessImageCV(filename, False)
        prob = feature_extractor.predict(batch).sum(axis=0).ravel()
        prob.tofile(f)
        cn = cn + 1
        if cn % 100 == 0:
            print(cn, time.time() - ts)
        
    print (time.time() - ts)


