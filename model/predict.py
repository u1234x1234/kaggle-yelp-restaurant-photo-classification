import mxnet as mx
import logging
import numpy as np
import time
from skimage import io, transform
import os

def PreprocessImage(path, show_img=True):
    # load image
    img = io.imread(path)
#    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 299, 299
    resized_img = transform.resize(crop_img, (299, 299))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (299, 299, 3) to (3, 299, 299)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - 128.
    normed_img /= 128.

    return np.reshape(normed_img, (1, 3, 299, 299))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG) 

prefix = "Inception-7"
num_round = 1
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=1)
synset = [l.strip() for l in open('synset.txt').readlines()]


internals = model.symbol.get_internals()
fea_symbol = internals["global_pool_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,
                                         arg_params=model.arg_params, aux_params=model.aux_params,
                                         allow_extra_params=True)

lines = [line.rstrip('\n') for line in open('test_list')]
cn = 0
ts = time.time()

os.remove('test_feat')
f = open('test_feat', 'ab')

for filename in lines:
    batch = PreprocessImage(filename, False)
#    prob = model.predict(batch)[0]
    prob = feature_extractor.predict(batch)[0]
#    print (prob.shape)
#    pred = np.argsort(prob)[::-1]
#    top1 = synset[pred[0]]
#    print("Top1: ", top1)
#    top5 = [synset[pred[i]] for i in range(5)]
#    print("Top5: ", top5)
    prob.tofile(f)
    cn = cn + 1
    if cn % 1000 == 0:
        print(cn, time.time() - ts)
    
print (time.time() - ts)
