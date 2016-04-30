# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn import manifold, decomposition
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde

def imscatter(x, y, image, ax=None, label=False):
    label=label==True
    im = OffsetImage(image)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=label)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

n = 2000
X = np.fromfile('train_feat', dtype=np.float32, count=n*2048).reshape(-1, 2048)
print (X.shape)
X = decomposition.PCA(n_components=50).fit_transform(X)
tsne = manifold.TSNE()
X = tsne.fit_transform(X)

fig, ax = plt.subplots()
scale_factor=8
fig.set_size_inches(16*scale_factor, 9*scale_factor, forward=True)

lines = [line.rstrip('\n') for line in open('train_list')]
for i, filename in enumerate(lines):
    print (filename)
    image = cv2.imread(filename)
    sf = 100/max(image.shape[0], image.shape[1])
    image = cv2.resize(image, (0,0), fx=sf, fy=sf)
    cv2.imshow('12', image)
#    cv2.waitKey()
    if i == n:
        break
    b,g,r = cv2.split(image)       # get b,g,r
    image = cv2.merge([r,g,b])     # switch it to rgb
    x1=X[i, 0]
    x2=X[i, 1]
    imscatter(x1, x2, image, ax)
    ax.plot(x1, x2)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.grid()
fig.savefig('7000.png', dpi=100)
#plt.show()
 



