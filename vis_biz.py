# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import sys
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

size = 70

def read_images(image_list):
    iml = list()
    for image in image_list:
        path = 'data/train_photos/' + image + '.jpg'
        image = cv2.imread(path)
        image = cv2.resize(image, (size, size))
        iml.append(image)
    return iml
    
def pan(iml):
    w = 10
    h = 10
    image_pan = np.zeros((h * size, w * size, 3), dtype=np.uint8)
    for i, image in enumerate(iml):
        x = (i % w) * size
        y = (i // w) * size
        image_pan[y : y + size, x : x + size] = image
    return image_pan

d = pd.read_csv('train_photo_to_biz.csv', dtype={0: str, 1:str})

train2biz = defaultdict(list)
for (x, y) in d.values:
    train2biz[y].append(x)

if len(sys.argv) == 2:
    image_list = train2biz.get(sys.argv[1])
    pn = pan(read_images(image_list))
    print(image_list[21])
    cv2.imshow('pn', pn)
    for image in image_list:
        path = 'data/train_photos/' + image + '.jpg'
        image = cv2.imread(path)
        cv2.imshow('im', image)
        cv2.waitKey()
    sys.exit(0)
    

for row in pd.read_csv('train.csv').values:
    image_list = train2biz.get(str(row[0]))
    if len(image_list) >= 100:
        continue
    images = read_images(image_list)
    print(row[0], row[1])
    pn = pan(images)
    cv2.destroyAllWindows()
    cv2.imshow(str(row[1]), pn)
    cv2.waitKey()