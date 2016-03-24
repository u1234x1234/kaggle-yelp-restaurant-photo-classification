# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict

def read_data(mode):
    # read dict biz -> (x1, x2)
    l = 234842
    if mode == 'test':
        l = 237152
    photo_features = np.fromfile(mode + '_feat', np.float32).reshape((l, -1))
    id_list = [line.rstrip('\n')[line.rfind('/') + 1:-4] for line in open(mode + '_list')]
    df = pd.read_csv(mode + '_photo_to_biz.csv', dtype={0: str, 1:str})
    biz_dict = defaultdict(list)
    photo_to_biz = defaultdict(list)
    for (x, y) in df.values:
        photo_to_biz[x].append(y)
#    photo_to_biz = {str(x): str(y) for (x, y) in df.values}
    for (image_id, feature) in zip(id_list, photo_features):
        biz_ids = photo_to_biz.get(image_id)
        for biz_id in biz_ids:
            biz_dict[biz_id].append(feature)

    return biz_dict

def gen_features(biz_dict, filename):
    # from dict to biz -> pool(x)
    df = pd.read_csv(filename)    
    y = np.zeros((len(df), 9))
    x = np.array([])    
    for (i, row) in enumerate(df.values):
        biz_id = str(row[0])
        if str(row[1]) != 'nan':
            for label in row[1].split(' '):
                y[i, label] = 1
        features = np.array(biz_dict.get(biz_id))
        biz_x = features.mean(axis=0)
        x = np.vstack((x, biz_x)) if x.size else biz_x
    return x, y

def read_vlad(filename, mode):
    l = 10000
    if mode == 'train':
        l = 2000
    vlad_business = np.genfromtxt('vlad_business_' + mode, dtype='str')
    vlad_feat = np.fromfile(filename + '_' + mode, dtype=np.float32).reshape((l, -1))
    vlad_business_dict = {idx: feat.reshape(1, -1) for idx, feat in  zip(vlad_business, vlad_feat[:])}
    return vlad_business_dict

for mode, filename in [('test', 'sample_submission.csv'), ('train', 'train.csv')]:

#    biz_dict = read_data(mode)
#    x, y = gen_features(biz_dict,  filename)
#    np.save(mode + '_21k_full', x)
    
    vlad_biz_train = read_vlad('vlad', mode)
    x, y = gen_features(vlad_biz_train, filename)
    np.save(mode + '_vlad_2_21k_full', x)
    
    if mode == 'train':
        np.save('y_train', y)
    
    