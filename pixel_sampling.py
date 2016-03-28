import numpy as np
import cv2
from sklearn import cluster 
from sklearn import neighbors
import os

def learn_color_clusters():
    samples = np.zeros((0, 3))
    cnt = 0
    with open('train_list') as f:
        for line in f:
            line = line[:-1]
            image = cv2.imread(line)
            image = cv2.resize(image, (100, 100))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            
            points = image.reshape((-1, 3))
            np.random.permutation(points.shape[0])
            samples = np.vstack([samples, points[:50]])
            
            print(samples.shape)
            cnt = cnt + 1
            if cnt % 10000 == 0:
                break
    
    km = cluster.KMeans(n_clusters=50, n_jobs=-1)
    km.fit(samples)
    np.save('lab_clusters.npy', km.cluster_centers_)
    return
    
#learn_color_clusters()

def extract_lab_histogram(mode, clusters):
    
    nn = neighbors.NearestNeighbors(n_neighbors=1)
    nn.fit(clusters)
    out_filename = mode + '_color'    
    try:
        os.remove(out_filename)
    except:
        pass
    out = open(out_filename, 'ab')
    cnt = 0    
    with open(mode + '_list') as f:
        for line in f:
            line = line[:-1]
            image = cv2.imread(line)
            image = cv2.resize(image, (100, 100))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            points = image.reshape((-1, 3))
            cn = nn.kneighbors(points)
            hist = np.histogram(cn[1], bins=50, range=(1, 50))[0]
            hist.tofile(out)            
            cnt = cnt + 1
            if cnt % 1000 == 0:
                print(cnt)

clusters = np.load('lab_clusters.npy') 
extract_lab_histogram('train', clusters)   
extract_lab_histogram('test', clusters)
