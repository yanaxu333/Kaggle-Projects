import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import h5py
import os

import matplotlib.pyplot as plt

ctx =mx.gpu()
ctx1 =[mx.gpu()]

labels = ['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent','Maize','Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet']

from glob import glob

n = len(glob('Segmented/*/*.png'))
X_224 = nd.zeros((n, 3, 224, 224))
X_299 = nd.zeros((n, 3, 299, 299))
y = nd.zeros((n,))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i, file_name in tqdm(enumerate(glob('Segmented/*/*.png')), total=n):
    img = cv2.imread(file_name)
    img_224 = ((cv2.resize(img, (224, 224))[:,:,::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
    img_299 = ((cv2.resize(img, (299, 299))[:,:,::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
    
    X_224[i] = nd.array(img_224)
    X_299[i] = nd.array(img_299)
    
    #print file_name.split('/')[1]
    y[i] = labels.index(file_name.split('/')[1])
    
    nd.waitall()


def get_features(model_name, data_iter):
    net = models.get_model(model_name, pretrained=True, ctx=ctx)
    features = []
    for data in tqdm(data_iter):       
	
        for data_slice in gluon.utils.split_and_load(data, ctx1, even_split=False):
            feature = net.features(data_slice)
            feature = gluon.nn.Flatten()(feature)
            features.append(feature.as_in_context(mx.cpu()))
        nd.waitall()

    features = nd.concat(*features, dim=0)
    return features

batch_size = 64

data_iter_224 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_224), batch_size=batch_size)
data_iter_299 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_299), batch_size=batch_size)																																												

model_names = ['inceptionv3', 'resnet152_v1']

features = []
import pickle as pkl

for model_name in model_names:
    if model_name == 'inceptionv3':
        features.append(get_features(model_name, data_iter_299))
	print("Done inceptionv3")
	data111 = pkl.dumps(features)
	
	
#    else:
#        features.append(get_features(model_name, data_iter_224))
#	print("Done resnet152_v1")
#	data222 = pkl.dumps(features)
	
features = nd.concat(*features, dim=1)
import pickle as pkl
data333 = pkl.dumps(features)
pkl.dump(data333, open('tmp.pickle', 'wb'))

