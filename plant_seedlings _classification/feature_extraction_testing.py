# extracting all image features from inceptionv3
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
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

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

df_test = pd.read_csv('sample_submission.csv')
n_test = len(df_test)
	

X_224_test = nd.zeros((n_test, 3, 224, 224))
X_299_test = nd.zeros((n_test, 3, 299, 299))


for i, fname in tqdm(enumerate(df_test['file']), total=n_test):
    
    img = cv2.imread('test/%s' % fname)
    
    img_224 = ((cv2.resize(img, (224, 224))[:,:,::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
    img_299 = ((cv2.resize(img, (299, 299))[:,:,::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
    
    X_224_test[i] = nd.array(img_224)
    X_299_test[i] = nd.array(img_299)
    
    nd.waitall()
nd.save('test.nd', [X_224_test, X_299_test])
X_224_test, X_299_test = nd.load('test.nd')

batch_size = 128



data_test_iter_224 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_224_test), 
                                           batch_size=batch_size)
data_test_iter_299 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_299_test), 
                                           batch_size=batch_size)


def save_features(model_name,  data_test_iter, ignore=False):
    
    
    net = models.get_model(model_name, pretrained=True, ctx=ctx)
    
    for prefix, data_iter in zip(['test'], [ data_test_iter]):
        features = []
        for data in tqdm(data_iter):
           
            for data_slice in gluon.utils.split_and_load(data, ctx1, even_split=False):
                feature = net.features(data_slice)       
                feature = gluon.nn.Flatten()(feature)
                features.append(feature.as_in_context(mx.cpu()))
            nd.waitall()
        
        features = nd.concat(*features, dim=0)
        nd.save('features_%s_%s.nd' % (prefix, model_name), features)
		


model_names = ['inceptionv3']#, 'resnet152_v1']

features = []
for model in model_names:
    print model
    if model == 'inceptionv3':
        save_features(model, data_test_iter_299)
 #   else:
 #       save_features(model, data_test_iter_224) 																																																																									


	
