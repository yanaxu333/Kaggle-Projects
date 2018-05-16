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

y = nd.zeros((n,))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


for i, file_name in tqdm(enumerate(glob('Segmented/*/*.png')), total=n):

    y[i] = labels.index(file_name.split('/')[1])
   
    
    nd.waitall()

batch_size = 64
#load features from file
import pickle as pkl

data1 = pkl.load(open('tmp.pickle','rb'))
b1 = pkl.loads(data1)


c1 = b1[1]
d1 = c1.asnumpy()
Q = d1.reshape(1,-1)

for i in tqdm(range(1,5544)):
    c1 = b1[i]
    d1 = c1.asnumpy()
    d1 = d1.reshape(1,-1)
    Q = np.concatenate((Q, d1), axis=0)

features = Q
																																																																																																																																				

data_iter_train = gluon.data.DataLoader(gluon.data.ArrayDataset(features, y), batch_size, shuffle=True)
model_names = ['inceptionv3']

def build_model():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.BatchNorm())
        net.add(nn.Dense(1024))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(12))

    net.initialize(ctx=ctx)
    return net

ctx = mx.gpu()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()

def evaluate(net, data_iter):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss/steps, acc/steps

net = build_model()

epochs = 100
batch_size = 64
lr_sch = mx.lr_scheduler.FactorScheduler(step=1500, factor=0.5)
trainer = gluon.Trainer(net.collect_params(), 'adam', 
                        {'learning_rate': 1e-3, 'lr_scheduler': lr_sch})

for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    steps = len(data_iter_train)
    for data, label in data_iter_train:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    print("Epoch %d. loss: %.4f, acc: %.2f%%" % (epoch+1, train_loss/steps, train_acc/steps*100))

evaluate(net, data_iter_train)
features_test = [nd.load('features_test_%s.nd' % model_name)[0] for model_name in model_names]
features_test = nd.concat(*features_test, dim=1)
output = nd.softmax(net(features_test.as_in_context(ctx))).asnumpy()
df_pred = pd.read_csv('sample_submission.csv')
print output.argmax(axis=1)


evaluate(net, data_iter_train)
features_test = [nd.load('features_test_%s.nd' % model_name)[0] for model_name in model_names]
features_test = nd.concat(*features_test, dim=1)

output = nd.softmax(net(features_test.as_in_context(ctx))).asnumpy()
df_pred = pd.read_csv('sample_submission.csv')

eee = output.argmax(axis =1)
print eee

print eee.shape
print"----------"
print output.shape[0]
for i in tqdm(range(output.shape[0])):
    templa = output[i].argmax()
    
    df_pred.iloc[:,1][i] = labels[templa]
print 'i=%d'%i
df_pred.to_csv('pred1.csv', index=None)
print features_test.shape	

