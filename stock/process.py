import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time

import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter 
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential 
from keras.layers.core import Dense,Dropout, Activation
from keras.layers.recurrent import LSTM 
from keras.models import load_model
import keras 
import h5py
import os

#read data and transform them to pandas dataframe
df = pd.read_csv('./input/prices-split-adjusted.csv',
index_col = 0)
df['adj close'] =df.close # create a new column called 
#adj close and copying all the elements into the last column

df.drop(['close'], axis=1, inplace=True)

print df.head()

df2 = pd.read_csv('./input/fundamentals.csv')
print df2.head()
#extract all symbols from csv file 
symbols = list(set(df.symbol))
print len(symbols)
print symbols[:11]

#extract a particular privr for stock in symbols 
print df.shape
df = df[df.symbol == 'GOOG']
print df.shape


df.drop(['symbol'],axis = 1,inplace = True)

print df.head()


#normalize the data 
def normalize_data(df):
	min_max_scaler = preprocessing.MinMaxScaler()
	df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
	df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
	df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
	df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
	df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1,1))
	return df 

df = normalize_data(df)

print df.head()

# create training set and testing set
def load_data(stock, seq_len):
	amount_of_features = len(stock.columns)
	data = stock.as_matrix()
	sequence_length = seq_len + 1 # index starting from 0
	result = []
	for index in range(len(data) - sequence_length):
		result.append(data[index: index+sequence_length])
		#index : index+22 days 

	result = np.array(result)
	row =round(0.9*result.shape[0]) #90%split
	train = result[:int(row),:] #90%date all features

	x_train = train[:,:-1]
	y_train = train[:, -1][:,-1]

	x_test = result[int(row):,:-1]
	y_test = result[int(row):,-1][:,-1]
	x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], amount_of_features))

	x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], amount_of_features))
	return [x_train, y_train, x_test,y_test]

#build the structure of model

def build_model(layers):
    d = 0.3
    model = Sequential()
    
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    
    # adam = keras.optimizers.Adam(decay=0.2)
        
    start = time.time()
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

window = 22
x_train, y_train, x_test, y_test = load_data(df, window)
print (x_train[0], y_train[0])



model = build_model([5,window,1])

model.fit(X_train,y_train,batch_size=512,epochs=90,validation_split=0.1,verbose=1)



