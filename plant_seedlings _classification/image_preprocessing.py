# This is the code for image prepocessing 
# i.e., getting rid of the background effects
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
from glob import glob


n = len(glob('Nonsegmented/*/*.png')) # this is the directory contains add the raw images


cap = cv2.VideoCapture(0)
for i, file_name in tqdm(enumerate(glob('Nonsegmented/*/*.png')), total=n):
	
	frame = cv2.imread(file_name)
	subfolder = file_name.split('/')[1]
	file_name = file_name.split('/')[2]
	#convert BGR to HSV 
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	#define the range of GREEN color in HSV
	lower_green = np.array([20,100,10])
	upper_green = upper_blue = np.array([75,255,255])
	#only remain green color in a image
	mask = cv2.inRange(hsv, lower_green, upper_green)
	res = cv2.bitwise_and(frame,frame, mask= mask)
	#print file_name
	#save all the images after preprocessing

	cv2.imwrite('/home/lein/project_kg/seed/Segmented/%s/%s'%(subfolder,file_name),res)

