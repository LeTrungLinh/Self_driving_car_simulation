import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
def get_name(filepath):
    return filepath.split('\\')[-1] #get the name of the picture from the direction
def load_data(path):
    columns=['center','left','right','steering','throttle','brake','speed']
    get_data = pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    #print(get_data.head())
    #print(get_data['center'])
    #print(get_name(get_data['center'][0]))
    get_data['center']=get_data['center'].apply(get_name)
    #print(get_data.head())
    print('total image in data set:',get_data.shape[0])
    return get_data
def draw_data(data,display=True):
    nbins=31
    sampleperbin = 1000
    hist,bins = np.histogram(data['steering'],nbins) #??
    #print(bins.shape)
    if display:
        center=(bins[:-1]+bins[1:])*0.5 #create the point 0 to distinguish negative and positive
        #print(center.shape)
        plt.bar(center,hist,width=0.06) # plot steering data
        plt.plot((-1,1),(sampleperbin,sampleperbin))
        plt.show()
    ##### remove redundant data
    # removeredundantdata=[]
    # for j in range(nbins):
    #     bindatalist=[]
    #     for i in range(len(data['steering'])):
    #         if data['steering'][i]>bins[j] and data['steering'][i]<bins[j+1]:
    #             removeredundantdata.append(i)
    #     bindatalist = shuffle(bindatalist) 
    #     bindatalist = bindatalist[sampleperbin:]
    #     removeredundantdata.extend(bindatalist)
    # print('removed image',len(removeredundantdata))
    # data.drop(data.index[removeredundantdata],inplace = True)
    # print('remaining image :',len(data))
    # hist, _ = np.histogram(data['steering'],nbins)
    # if display:
    #     plt.bar(center,hist,width=0.06) # plot steering data
    #     plt.plot((-1,1),(sampleperbin,sampleperbin))
    #     plt.show()
    return data
def load_data_1(path,data):
    imagepath=[]
    steering=[]
    for i in range(len(data)):
        indexdata=data.iloc[i]
        #print(indexdata)
        imagepath.append(os.path.join(path,'IMG',indexdata[0]))
        #print(os.path.join(path,'IMG',indexdata[0]))
        steering.append(float(indexdata[3]))
    imagepath=np.asarray(imagepath)
    steering =np.asarray(steering)
    return imagepath,steering
def augmentimage(imagepath,steering):
    img=mpimg.imread(imagepath)
    ##pan
    if np.random.rand() < 0.5 :
        pan=iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)
    ##zoom
    if np.random.rand() < 0.5 :
        zoom=iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    ## brightness
    if np.random.rand() < 0.5 :
        brightness=iaa.Multiply((0.4,1.2))
        img=brightness.augment_image(img)
    ## flip
    if np.random.rand() < 0.5 :
        img = cv2.flip(img,1)
        steering=-steering
    return img,steering
# imgRe, st = augmentimage('./dataset/testimg.jpg',0)
# plt.imshow(imgRe)
# plt.show()
def preprocessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img=img/255
    return img
# imgRe = preprocessing(mpimg.imread('./dataset/testimg.jpg'))
# plt.imshow(imgRe)
# plt.show()
def batchgen(imagepath,steeringlist,batchsize,trainflag):
    while True:
        imgbatch= []
        steeringbatch = []
        for i in range(batchsize):
            index = random.randint(0,len(imagepath)-1)
            if trainflag:
                img,steering = augmentimage(imagepath[index],steeringlist[index])
            else : 
                img = mpimg.imread(imagepath[index])
                steering = steeringlist[index]
            img = preprocessing(img)
            imgbatch.append(img)
            steeringbatch.append(steering)
        yield(np.asarray(imgbatch),np.asarray(steeringbatch))
def createmodel():
    model=Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(48,(5,5),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))

    model.add(Flatten())

    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001),loss='mse')

    return model

