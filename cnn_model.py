print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
###step 1 : load data
path='dataset'
data= load_data(path)

###step2 : visualize data
data=draw_data(data,display=False)


###step 3
imagepath,steering=load_data_1(path,data)
#print(imagepath[0],steering[0])

###step 4
xtrain,xval,ytrain,yval=train_test_split(imagepath,steering,test_size=0.2,random_state=5)
#print('total training image',len(xtrain))
#print('total validation image',len(xval))

###step5

###step6

###step7

###step8
model=createmodel()
model.summary()

###step9
history=model.fit(batchgen(xtrain,ytrain,100,1),steps_per_epoch=300,epochs=10,
          validation_data =batchgen(xval,yval,100,0),validation_steps=200)

# ###step10
# model.save('model.h5')
# print('Model save')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training','validation'])
# plt.ylim([0,1])
# plt.title('loss')
# plt.xlabel('epoch')
# plt.show()

###test model after train
# model=load_model('model.h5')
# image_test = preprocessing(mpimg.imread('./dataset/testimg.jpg'))
# image_test= np.asarray(image_test)
# image_test = np.expand_dims(image_test, axis=0)
# print(model.predict(image_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')


