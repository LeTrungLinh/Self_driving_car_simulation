import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import socketio
import eventlet
import eventlet.wsgi
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt

from tensorflow.python.keras.backend import set_learning_phase
sio = socketio.Server()

app= Flask(__name__)
maxSpeed = 2

def preprocessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.Canny(img,122,255)
    img = cv2.resize(img,(200,66))
    # img=img/255
    return img

@sio.on('telemetry')
def telemetry(sid,data):
    speed= float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preprocessing(image)
    arr = []
    linerow = image[50,:]
    for x,y in enumerate(linerow):
        if y>200:
            arr.append(x)
    arrmax = max(arr)
    arrmin = min(arr)
    center = int((arrmax+arrmin)/2)
    steering = math.degrees(math.atan((center-image.shape[1]/2)/(image.shape[0]-50)))
    print(steering)
    cv2.circle(image,(arrmin,50),5,(255,255,0),5)
    cv2.circle(image,(arrmax,50),5,(255,255,0),5)
    cv2.line(image,(center,50),(int(image.shape[1]/2),image.shape[0]),(255,255,0),(5))
    throttle = 1.0 -speed/maxSpeed
    print('{} {} {}'.format(steering,throttle,speed))
    print(arr)
    cv2.imshow('',image)
    cv2.waitKey(1)
    
   
    sendcontrol(steering,speed)
    

@sio.on('connect')
def connect(sid,environ):
    print('connected')
    sendcontrol(0,0) # send steering and speed'
    
def sendcontrol(steering,throttle):
    sio.emit('steer',data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    model=load_model('model.h5')
    app= socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)