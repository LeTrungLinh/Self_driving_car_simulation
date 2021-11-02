# Self_driving_car_simulation
# Overview
This project is simulation self driving car by unity. Using CNN to predict steering angle.
## Software
Install simulation software according to this link :
[software](https://github.com/udacity/self-driving-car-sim)
## Enviroment
we need have anaconda to use enviroment
```html
 #Use TensorFlow without GPU**
 conda env create -f environments.yml 
 #Use TensorFlow with GPU**
 conda env create -f environment-gpu.yml
```
## NOTE
- cnn_model.py : trainning file 
- test.py: control and send speed, steering to software
- utlis.py: preprocessing image 
## Result
[![Result](https://img.youtube.com/vi/EBj5x94XJZk/0.jpg)](https://www.youtube.com/watch?v=EBj5x94XJZk)
