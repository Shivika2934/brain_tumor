import cv2
import os
from PIL import Image
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense

image_dir='datasets/'
no_tumor=os.listdir(image_dir+'no/')
yes_tumor=os.listdir(image_dir+'yes/')
#print(no_tumor)
dataset=[]
label=[]
for i,img_name in enumerate(no_tumor):
    if(img_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'no/'+img_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)
for i,img_name in enumerate(yes_tumor):
    if(img_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_dir+'yes/'+img_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)
dataset=np.array(dataset)
label=np.array(label)
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=0)

#print(x_test.shape)
#print(y_test.shape)
#Reshape=(n,image_width,image_height,n_channel)

x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

#model building
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Binary CrossEntropy=1,sigmoid
#categorical Cross Entropy=2,softmax
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=10,validation_data=(x_test,y_test),shuffle=False)
model.save('BrainTumor.h5')