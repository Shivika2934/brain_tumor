import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor.h5')
image=cv2.imread('pred\\pred15.jpg')
img=Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
#print(img)
input_img=np.expand_dims(img,axis=0)
#prediction=model.predict(input_img)
result=model.predict(input_img)
print(result)