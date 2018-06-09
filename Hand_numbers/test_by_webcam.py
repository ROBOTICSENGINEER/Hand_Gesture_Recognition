import numpy
import cv2

	
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten




# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")


cap = cv2.VideoCapture(1)

ret = cap.set(3,640) # WIDTH
ret = cap.set(4,360) # Height 
ret = cap.set(5,60) # FPS Frame rate


print('Warming the camera')
for k in range(10):
  ret, frame = cap.read()
  cv2.imshow('raw',frame)
  cv2.waitKey(33)


print('Main Loop')
for k in range(1000):
  ret, frame = cap.read()
  im=frame[116:244, 256:384]
  cv2.imshow('raw',im)
  cv2.waitKey(100)
  img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  img = cv2.GaussianBlur(img, (7,7), 3)
  img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
  ret, edge = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  cv2.imshow('Edge',edge)
  cv2.waitKey(100)
  #print(img.shape)
  x=edge.reshape(1,128,128,1)
  #print(x.shape)
  y=model.predict(x)
  #print(y)
  z=numpy.argmax(y.reshape(6))
  print(z)
cap.release()
cv2.destroyAllWindows()