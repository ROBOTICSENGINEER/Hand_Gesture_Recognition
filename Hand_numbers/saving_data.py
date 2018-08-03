#import tensorflow
import numpy
import cv2
import os

current_dir = os.getcwd()

STATES = ['ZERO' , 'ONE' , 'TWO' , 'THREE' , 'FOUR' , 'FIVE']


print('reading 0 from train set')
names = os.listdir('./images/train/ZERO')
N=len(names)
x0 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/train/ZERO/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  x0[k,:]=pic2
y0=numpy.zeros((N, 1), dtype=numpy.int)


print('reading 1 from train set')
names = os.listdir('./images/train/ONE')
N=len(names)
x1 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/train/ONE/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  x1[k,:]=pic2
y1=numpy.ones((N, 1), dtype=numpy.int)


print('reading 2 from train set')
names = os.listdir('./images/train/TWO')
N=len(names)
x2 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/train/TWO/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  x2[k,:]=pic2
y2=numpy.ones((N, 1), dtype=numpy.int)*2



print('reading 3 from train set')
names = os.listdir('./images/train/THREE')
N=len(names)
x3 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/train/THREE/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  x3[k,:]=pic2
y3=numpy.ones((N, 1), dtype=numpy.int)*3



print('reading 4 from train set')
names = os.listdir('./images/train/FOUR')
N=len(names)
x4 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/train/FOUR/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  x4[k,:]=pic2
y4=numpy.ones((N, 1), dtype=numpy.int)*4



print('reading 5 from train set')
names = os.listdir('./images/train/FIVE')
N=len(names)
x5 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/train/FIVE/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  x5[k,:]=pic2
y5=numpy.ones((N, 1), dtype=numpy.int)*5


print('concatenate train set')
X_train = numpy.concatenate((x0,x1,x2,x3,x4,x5), axis=0)
Y_train = numpy.concatenate((y0,y1,y2,y3,y4,y5), axis=0)

print('saving train set as binary')
numpy.save('train_data.npy', X_train)
numpy.save('train_label.npy', Y_train)



print('reading 0 from test set')
names = os.listdir('./images/test/ZERO')
N=len(names)
z0 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/test/ZERO/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  z0[k,:]=pic2
t0=numpy.zeros((N, 1), dtype=numpy.int)


print('reading 1 from test set')
names = os.listdir('./images/test/ONE')
N=len(names)
z1 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/test/ONE/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  z1[k,:]=pic2
t1=numpy.ones((N, 1), dtype=numpy.int)


print('reading 2 from test set')
names = os.listdir('./images/test/TWO')
N=len(names)
z2 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/test/TWO/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  z2[k,:]=pic2
t2=numpy.ones((N, 1), dtype=numpy.int)*2



print('reading 3 from test set')
names = os.listdir('./images/test/THREE')
N=len(names)
z3 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/test/THREE/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  z3[k,:]=pic2
t3=numpy.ones((N, 1), dtype=numpy.int)*3



print('reading 4 from test set')
names = os.listdir('./images/test/FOUR')
N=len(names)
z4 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/test/FOUR/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  z4[k,:]=pic2
t4=numpy.ones((N, 1), dtype=numpy.int)*4


print('reading 5 from test set')
names = os.listdir('./images/test/FIVE')
N=len(names)
z5 = numpy.empty((N,16384), int)
for k in range(N):
  address = current_dir + '/images/test/FIVE/' + names[k]
  #print(k, ' / ' , N, '\t=\t', k/N)
  pic = cv2.imread(address,0)
  resized = numpy.rint((cv2.resize(pic, (128,128), interpolation = cv2.INTER_AREA))/255)
  pic2 = resized.reshape((1, 16384))
  z5[k,:]=pic2
t5=numpy.ones((N, 1), dtype=numpy.int)*5


print('concatenate test set')
z_test = numpy.concatenate((z0,z1,z2,z3,z4,z5), axis=0)
Y_test = numpy.concatenate((t0,t1,t2,t3,t4,t5), axis=0)

print('saving test set as binary')
numpy.save('test_data.npy', z_test)
numpy.save('test_label.npy', Y_test)
