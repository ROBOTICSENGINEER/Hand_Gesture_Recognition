import numpy

	
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten

print('Loading data')
train_x = numpy.load('train_data.npy')
train_y = numpy.load('train_label.npy')
test_x = numpy.load('test_data.npy')
test_y = numpy.load('test_label.npy')


X_train = train_x.reshape(train_x.shape[0], 128, 128, 1)
X_test = test_x.reshape(test_x.shape[0],  128, 128, 1)
Y_train = np_utils.to_categorical(train_y, 6)
Y_test = np_utils.to_categorical(test_y, 6)

print('creating model')
model = Sequential()
# 128 x 128 x 1
model.add(Convolution2D(filters = 8, kernel_size = 3, strides=2, padding='SAME', activation='relu', input_shape=(128,128,1)))
# 64 x 64 x 8
model.add(Convolution2D(filters = 16, kernel_size = 3, strides=2, padding='SAME', activation='relu'))
# 32 x 32 x 16
model.add(Convolution2D(filters = 32, kernel_size = 3, strides=2, padding='SAME', activation='relu'))
# 16 x 16 x 32
model.add(Convolution2D(filters = 64, kernel_size = 3, strides=2, padding='SAME', activation='relu'))
# 8 x 8 x 64
model.add(Convolution2D(filters = 128, kernel_size = 3, strides=2, padding='SAME', activation='relu'))
# 4 x 4 x 128
model.add(Convolution2D(filters = 128, kernel_size = 3, strides=2, padding='SAME', activation='relu'))
# 2 x 2 x 128
model.add(Convolution2D(filters = 128, kernel_size = 3, strides=2, padding='SAME', activation='relu'))
# 1 x 1 x 128
model.add(Flatten())
# 128
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# 64
model.add(Dense(6, activation='softmax'))
# 6

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

print('training model')
model.fit(x=X_train, y=Y_train, epochs=10, batch_size=128, validation_data = (X_test, Y_test) , verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
