from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution1D(32, 3, init = 'lecun_uniform', border_mode='valid', input_dim=4096)
model.add(Activation('relu'))
model.add(Convolution1D(32, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length = 2, stride = 2))
model.add(Dropout(0.25))

model.add(Convolution1D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution1D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length = 2, stride = 2))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)