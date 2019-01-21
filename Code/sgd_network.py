from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.optimizers import SGD, Adagrad


# left_branch = Sequential()
# left_branch.add(Dense(32, input_dim=4096))

# right_branch = Sequential()
# right_branch.add(Dense(32, input_dim=4096))

# merged = Merge([left_branch, right_branch], mode='concat')

model = Sequential()
model.add(Dense(1024, input_dim=4096, init='lecun_uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, input_dim=4096, init='lecun_uniform', activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(1024, input_dim=4096, init='he_normal', activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='he_normal', activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution1D(32, 2, init='lecun_uniform', activation='relu', border_mode='valid', input_dim=64))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.00005, decay=1e-6, momentum=0.9)
ag = Adagrad(lr=0.01, epsilon=1e-06)
# model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
