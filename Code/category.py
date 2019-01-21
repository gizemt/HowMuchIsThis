from keras.models import Sequential
from keras.layers import Dense, Dropout
import reader
import params
import numpy as np
import decision_tree


x_train, y_train, z_train, x_test, y_test, z_test, scale = reader.get_joint_category_data(
    params.joint_data_path, params.max_size)

print 'Training ', np.shape(x_train), np.shape(y_train), np.shape(z_train)
print 'Testing', np.shape(x_test), np.shape(y_test), np.shape(z_test)

model1 = Sequential()
model1.add(Dense(4096, input_dim=len(x_test[0]), init='uniform', activation='sigmoid'))
model1.add(Dropout(0.5))
model1.add(Dense(4096, init='uniform', activation='sigmoid'))
model1.add(Dropout(0.5))
model1.add(Dense(len(z_test[0]), init='uniform', activation='sigmoid'))

# print 'Loading model weights ...'
# model1.load_weights('model1.h5')

print 'Compiling model ...'
model1.compile(optimizer='sgd', loss='categorical_crossentropy')

print 'Training model ...'
model1.fit(x_train, z_train, nb_epoch=params.num_epochs)

# print 'Saving model weights...'
# model1.save_weights('model1.h5')

print 'Evaluating ...'
score = model1.evaluate(x_test, z_test)
print 'Loss on test', score


model2 = Sequential()
model2.add(Dense(4096, input_dim=len(z_test[0]), init='uniform', activation='sigmoid'))
model2.add(Dropout(0.5))
model2.add(Dense(4096, init='uniform', activation='sigmoid'))
model2.add(Dropout(0.5))
model2.add(Dense(len(y_test[0]), init='uniform', activation='sigmoid'))

# print 'Loading model weights ...'
# model2.load_weights('model2.h5')

print 'Compiling model ...'
model2.compile(optimizer='sgd', loss='mse')

print 'Training model ...'
model2.fit(model1.predict(x_train), y_train, nb_epoch=params.num_epochs)

# print 'Saving model weights...'
# model2.save_weights('model2.h5')

print 'Evaluating ...'
score = model2.evaluate(z_test, y_test)
print 'Loss on test', score

print 'Running pipeline on train data ...'
z_pred = model1.predict(x_train)
y_pred = model2.predict(z_pred)
err = ((y_pred - y_train)*(y_pred - y_train))
acc = np.sum(err <= ((params.price_threshold*params.price_threshold)/(scale*scale)))
print 'Accuracy = ', acc, ' / ', len(y_train), ' = ', acc * 1.0 / len(y_train)

print 'Running pipeline on test data ...'
z_pred = model1.predict(x_test)
y_pred = model2.predict(z_pred)
err = ((y_pred - y_test)*(y_pred - y_test))
acc = np.sum(err <= ((params.price_threshold*params.price_threshold)/(scale*scale)))
print 'Accuracy = ', acc, ' / ', len(y_test), ' = ', acc * 1.0 / len(y_test)

print y_pred
print y_test

decision_tree.plot_histogram(y_test, y_pred, scale)
