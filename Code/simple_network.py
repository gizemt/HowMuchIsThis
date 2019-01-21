from keras.models import Sequential
from keras.layers import Dense, Dropout
import reader
import params
import numpy as np
import decision_tree


x_train, y_train, x_test, y_test, scale, asin = reader.get_joint_data(
    params.joint_data_path, params.max_size)

print 'Training ', np.shape(x_train), np.shape(y_train)
print 'Testing', np.shape(x_test), np.shape(y_test)

model1 = Sequential()
model1.add(Dense(4096, input_dim=len(x_test[0]), init='uniform', activation='sigmoid'))
model1.add(Dropout(0.5))
model1.add(Dense(4096, init='uniform', activation='sigmoid'))
model1.add(Dropout(0.5))
model1.add(Dense(len(y_test[0]), init='uniform', activation='sigmoid'))

print 'Compiling model ...'
model1.compile(optimizer='sgd', loss='mse')

print 'Training model ...'
model1.fit(x_train, y_train, nb_epoch=params.num_epochs)

print 'Evaluating ...'
score = model1.evaluate(x_test, y_test)
print 'Loss on test', score

y_pred = model1.predict(x_train)
err = ((y_pred - y_train)*(y_pred - y_train))
acc = np.sum(err <= ((params.price_threshold*params.price_threshold)/(scale*scale)))
print 'Train Accuracy = ', acc, ' / ', len(y_train), ' = ', acc * 1.0 / len(y_train)

y_pred = model1.predict(x_test)
err = ((y_pred - y_test)*(y_pred - y_test))
acc = np.sum(err <= ((params.price_threshold*params.price_threshold)/(scale*scale)))
print 'Test Accuracy = ', acc, ' / ', len(y_test), ' = ', acc * 1.0 / len(y_test)

print y_pred
print y_test

decision_tree.plot_histogram(y_test, y_pred, scale)
