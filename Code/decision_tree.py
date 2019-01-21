from keras.models import Sequential
from keras.layers import Dense, Dropout
import reader
import params
import numpy as np
import matplotlib.pyplot as plt

# Has to be TrainAndTest or Test
mode = 'TrainAndTest'


def main():
    x_train, y_train, x_test, y_test, scale, asin = reader.get_joint_data(
        params.joint_data_path, params.max_size)
    models = {}
    models[(0.0, 0.1, 0.0, 0.2)] = get_trained_model(x_train, y_train, x_test, y_test, 0.0, 0.1, 0.0, 0.2)

    models[(0.0, 0.05, 0.0, 0.1)] = get_trained_model(x_train, y_train, x_test, y_test, 0.0, 0.05, 0.0, 0.1)
    models[(0.1, 0.15, 0.1, 0.2)] = get_trained_model(x_train, y_train, x_test, y_test, 0.1, 0.15, 0.1, 0.2)

    models[(0.0, 0.025, 0.0, 0.05)] = get_trained_model(x_train, y_train, x_test, y_test, 0.0, 0.025, 0.0, 0.05)
    models[(0.05, 0.075, 0.05, 0.1)] = get_trained_model(x_train, y_train, x_test, y_test, 0.05, 0.075, 0.05, 0.1)
    models[(0.1, 0.125, 0.1, 0.15)] = get_trained_model(x_train, y_train, x_test, y_test, 0.1, 0.125, 0.1, 0.15)
    models[(0.15, 0.175, 0.15, 0.2)] = get_trained_model(x_train, y_train, x_test, y_test, 0.15, 0.175, 0.15, 0.2)

    y_pred = []
    for i in range(len(x_test)):
        y_pred.append([predict_decision_tree(x_test[i], models)])

    y_pred = np.array(y_pred)

    print y_pred
    print y_test

    plot_histogram(y_test, y_pred, scale)
    # get best predicted item id
    print 'Min error = ', np.min(abs(y_pred - y_test))*scale, 'on item with ASIN ', asin[np.argmin(y_pred - y_test)]

    err = ((y_pred - y_test)*(y_pred - y_test))
    acc = np.sum(err <= ((params.price_threshold*params.price_threshold)/(scale*scale)))
    print 'Test Accuracy = ', acc, ' / ', len(y_test), ' = ', acc * 1.0 / len(y_test)


def predict_decision_tree(x, models):
    l = len(x)
    x = x.reshape(1, l)
    y = models[(0.0, 0.1, 0.0, 0.2)].predict(x)
    if y > 0.5:
        y = models[(0.0, 0.05, 0.0, 0.1)].predict(x)
        if y > 0.5:
            y = models[(0.0, 0.025, 0.0, 0.05)].predict(x)
            if y > 0.5:
                return 0.0125
            else:
                return 0.0375
        else:
            y = models[(0.05, 0.075, 0.05, 0.1)].predict(x)
            if y > 0.5:
                return 0.0625
            else:
                return 0.0875
    else:
        y = models[(0.1, 0.15, 0.1, 0.2)].predict(x)
        if y > 0.5:
            y = models[(0.1, 0.125, 0.1, 0.15)].predict(x)
            if y > 0.5:
                return 0.1125
            else:
                return 0.1375
        else:
            y = models[(0.15, 0.175, 0.15, 0.2)].predict(x)
            if y > 0.5:
                return 0.1625
            else:
                return 0.1875


def get_trained_model(x_train, y_train, x_test, y_test, min_gold, max_gold, min_total, max_total):
    x_tr = []
    y_tr = []
    x_te = []
    y_te = []
    for i in range(len(x_train)):
        if min_gold < y_train[i][0] < max_gold:
            y_tr.append([1])
            x_tr.append(x_train[i])
        elif min_total < y_train[i][0] < max_total:
            y_tr.append([0])
            x_tr.append(x_train[i])

    for i in range(len(x_test)):
        if min_gold < y_test[i][0] < max_gold:
            y_te.append([1])
            x_te.append(x_test[i])
        elif min_total < y_test[i][0] < max_total:
            y_te.append([0])
            x_te.append(x_test[i])

    print 'Training ', np.shape(x_tr), np.shape(y_tr)
    print 'Testing', np.shape(x_te), np.shape(y_te)

    model = Sequential()
    model.add(Dense(4096, input_dim=len(x_te[0]), init='uniform', activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, init='uniform', activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    if mode == 'Test':
        print 'Loading model weights ...'
        model.load_weights(str(min_gold)+str(max_gold)+str(min_total)+str(max_total)+'.model')

    print 'Compiling model ...'
    model.compile(optimizer='sgd', loss='binary_crossentropy')

    if mode == 'Test':
        return model

    print 'Training model ...'
    x_tr = np.array(x_tr)
    y_tr = np.array(y_tr)
    x_te = np.array(x_te)
    y_te = np.array(y_te)
    fitlog = model.fit(x_tr, y_tr, nb_epoch=params.num_epochs)

    # plt.plot(fitlog.epoch, fitlog.history['acc'], 'g-', label = 'Accuracy')
    # plt.savefig('acc.pdf')
    # plt.close()
    # plt.plot(fitlog.epoch, fitlog.history['val_acc'], 'g--')
    # plt.plot(fitlog.epoch, fitlog.history['loss'], 'r-', label = 'Error')
    # plt.xlabel('Iterations')
    # plt.ylabel('Error')
    # plt.plot(fitlog.epoch, fitlog.history['val_loss'], 'r--')
    # plt.legend()
    # plt.savefig('fitlog.pdf')
    # plt.close()

    print 'Saving model weights...'
    model.save_weights(str(min_gold)+str(max_gold)+str(min_total)+str(max_total)+'.model')

    print 'Evaluating ...'
    y_pred = model.predict(x_te)
    y_pred = y_pred > 0.5
    acc = np.sum(y_pred == y_te)
    print 'Accuracy = ', acc, ' / ', len(y_te), ' = ', acc * 1.0 / len(y_te)
    return model


def plot_histogram(y_test, y_pred, scale):
    dumb, test_bins, patches = plt.hist(np.multiply(y_test, scale), bins=100, color='g', label='Test')
    plt.hist(np.multiply(y_pred, scale), bins=100, range=(test_bins[0], test_bins[-1]), alpha=0.5, color='r', label='Predicted')
    plt.xlabel('Price')
    plt.legend()
    plt.savefig('hist.pdf')
    plt.close()
    plt.hist(np.multiply(y_test, scale), bins=100, range=(0, 100), color='g', label='Test')
    plt.hist(np.multiply(y_pred, scale), bins=100, range=(0, 100), alpha=0.5, color='r', label='Predicted')
    plt.xlabel('Price')
    plt.legend()
    plt.savefig('hist_crop.pdf')
    plt.close()


if __name__ == "__main__":
    main()
