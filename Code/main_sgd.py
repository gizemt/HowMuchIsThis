import sgd_network
import reader
import params
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main_sgd():
    x_train, y_train, x_test, y_test, scale = reader.get_joint_data(params.joint_data_path, params.max_size)

    print y_train.max(), y_test.max()
    tbins = np.linspace(0, np.ceil(np.max([y_train.max()*scale, y_test.max()*scale])/10)*10, np.ceil(np.max([y_train.max()*scale, y_test.max()*scale])/10))/scale
    train_idx = np.digitize(y_train, tbins)
    y_train = tbins[train_idx-1]
    test_idx = np.digitize(y_test, tbins)
    y_test = tbins[test_idx-1]

    print 'Scale ', scale
    print 'Training ', np.shape(x_train), np.shape(y_train), 'Testing', np.shape(x_test), np.shape(y_test)

    print 'Training model ...'
    fitlog = sgd_network.model.fit(x_train, y_train, shuffle = True, batch_size = 512, nb_epoch=30)

    plt.plot(fitlog.epoch, fitlog.history['acc'], 'g-', label = 'Accuracy')
    #plt.plot(fitlog.epoch, fitlog.history['val_acc'], 'g--')
    plt.plot(fitlog.epoch, fitlog.history['loss'], 'r-', label = 'Error')
    plt.xlabel('Iterations')
    #plt.ylabel('Error')
    # plt.plot(fitlog.epoch, fitlog.history['val_loss'], 'r--')
    plt.legend()
    plt.savefig('fitlog.pdf')
    plt.close()
    print 'Evaluating ...'
    score = sgd_network.model.evaluate(x_test, y_test)
    print 'Loss on test', score

    y_pred = sgd_network.model.predict(x_test)
    pred_idx = np.digitize(y_pred, tbins)
    y_pred = tbins[pred_idx-1]
   # plt.figure(figsize=(5, 5n))
    dumb, test_bins, patches = plt.hist(y_test*scale, bins = 100, color='g', label='Test')
    plt.hist(y_pred*scale, bins =100, range = (test_bins[0], test_bins[-1]), alpha = 0.5, color='r', label='Predicted')
    plt.xlabel('Price')
    plt.legend()
    plt.savefig('hist.pdf')
    plt.close()
    plt.hist(y_test*scale, bins = 100, range = (0, 100), color='g', label='Test')
    plt.hist(y_pred*scale, bins =100, range = (0, 100), alpha = 0.5, color='r', label='Predicted')
    plt.xlabel('Price')
    plt.legend()
    plt.savefig('hist_crop.pdf')
    plt.close()


    y_rand = np.random.uniform(y_test.min(), y_test.max(), np.shape(y_test))
    err_rand = (y_rand - y_test)*(y_rand - y_test)
    acc_rand = np.sum(err_rand <= ((params.price_threshold*params.price_threshold)/(scale*scale)))

    print 'Gold : ', y_test[:20]*scale
    print 'Prediction : ', y_pred[:20]*scale

    err = ((y_pred - y_test)*(y_pred - y_test))
    ind_threshold = (y_test*0.2)*(y_test*0.2)
    #acc = np.sum(err - ind_threshold <= 0)
    acc = np.sum(err <= (params.price_threshold*params.price_threshold)/(scale*scale))
    print 'Accuracy = ', acc, ' / ', len(y_test), ' = ', acc * 1.0 / len(y_test)
    print 'Random guess accuracy = ', acc_rand, ' /', len(y_test), ' = ', acc_rand * 1.0 / len(y_test)


if __name__ == "__main__":
    main_sgd()
