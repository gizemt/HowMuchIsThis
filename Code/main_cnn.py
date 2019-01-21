import cnn
import reader
import params
import numpy as np


def main_cnn():
    k = 1
    x_train, y_train, x_test, y_test, scale = reader.get_joint_data(params.joint_data_path, params.max_size)
    print 'Training ', np.shape(x_train), np.shape(y_train), 'Testing', np.shape(x_test), np.shape(y_test)

    print 'Training model ...'
    cnn.model.fit(x_train, y_train, nb_epoch=5)

    print 'Evaluating ...'
    score = cnn.model.evaluate(x_test, y_test)
    print 'Loss on test', score

    # y_pred = []
    # for i in range(np.shape(x_test)[0]):
    #     x = np.tile(x_test[i, :], (np.shape(x_train)[0], 1))
    #     distance = np.mean((x - x_train)**2, 1)
    #     nn_indices = np.argsort(distance, axis=0)[0:k]
    #     nn_price = np.mean(y_train[nn_indices])
    #     y_pred.append(nn_price)

    y_pred = cnn.model.predict(x_test)
    # print 'Gold : ', y_test
    # print 'Prediction : ', y_pred
    err = ((y_pred - y_test)*(y_pred - y_test))
    acc = np.sum(err <= ((params.price_threshold*params.price_threshold)/(scale*scale)))
    print 'Accuracy = ', acc, ' / ', len(y_test), ' = ', acc * 1.0 / len(y_test)


if __name__ == "__main__":
    main_cnn()