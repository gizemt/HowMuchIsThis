import reader
import params
import numpy as np

#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

def mean_estimates():

        y_train, y_test, scale, cats_test, cats_train, means = reader.get_mean_estimates(params.joint_data_path, params.max_size)

        y_train_dict = {}
        y_mean = {}
        keys = []
        y_pred = []
        for i in np.arange(len(cats_train)):
                category = cats_train[i]
                if category not in keys:
                    keys.append(category)
                    y_train_dict[category] = []
                y_train_dict[category].append(y_train[i])
        
        for j in np.arange(len(y_test)):
            #print cats_test[i]
            y_pred.append(means[cats_test[j]]/scale)
            #print means[cats_test[j]]/scale

        for cats in keys:
            print 'Category ', cats, ' min price ', np.min(y_train_dict[cats])*scale, ' max price ', np.max(y_train_dict[cats])*scale,' average of category ', means[cats]
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(len(y_test),1)
        #print np.shape(y_pred), np.shape(y_test), np.shape(means)
        print 'Min error', np.min(abs(y_pred - y_test))*scale, 'Min idx = ', np.argmin(y_pred - y_test)
        print 'Gold : ', y_test[:20]*scale
        print 'Prediction : ', y_pred[:20]*scale

        err = ((y_pred - y_test)*(y_pred - y_test))

        acc = np.sum(err <= (params.price_threshold*params.price_threshold)/(scale*scale))
        print 'Accuracy = ', acc, ' / ', len(y_test), ' = ', acc * 1.0 / len(y_test)

if __name__ == "__main__":
    mean_estimates() 