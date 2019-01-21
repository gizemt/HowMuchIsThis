import reader
import params
import numpy as np
x_train, y_train, x_test, y_test, scale, asin_test = reader.get_joint_data(params.joint_data_path, 5)
print y_test*scale, asin_test
print asin_test[np.argmax(y_test)]
