import gzip
import params
import struct
import numpy as np
import json


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def read_image_features(path):
    f = open(path, 'rb')
    while True:
        asin = f.read(10)
        if asin == '':
            break
        feature = []
        for i in range(4096):
            feature.append(struct.unpack('f', f.read(4)))
        yield asin, feature


def get_metadata_map(path):
    metadata = {}
    for k in parse(path):
        metadata[k['asin']] = k
    return metadata


def create_joint_data(metadata_path, image_feats_path, joint_data_path, max_size=280000):
    print 'Creating joint data ...'
    count = 0
    metadata_map = get_metadata_map(metadata_path)
    outfile = gzip.GzipFile(joint_data_path, 'w')
    for k in read_image_features(image_feats_path):
        asin = k[0]
        if asin in metadata_map and 'price' in metadata_map[asin]:
            metadata_map[asin]['image_feats'] = [l[0] for l in k[1]]
            outfile.write(json.dumps(metadata_map[asin]) + '\n')
            count += 1
            if count >= max_size:
                break
            if count % 10000 == 0:
                print 'Gathered ', count, ' joint data points'

    print 'Done creating joint data'


def get_joint_data(joint_data_path, max_size=280000):
    print 'Reading data from joint data file ...'
    x = []
    y = []
    asin = []
    count = 0
    for k in parse(joint_data_path):
        asin.append(k['asin'])
        x.append(k['image_feats'])
        y.append(k['price'])
        count += 1
        if count % 1000 == 0:
            print 'Read ', count, ' data points'
        if count >= max_size:
            break

    # Convert to numpy arrays, and scale prices to be between 0 and 1
    x = np.array(x)
    y = np.array(y)
    y_max = np.max(y)
    y = y * 1.0 / y_max
    x_test = x[:int(params.test_split*len(y))]
    y_test = y[:int(params.test_split*len(y))]
    asin_test = asin[:int(params.test_split*len(y))]
    x_train = x[int(params.test_split*len(y)):]
    y_train = y[int(params.test_split*len(y)):]
    train_size = len(y_train)
    test_size = len(y_test)
    y_test = y_test.reshape(test_size, 1)
    y_train = y_train.reshape(train_size, 1)
    print 'Done reading joint data'
    return x_train, y_train, x_test, y_test, y_max, asin_test


def get_joint_category_data(joint_data_path, max_size=280000):
    print 'Reading category data from joint data file ...'
    categories = []
    count = 0
    x = []
    y = []
    z_str = []

    for k in parse(joint_data_path):
        for category in k['categories'][0]:
            if category not in categories:
                categories.append(category)
        x.append(k['image_feats'])
        y.append(k['price'])
        z_str.append(k['categories'][0][-1:])
        count += 1
        if count % 1000 == 0:
            print 'Read ', count, ' data points'
        if count >= max_size:
            break

    z = []
    for i in range(len(z_str)):
        z.append(np.zeros(len(categories), 'uint8'))
        for j in z_str[i]:
            z[i][categories.index(j)] = 1

    # Convert to numpy arrays, and scale prices to be between 0 and 1
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    y_max = np.max(y)
    y = y * 1.0 / y_max

    x_test = x[:int(params.test_split*len(y))]
    y_test = y[:int(params.test_split*len(y))]
    z_test = z[:int(params.test_split*len(y))]

    x_train = x[int(params.test_split*len(y)):]
    y_train = y[int(params.test_split*len(y)):]
    z_train = z[int(params.test_split*len(y)):]

    train_size = len(y_train)
    test_size = len(y_test)

    y_test = y_test.reshape(test_size, 1)
    y_train = y_train.reshape(train_size, 1)

    print 'Done reading joint data'
    return x_train, y_train, z_train, x_test, y_test, z_test, y_max


def category_stats(joint_data_path, max_size=280000):
    print 'Generating category stats ...'
    max_price = {}
    min_price = {}
    count = 0
    for k in parse(joint_data_path):
        for category in k['categories'][0]:
            if category not in max_price.keys():
                max_price[category] = k['price']
            if category not in min_price.keys():
                min_price[category] = k['price']
            if category in max_price.keys() and max_price[category] < k['price']:
                max_price[category] = k['price']
            if category in min_price.keys() and min_price[category] > k['price']:
                min_price[category] = k['price']

        count += 1
        if count % 1000 == 0:
            print 'Read ', count, ' data points'
        if count >= max_size:
            break

    lessThan10 = 0
    lessThan20 = 0
    lessThan30 = 0
    good_categories = []
    for category in max_price.keys():
        if max_price[category] - min_price[category] < 10:
            lessThan10 += 1
            good_categories.append(category)
        if max_price[category] - min_price[category] < 20:
            lessThan20 += 1
        if max_price[category] - min_price[category] < 30:
            lessThan30 += 1

    print lessThan10, lessThan20, lessThan30, len(max_price.keys())

    good_cover = 0
    count = 0
    for k in parse(joint_data_path):
        for category in k['categories'][0]:
            if category in good_categories:
                good_cover += 1
                break

        count += 1
        if count % 1000 == 0:
            print 'Read ', count, ' data points'
        if count >= max_size:
            break

    print 'Good cover : ', good_cover, count


def main():
    category_stats(params.joint_data_path, params.max_size)


if __name__ == "__main__":
    main()