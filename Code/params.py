# Path names of downloaded files
metadata_path = 'meta_Home_and_Kitchen.json.gz'
image_feats_path = 'image_features_Home_and_Kitchen.b'

# Path name of joint data with image features
joint_data_path = '/shared/austen/sroy9/vision/joint_Home_and_Kitchen.json.gz'

# Maximum number of data points to be read
max_size = 10000

# Fraction of data to be used as test
test_split = 0.2

# Difference in predicted price which can still be considered correct
# Used only in evaluation
price_threshold = 10.0

# Number of epochs
num_epochs = 10
