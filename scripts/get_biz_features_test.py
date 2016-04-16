"""
Get test set business features by averaging image features
"""

__author__ = 'bshang'

import numpy as np
import pandas as pd
import h5py


MODEL = 'inception-v3'
LAYER = 'global_pool_output'
NUM_EPOCH = 30

FEATURES_PATH = '/data/test_image_features_{0}_{1}_{2}.h5'.format(MODEL, LAYER, NUM_EPOCH)
BIZ_FEATURES_PATH = '/data/test_biz_features_{0}_{1}_{2}.h5'.format(MODEL, LAYER, NUM_EPOCH)

with h5py.File(FEATURES_PATH, 'r') as f:
    features = f['feature'][()]
    pids = f['pids'][()]

a = np.concatenate([pids.reshape((pids.shape[0], 1)), features], axis=1)

cols = []
cols.append('photo_id')
cols.extend(["F" + str(i+1) for i in range(0, 2048)])
photo_to_features = pd.DataFrame(a, columns=cols)

photo_to_biz = pd.read_csv('/data/test_photo_to_biz.csv')

biz_set = set()
for biz in photo_to_biz['business_id']:
    biz_set.add(biz)

output = []
output.append("business_id")
output.extend(["F" + str(i+1) for i in range(0, 2048)])
output.extend(["L" + str(i+1) for i in range(0, 9)])

with open(BIZ_FEATURES_PATH, 'w') as f:
    f.write(','.join(str(i) for i in output) + '\n')

for idx, biz in enumerate(biz_set):
    if idx % 100 == 0:
        print(idx)
    output = []
    output.append(biz)
    photos = set(photo_to_biz[photo_to_biz['business_id'] == biz]['photo_id'].values)
    features = photo_to_features[photo_to_features['photo_id'].isin(photos)][["F" + str(i+1) for i in range(0, 2048)]].values
    m = np.mean(features, axis=0)
    output.extend(m)
    output.extend([1, 0, 0, 0, 0, 0, 0, 0, 0])
    with open(BIZ_FEATURES_PATH, 'a') as f:
        f.write(','.join(str(i) for i in output) + '\n')
