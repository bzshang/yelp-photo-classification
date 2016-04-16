"""
Get training set business features by averaging image features
"""

__author__ = 'bshang'

import numpy as np
import pandas as pd
import h5py


train_photo_to_biz = pd.read_csv('/data/train_photo_to_biz_ids.csv')

cols = []
cols.append('rid')
cols.extend(["L" + str(i+1) for i in range(0, 9)])
cols.append("path")
photo_to_label = pd.read_table('/data/rec/train_imgs.lst', sep='\t', header=None, names=cols)
photo_to_label['photo_id'] = photo_to_label['path'].apply(lambda x: int(x.split('.')[0]))

MODEL = 'inception-v3'
LAYER = 'global_pool_output'
NUM_EPOCH = 30

FEATURES_PATH = '/data/train_image_features_{0}_{1}_{2}.h5'.format(MODEL, LAYER, NUM_EPOCH)
with h5py.File(FEATURES_PATH, 'r') as f:
    features = f['feature'][()]
    pids = f['pids'][()]

print("features", features.shape)
print("pids", pids.shape)

a = np.concatenate([pids.reshape((pids.shape[0], 1)), features], axis=1)

cols = []
cols.append('photo_id')
cols.extend(["F" + str(i+1) for i in range(0, 2048)])
photo_to_features = pd.DataFrame(a, columns=cols)

df = pd.merge(train_photo_to_biz, photo_to_features, how='inner', on='photo_id')
df = pd.merge(df, photo_to_label, how='inner', on='photo_id')

cols = []
cols.append('business_id')
cols.extend(["F" + str(i+1) for i in range(0, 2048)])
cols.extend(["L" + str(i+1) for i in range(0, 9)])
dfg_mean = df[cols].groupby('business_id').agg(np.mean)

with open('/data/train_biz_features_{0}_{1}_{2}.h5'.format(MODEL, LAYER, NUM_EPOCH), 'w') as f:
    dfg_mean.to_csv(f, index=True)
