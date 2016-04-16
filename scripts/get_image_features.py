"""
Extract image features from next to last layer (global_pool)
"""

__author__ = 'bshang'

import numpy as np
import h5py

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import sys
sys.path.append('/home/ubuntu/yelp/mxnet/python')
import mxnet as mx


MODEL = 'inception-v3'
MODEL_PATH = '/data/checkpoint/{0}'.format(MODEL)
LAYER = 'global_pool_output'

NUM_EPOCH = 30

TEST = False

if TEST:
    FEATURES_PATH = '/data/test_image_features_{0}_{1}_{2}.h5'.format(MODEL, LAYER, NUM_EPOCH)
    REC_FILE = '/data/rec/test_imgs.rec'
    LAB_FILE = '/data/rec/test_imgs.lst'
else:
    FEATURES_PATH = '/data/train_image_features_{0}_{1}_{2}.h5'.format(MODEL, LAYER, NUM_EPOCH)
    REC_FILE = '/data/rec/train_imgs.rec'
    LAB_FILE = '/data/rec/train_imgs.lst'

f = h5py.File(FEATURES_PATH, 'w')
filenames = f.create_dataset('pids', (0,), maxshape=(None,))
feature = f.create_dataset('feature', (0, 2048), maxshape=(None, 2048))  # 2048 features in global_pool
f.close()

with open(LAB_FILE, 'r') as f:
    pids = [line.split('\t')[-1].split('.')[0] for line in f]
with h5py.File(FEATURES_PATH, 'r+') as f:
    f['pids'].resize((len(pids),))
    f['pids'][0: len(pids)] = np.array(pids, dtype=np.int64)

model = mx.model.FeedForward.load(MODEL_PATH, NUM_EPOCH, ctx=mx.gpu())
fea_symbol = model.symbol.get_internals()[LAYER]
feature_extractor = mx.model.FeedForward(
    ctx=mx.gpu(),
    symbol=fea_symbol,
    arg_params=model.arg_params,
    aux_params=model.aux_params,
    allow_extra_params=True)

model_iter = mx.io.ImageRecordIter(
    path_imgrec = REC_FILE,
    mean_r      = 117,
    mean_g      = 117,
    mean_b      = 117,
    data_shape  = (3, 299, 299),
    batch_size  = 32,
    rand_crop   = False,
    rand_mirror = False,
    path_imglist= LAB_FILE,
    label_width = 9
)

features = feature_extractor.predict(model_iter)
features = features[:, :, 0, 0]

with h5py.File(FEATURES_PATH, 'r+') as f:
    f['feature'].resize((features.shape[0], features.shape[1]))
    f['feature'][0: features.shape[0], :] = features
