"""
Train modified Inception V3 network using mxnet
"""

__author__ = 'bshang'

import sys
sys.path.append('/home/ubuntu/yelp/mxnet/python')
import mxnet as mx

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from metrics import f1

import importlib


def get_iter():
    train_iter = mx.io.ImageRecordIter(
        path_imgrec = '/data/rec/train_imgs.rec',
        mean_r      = 117,
        mean_g      = 117,
        mean_b      = 117,
        data_shape  = (3, 299, 299),
        batch_size  = 16,
        path_imglist= '/data/rec/train_imgs.lst',
        label_width = 9,
        rand_crop   = True,
        rand_mirror = True,
        max_rotate_angle = 15,
        max_crop_size = 299,
        min_crop_size = 224,
        shuffle = True,
        seed = 0
        )
    return train_iter

MODEL_PREFIX = '/data/checkpoint/inception-v3'

PARAM_FILE = './mxnet_model/inception_v3/Inception-7-0001.params'
sym_net = importlib.import_module('symbol_inception_v3').get_symbol(9)
init_exist = mx.init.Load(PARAM_FILE,
                          default_init=mx.init.Xavier(factor_type="in", magnitude=2.34),
                          verbose=False)

train_iter = get_iter()

RECOVER = False
RECOVER_START = 0

TOTAL_EPOCHS = 3
EFFECTIVE_EPOCHS = TOTAL_EPOCHS*10  # batch_size*epoch_size ~ 10% of total images

if RECOVER:
    cnn_model = model = mx.model.FeedForward.load(
        MODEL_PREFIX,
        RECOVER_START,
        ctx                 = mx.gpu(),
        num_epoch           = EFFECTIVE_EPOCHS,
        epoch_size          = 1466,
        learning_rate       = 0.001,
        momentum            = 0.9,
        wd                  = 0.00001,
    )
    print('Recovered model')
else:
    cnn_model = mx.model.FeedForward(
        ctx                = [mx.gpu(i) for i in range(1)],
        symbol             = sym_net,
        num_epoch          = EFFECTIVE_EPOCHS,
        epoch_size         = 1466,  # checkpoint every 10% complete/epoch
        learning_rate      = 0.001,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = init_exist,
        allow_extra_params = True
    )

print('fitting')
cnn_model.fit(
    X                  = train_iter,
    eval_metric        = mx.metric.np(f1),
    batch_end_callback = mx.callback.Speedometer(1),
    epoch_end_callback = mx.callback.do_checkpoint(MODEL_PREFIX),
)
