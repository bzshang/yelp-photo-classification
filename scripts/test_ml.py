"""
Predict labels using trained ML models. Use average probability ensemble.
"""

__author__ = 'bshang'

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.externals import joblib


def convert_label_to_array(str_label):
    str_label = str_label.split(' ')
    return [int(x) for x in str_label if len(x) > 0]

MODEL = 'inception-v3'
LAYER = 'global_pool_output'
NUM_EPOCH = 30
BIZ_FEATURES_PATH = '/data/test_biz_features_{0}_{1}_{2}.h5'.format(MODEL, LAYER, NUM_EPOCH)

df = pd.read_csv(BIZ_FEATURES_PATH, header=0)

cols = ["F" + str(i+1) for i in range(0, 2048)]
X = df[cols].values

model_svc = joblib.load('/data/skmodels/svc_inception-v3.pkl')
model_lrc = joblib.load('/data/skmodels/lrc_inception-v3.pkl')
model_rfc = joblib.load('/data/skmodels/rfc_inception-v3.pkl')

print('predict svc')
y_predict_proba_svc = model_svc.predict_proba(X)

print('predict lrc')
y_predict_proba_lrc = model_lrc.predict_proba(X)

print('predict rfc')
y_predict_proba_rfc = model_rfc.predict_proba(X)

y_predict_proba = np.mean(
    np.array([y_predict_proba_svc, y_predict_proba_lrc, y_predict_proba_rfc]), axis=0)

THRESHOLD = 0.46  # estimated from cross-validation
y_predict = preprocessing.binarize(y_predict_proba, threshold=THRESHOLD)

# convert the binary labels back to numbered labels
df_biz2lab = pd.read_csv('/data/train.csv').dropna()
y = np.array([convert_label_to_array(y) for y in df_biz2lab['labels']])
mlb = preprocessing.MultiLabelBinarizer()
mlb.fit_transform(y)
y_ = mlb.inverse_transform(y_predict)  # y_ contain the numbered labels

y_ = [' '.join(str(x) for x in ls) for ls in y_]

df['labels'] = pd.Series(y_, index=df.index)
df = df.sort_values('business_id')

with open('/data/submission/inception_v3_svc_rfc_lrc_epoch3.csv', 'w') as f:
    df[['business_id', 'labels']].to_csv(f, index=False)



