"""
Train ML models using extracted CNN features
"""

__author__ = 'bshang'

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.externals import joblib


MODEL = 'inception-v3'
LAYER = 'global_pool_output'
NUM_EPOCH = 30

FEATURES_PATH = '/data/train_biz_features_{0}_{1}_{2}.h5'.format(MODEL, LAYER, NUM_EPOCH)

df = pd.read_csv(FEATURES_PATH, header=0)

X_cols = ["F" + str(i+1) for i in range(0, 2048)]
y_cols = ["L" + str(i+1) for i in range(0, 9)]

X = df[X_cols].values
y = df[y_cols].values

random_state = np.random.RandomState(2016)

# ----------setup models----------

svc = svm.SVC(
    cache_size=1000,
    kernel='linear',
    probability=True,
    random_state=random_state,
    C=0.10)
ovr_svc = OneVsRestClassifier(estimator=svc, n_jobs=1)

rfc = RandomForestClassifier(
    n_jobs=-1,
    n_estimators=8000,
    min_samples_split=2,
    random_state=random_state,
    verbose=0)
ovr_rfc = OneVsRestClassifier(estimator=rfc, n_jobs=1)

lrc = LogisticRegressionCV(
    Cs=15,
    n_jobs=-1,
    solver='lbfgs',
    random_state=random_state,
    multi_class='ovr',
    max_iter=1000,
    fit_intercept=False,
    verbose=0)
ovr_lrc = OneVsRestClassifier(estimator=lrc, n_jobs=1)

# ----------fit models----------

print('fitting')

print('svc')
ovr_svc.fit(X, y)
joblib.dump(ovr_svc, '/data/skmodels/svc_inception-v3.pkl')

print('lrc')
ovr_lrc.fit(X, y)
joblib.dump(ovr_svc, '/data/skmodels/lrc_inception-v3.pkl')

print('rfc')
ovr_rfc.fit(X, y)
joblib.dump(ovr_svc, '/data/skmodels/rfc_inception-v3.pkl')
