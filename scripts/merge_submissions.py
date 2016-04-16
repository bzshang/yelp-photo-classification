"""
Generate ensemble prediction using majority vote from prior predictions
"""

__author__ = 'bshang'

import numpy as np
import pandas as pd
from sklearn import preprocessing


def convert_label_to_array(str_label):
    if isinstance(str_label, float):
        return []
    str_label = str_label.split(' ')
    return [int(x) for x in str_label if len(x) > 0]

paths = []
paths.append('/data/submission/inception_v3_svc_rfc_lrc_epoch3.csv')
paths.append('/data/submission/inception_v3_svc_rfc_lrc_epoch4.csv')
paths.append('/data/submission/inception_v3_svc_rfc_lrc_epoch5.csv')

votes = []
for p in paths:
    print(p)
    df = pd.read_csv(p, header=0, index_col=False)
    y = np.array([convert_label_to_array(y) for y in df['labels'].values])
    mlb = preprocessing.MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    votes.append(y)

avg_votes = np.mean(np.array(votes), axis=0)

y_predict = preprocessing.binarize(avg_votes, threshold=0.5)

df = pd.read_csv(paths[0], header=0, index_col=False)
y = np.array([convert_label_to_array(y) for y in df['labels'].values])
mlb = preprocessing.MultiLabelBinarizer()
y = mlb.fit_transform(y)
y_ = mlb.inverse_transform(y_predict)
y_ = [' '.join(str(x) for x in ls) for ls in y_]

df['labels'] = pd.Series(y_, index=df.index)

df = df.sort_values('business_id')

with open('/data5/submission/maj_vote_prediction_1.csv', 'w') as f:
    df[['business_id', 'labels']].to_csv(f, index=False)

with open('/data5/submission/maj_vote_prediction_1.csv', 'w') as f:
    for p in paths:
        f.write(p + '\n')


