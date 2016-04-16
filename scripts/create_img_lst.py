"""
Create a multi-label image list file according to
http://myungjun-youn-demo.readthedocs.org/en/latest/python/io.html
"""

__author__ = 'bshang'

import numpy as np
import pandas as pd
from sklearn import preprocessing

def convert_label_to_array(str_label):
    str_label = str_label.split(' ')
    return [int(x) for x in str_label if len(x) > 0]

DATA_FOLDER = '/data/'

TEST = False

if TEST:
    pids = set()
    with open(DATA_FOLDER + 'test_photo_to_biz.csv', 'r') as f:
        next(f)
        for line in f:
            pid = str(line.split(',')[0])
            pids.add(pid)

    with open('/data/rec/test.lst', 'w') as fw:
        for i, pid in enumerate(pids):
                output = []
                pida = pid + '.jpg'
                output.append(str(i))
                output.extend([1, 0, 0, 0, 0, 0, 0, 0, 0])
                output.append(pida)
                fw.write('\t'.join(str(i) for i in output) + '\n')
else:
    df_photo2biz = pd.read_csv(DATA_FOLDER + 'train_photo_to_biz_ids.csv')
    df_photo2biz['path'] = df_photo2biz['photo_id'].apply(lambda x: str(x)+'.jpg')

    df_biz2lab = pd.read_csv(DATA_FOLDER + 'train.csv').dropna()
    df_photo2lab = pd.merge(df_photo2biz, df_biz2lab, how='inner', on='business_id', sort=False)

    y = np.array([convert_label_to_array(y) for y in df_photo2lab['labels']])
    y = preprocessing.MultiLabelBinarizer().fit_transform(y)

    dfy = pd.DataFrame(y, columns=["L" + str(i) for i in range(0, y.shape[1])])
    dfy['photo_id'] = df_photo2lab['photo_id']
    df = pd.merge(df_photo2lab, dfy, how='inner', on='photo_id', sort=False)

    cols = []
    cols.extend(["L" + str(i) for i in range(0, y.shape[1])])
    cols.append('path')

    df_shuffled = df[cols].sample(frac=1, random_state=np.random.RandomState(2017)).reset_index(drop=True)

    df_shuffled.to_csv('/data/rec/all_imgs.lst', sep='\t', header=False)
